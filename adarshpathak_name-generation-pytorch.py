DATA_DIR = "/kaggle/input/name-corpus-english-for-nlp-task/names.txt"
import torch
import torch.nn as nn
import random
import string
import sys
from tqdm import tqdm_notebook as tqdm
import unidecode

import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
device
all_chars = string.printable
n_char = len(all_chars)
file = unidecode.unidecode(open(DATA_DIR).read())
class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size):
        super(RNN,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
    
        self.embed = nn.Embedding(self.input_size,self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size,self.hidden_size,num_layers=self.num_layers,batch_first = True)
        self.fc = nn.Linear(self.hidden_size,self.output_size)
        
    def forward(self,x,h,c):
        x = self.embed(x)
        x,(h,c) = self.lstm(x.unsqueeze(1),(h,c))
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        return x,(h,c)
    
    def init_hidden(self,batch_size):
        h = torch.zeros(self.num_layers ,batch_size,self.hidden_size).to(device)
        c = torch.zeros(self.num_layers ,batch_size,self.hidden_size).to(device)
        return h,c
class Generator:
    def __init__(self):
        self.chunk_len = 250
        self.num_epochs = 250
        self.batch_size = 1
        self.print_every = 10
        self.hidden_size = 1024
        self.num_layers = 2
        self.lr = 1e-2
    
    def char2tensor(self,string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = all_chars.index(string[c])
        return tensor
    
    def get_random_batches(self):
        start_dx = random.randint(0,len(file)-self.chunk_len)
        end_idx = start_dx + self.chunk_len +1
        text_str = file[start_dx:end_idx]
        text_input = torch.zeros(self.batch_size , self.chunk_len)
        text_target = torch.zeros(self.batch_size , self.chunk_len)
        for i in range(self.batch_size):
            text_input[i,:] = self.char2tensor(text_str[:-1])
            text_target[i,:] = self.char2tensor(text_str[1:])
        return text_input.long(),text_target.long()
    
    def generate(self, initial_str="A", predict_len=100, temperature=0.85):
        hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
        initial_input = self.char2tensor(initial_str)
        predicted = initial_str

        for p in range(len(initial_str) - 1):
            _, (hidden, cell) = self.rnn(
                initial_input[p].view(1).to(device), hidden, cell
            )

        last_char = initial_input[-1]

        for p in range(predict_len):
            output, (hidden, cell) = self.rnn(
                last_char.view(1).to(device), hidden, cell
            )
            output_dist = output.data.view(-1).div(temperature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]
            predicted_char = all_chars[top_char]
            predicted += predicted_char
            last_char = self.char2tensor(predicted_char)

        return predicted

    def train(self):
        self.rnn = RNN(input_size=n_char,
                      hidden_size=self.hidden_size,
                      num_layers=self.num_layers,
                      output_size=n_char).to(device)
        
        optimizer = torch.optim.Adam(self.rnn.parameters(),lr = self.lr)
        criterian = nn.CrossEntropyLoss()
        
        print("=> Start Training")
        
        for epoch in tqdm(range(1,self.num_epochs+1)):
            txt,target = self.get_random_batches()
            h,c = self.rnn.init_hidden(self.batch_size)
            
            self.rnn.zero_grad()
            loss = 0
            txt = txt.to(device)
            target = target.to(device)
            
            for j in range(self.chunk_len):
                output,(h,c) = self.rnn(txt[:,j],h,c)
                loss += criterian(output,target[:,j])
            loss.backward()
            optimizer.step()
            loss = loss.item()/self.chunk_len
            
            if epoch%self.print_every == 0:
                print(f"loss: {loss}")
                print(self.generate())
gen = Generator()
gen.train()