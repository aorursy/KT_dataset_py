DATA_DIR = '/kaggle/input/english-to-german-dataset/deu.txt'
NOTEBOOK_VERSION = 1
import torch 
import numpy as np
import torch.nn as nn
from tqdm.notebook import tqdm
import re
import random
from torch.utils.data.sampler import SubsetRandomSampler
with open(DATA_DIR) as fp:
    sentences = fp.readlines()
eng_sentences , deu_sentences = [],[]
eng_words,deu_words = set(),set()
MAX_SENTENCE_LEN = 16
for i in tqdm(range(25000)):
    idx = np.random.randint(len(sentences))
    eng_sent , deu_sent = ['<SOS>'],['<SOS>']
    eng_sent += re.findall(r'\w+',sentences[idx].split('\t')[0])
    deu_sent += re.findall(r'\w+',sentences[idx].split('\t')[1])
    
    eng_sent = [x.lower() for x in eng_sent]
    deu_sent = [x.lower() for x in deu_sent]
    
    if len(eng_sent) >= MAX_SENTENCE_LEN:
        eng_sent = eng_sent[:MAX_SENTENCE_LEN]
    else:
        for _ in range(MAX_SENTENCE_LEN - len(eng_sent)):
            eng_sent.append('<PAD>')
            
    if len(deu_sent) >= MAX_SENTENCE_LEN:
        deu_sent = deu_sent[:MAX_SENTENCE_LEN]
    else:
        for _ in range(MAX_SENTENCE_LEN - len(deu_sent)):
            deu_sent.append('<PAD>')
            
    eng_sentences.append(eng_sent)
    deu_sentences.append(deu_sent)
    
    eng_words.update(eng_sent)
    deu_words.update(deu_sent)
eng_words = list(eng_words)
deu_words = list(deu_words)
for i in tqdm(range(len(eng_sentences))):
    eng_sentences[i] = [eng_words.index(i) for i in eng_sentences[i]]
    deu_sentences[i] = [deu_words.index(i) for i in deu_sentences[i]]
print(eng_words[:5])
print(deu_words[:5])
print(eng_sentences[10])
print(deu_sentences[10])
class DataSet(torch.utils.data.Dataset):
    def __init__(self):
        self.source = np.array(eng_sentences,dtype = int)
        self.target = np.array(deu_sentences,dtype = int)
        
    def __getitem__(self,idx):
        return self.source[idx],self.target[idx]
    
    def __len__(self):
        return len(self.source)
np.random.seed(777)
BATCH_SIZE = 128
dataset = DataSet()
NUM_INSTANCES = len(dataset)
TEST_RATIO = 0.3
TEST_SIZE = int(NUM_INSTANCES * 0.3)
indices = list(range(NUM_INSTANCES))

test_idx = np.random.choice(indices, size = TEST_SIZE, replace = False)
train_idx = list(set(indices) - set(test_idx))
train_sampler, test_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(test_idx)
train_loader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, sampler = train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, sampler = test_sampler)
class Encoder(nn.Module):
    def __init__(self,input_size,embed_dims,hidden_size,num_layers,p):
        super(Encoder,self).__init__()
        self.input_size = input_size
        self.embed_dims = embed_dims
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        
        self.embed = nn.Embedding(self.input_size,self.embed_dims)
        self.lstm = nn.LSTM(self.embed_dims,self.hidden_size,self.num_layers,dropout = p)
        
    def forward(self,x):
        x = self.dropout(self.embed(x))
        x,(h,c) = self.lstm(x)
        
        return h,c
class Decoder(nn.Module):
    def __init__(self,input_size,embed_dims,hidden_size,num_layers,p):
        super(Decoder,self).__init__()
        self.input_size = input_size
        self.embed_dims = embed_dims
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        
        self.embed = nn.Embedding(self.input_size,self.embed_dims)
        self.lstm = nn.LSTM(self.embed_dims,self.hidden_size,self.num_layers,dropout = p)
        self.fc = nn.Linear(self.hidden_size,self.input_size)
        
    def forward(self,x,hidden,cell):
        x = x.unsqueeze(0)
        x = self.dropout(self.embed(x))
        out,(hidden,cell) = self.lstm(x,(hidden,cell))
        out = self.fc(out.squeeze(0))
        #print(out.shape)
        #out = out.squeeze(0)
        return out,hidden,cell
class seq2seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(seq2seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self,source,target,teachers_force_ratio = 0.5):
        batch_size = source.shape[1]
        seq_len = target.shape[0]
        target_vocab_len = len(deu_words)
        hidden,cell = self.encoder(source)
        outputs = torch.zeros(seq_len,batch_size,target_vocab_len).to(device)
        x = target[0]
        for t in range(1,seq_len):
            output,hidden,cell = self.decoder(x,hidden,cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teachers_force_ratio else best_guess
        return outputs
enc_input_size = len(eng_words)
dec_input_size = len(deu_words)

enc_embed_dims = 300
dec_embed_dims = 300

enc_hidden_size = 1024
dec_hidden_size = 1024

enc_num_layers = 2
dec_num_layers = 2

p = 0.2

EPOCHS = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
encoder = Encoder(enc_input_size,enc_embed_dims,enc_hidden_size,enc_num_layers,p).to(device)
decoder = Decoder(dec_input_size,dec_embed_dims,dec_hidden_size,dec_num_layers,p).to(device)

model = seq2seq(encoder,decoder).to(device)
criterian = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 1e-2)
losses = []
curr_loss = 0
for epoch in tqdm(range(EPOCHS)):
    model.train()
    for data,target in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        outputs = model.forward(data,target)
        loss = criterian(outputs.resize(outputs.size(0) * outputs.size(1),
                                        outputs.size(-1)), target.resize(target.size(0) * target.size(1)))
        loss.backward()
        optimizer.step()
        curr_loss += loss.item()
    print(f'epochs: [{epoch}/{EPOCHS}] loss: {loss.item()}')
    losses.append(curr_loss)
torch.save(model.state_dict(),'model.pth')
torch.save(optimizer.state_dict(),'optimizer.pth')
from torchtext.data.metrics import bleu_score
predictions = []
for i, (x,y) in enumerate(test_loader):
    with torch.no_grad():
        x, y  = x.to(device), y.to(device)
        outputs = model(x, y)
        for output in outputs:
            _, indices = output.max(-1)
            predictions.append(indices.detach().cpu().numpy())
idx = 14
print([eng_words[i] for i in eng_sentences[idx]])
print([deu_words[i] for i in predictions[idx]])
