import torch

import numpy as np
file_object = open(r'../input/kanye_verses.txt',mode='r',encoding='utf8')

raps = file_object.read().replace('\n'," \\n ")

raps = raps.replace(',',' ,')

raps = " ".join(raps.split())
import unicodedata

import string
characters = string.ascii_letters + " .,;'\\"

n_characters = len(characters)



def unicodeToAscii(s):

    return ''.join(

        c for c in unicodedata.normalize('NFD', s)

        if unicodedata.category(c) != 'Mn'

        and c in characters+'\n'

    )

raps = unicodeToAscii(raps)
def letter_encode(letter):

    x = np.zeros((1,n_characters))

    x[0][characters.index(letter)] = 1

    return x



def word_encode(word):

    return (words_arr == word).astype(np.float).reshape(1,-1)
words_arr = np.array(list(set(raps.split())))

n_words = len(words_arr)
# input is (batch,seq_len,input_size)

X_train = torch.FloatTensor([word_encode(x) for x in raps.split()])

y_train = X_train[1:].argmax(dim=2).long()

X_train = X_train[:-1]
input_size = n_words

hidden_size = 256 

sequence_length = 1 # character by character

num_layers = 1 # one-layer rnn

output_size = n_words
import torch.nn as nn
class Model(nn.Module):

    def __init__(self):

        super(Model,self).__init__()

        

        self.cell = nn.GRU(input_size=input_size,

                          hidden_size=hidden_size,

                          num_layers=num_layers,

                          batch_first=True)

        

        self.fc = nn.Linear(hidden_size,output_size)

        

#         self.encoder = nn.Embedding(input_size, hidden_size)

        

    def forward(self,input):

        batch_size = input.size()[0]

        hidden = self._init_hidden(batch_size)

        output,hidden = self.cell(input,hidden)

        fc_output = self.fc(output.view(-1,hidden_size))

        return fc_output

    

    def forward2(self,input,hidden):

        batch_size = input.size()[0]

        output,hidden = self.cell(input,hidden)

        fc_output = self.fc(output.view(-1,hidden_size))

        return fc_output,hidden

    

    def _init_hidden(self,batch_size):

        hidden = torch.zeros(num_layers,batch_size,hidden_size)

        return hidden
rnn = Model()

indexer = rnn.forward((X_train[0:64])).argmax(dim=1)
import torch.optim as optim
optimizer = optim.Adam(rnn.parameters(),lr=.005)

loss_fn = nn.CrossEntropyLoss()
chunk_len = 200

def random_chunk():

    start_index = torch.randint(0,X_train.size()[0]-chunk_len,(1,1))

    end_index = start_index + chunk_len + 1

    char_chunk = X_train[start_index:end_index]

    return char_chunk[:-1].permute(1,0,2),char_chunk[1:].argmax(dim=2).squeeze()
import torch.nn.functional as F
def evaluate():

    string = ""

    

    start_letter = words_arr[torch.randint(0,n_words,(1,1))]

    hidden = rnn._init_hidden(1)

    with torch.no_grad():

        for i in range(chunk_len-1):

            letter,hidden = rnn.forward2(torch.FloatTensor(word_encode(start_letter)).unsqueeze(0),hidden)

            letter = words_arr[torch.multinomial(F.softmax(letter.view(1,-1)),1)]

            start_letter = str(letter)

            string += (" " + letter)

    

    print(string.replace('\\n','\n'))
rap_length = 50

def rap_lyrics():

    string = ""

    

    start_letter = words_arr[torch.randint(0,n_characters,(1,1))]

    hidden = rnn._init_hidden(1)

    with torch.no_grad():

        for i in range(rap_length -1):

            

            letter,hidden = rnn.forward2(torch.FloatTensor(word_encode(start_letter)).unsqueeze(0),hidden)

            if (i+1)%2 == 0:

                letter = words_arr[torch.multinomial(F.softmax(letter.view(1,-1)),1)]

            else:

                letter = words_arr[letter.view(1,-1).argmax(1)]

            start_letter = str(letter)

            string += (" " + letter)

    

    print(string.replace('\\n','\n'))
epochs = 4000

for epoch in range(epochs):

    epoch_loss = 0

    xb,yb = random_chunk()

    output = rnn(xb)

    loss = loss_fn(output,yb)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if (epoch+1) % 50 == 0:

        print()

        print(epoch+1,'/',epochs," Loss: ",loss.item(),sep='')

        print()        

#         evaluate()

        rap_lyrics()
# write a wrap dog

rap_length = 1000



rap_lyrics()