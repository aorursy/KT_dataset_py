import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



import torch

import torch.nn as nn

import torch.optim as optim

from torchtext.data import Field,ReversibleField,TabularDataset,Iterator



def tokenize(text):

    return text.split()



def proc_float(value):

    return float(value)



def proc_int(value):

    return int(value)



TEXT      = Field(sequential=True, tokenize=tokenize) #might alternatively specify cuda data types to get the dataset to live permanently on the GPU

FLOAT     = Field(sequential=False, use_vocab=False,dtype=torch.float,preprocessing=proc_float) 

INTEGER   = Field(sequential=False, use_vocab=False,preprocessing=proc_int)



df         = TabularDataset("../input/SICK_train_logistic.txt","tsv",skip_header=True,\

                            fields=[('idx',INTEGER),('sentA',TEXT),('sentB',TEXT),('Relatedness',FLOAT)])

df_train,df_dev  = df.split(split_ratio=0.8)

TEXT.build_vocab(df_train)



#Prints out the first few lines of the dataset

for elt in df_dev[:10]: #prints out the ten first examples

    print(elt.idx,' '.join(elt.sentA),'||',' '.join(elt.sentB),elt.Relatedness)
class ParaphraseClassifier(nn.Module):

    

    def __init__(self,hidden_dim,embedding_dim):

        

        super(ParaphraseClassifier, self).__init__()

        

        self.hidden_dim    = hidden_dim

        self.embedding_dim = embedding_dim

        self.embedding     = nn.Embedding(len(TEXT.vocab), embedding_dim)

        self.bilstm        = nn.LSTM(embedding_dim, hidden_dim, num_layers=1,bidirectional=True)

        self.W             = nn.Linear(hidden_dim*4,1)    

        

        

    def forward(self,xinputA,xinputB):

        """

        Args: 

            xinputA is a sequence of word indexes

            xinputB is a sequence of word indexes

        The forward method also works for batched input.        

        """

        ##details for dimensionalities 

        #embeddings 

        #  input : batch_size x seq_length

        #  output: batch-size x seq_length x embedding_dimension

        #lstm 

        #  input : seq_length x batch_size x embedding_size

        #  output: seq_length x batch_size x hidden_size  (for the sequence)

        #  output: batch_size x hidden_size (for the last hidden/cell state)

        xembeddedA                       = self.embedding(xinputA)                   #catches embedding vectors

        lstm_outA, (hiddenA,cellA)       = self.bilstm(xembeddedA.view(len(xinputA), -1, self.embedding_dim), None) #-1 is a wildcard (here we let pytorch guess batch size)

        

        xembeddedB                       = self.embedding(xinputB)                   #catches embedding vectors

        lstm_outB, (hiddenB,cellB)       = self.bilstm(xembeddedB.view(len(xinputB), -1, self.embedding_dim), None)

        

        #concat sentence representations

        hA = hiddenA.view(-1,self.hidden_dim * 2) #-1 is a wildcard (here we let pytorch guess batch size)

        hB = hiddenB.view(-1,self.hidden_dim * 2) 

        H  = torch.cat((hA,hB),1)

        return torch.sigmoid(self.W(H)) #sigmoid is the logistic function

    

    def train(self,train_set,dev_set,epochs,learning_rate=0.001):

        

        loss_func  = nn.BCELoss() 

        optimizer  = optim.Adam(self.parameters(), lr=learning_rate)

        

        train_iterator   = Iterator(train_set, batch_size=1, device=-1, sort=False, sort_within_batch=False, repeat=False)

        

        for e in range(epochs):

            idx = 0

            for batch in train_iterator: 

                xvecA,xvecB,yRelness = batch.sentA,batch.sentB,batch.Relatedness

                self.zero_grad()

                prob            = self.forward(xvecA,xvecB).squeeze()

                loss            = loss_func(prob,yRelness)

                loss.backward()

                optimizer.step()

                print(idx)

                idx += 1

                

pc = ParaphraseClassifier(100,100)

pc.train(df_train,df_dev,3)