import os

import torch

import torch.nn as nn

import torch.optim as optim

import numpy as np

import pandas as pa

from collections import Counter

from torch.utils.data import DataLoader, Dataset,random_split



print(os.listdir("../input"))

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
class SentimentDataset(Dataset):

    """

    This is a subclass of torch.utils.data Dataset and it implements 

    methods that make the dataset compatible with pytorch data utilities, notably the DataLoader

    """

    def __init__(self,datalines):

        self.xydata = datalines

    

    def __len__(self):              #API requirement

        return len(self.xydata)

    

    def __getitem__(self,idx):      #API requirement

        return self.xydata[idx]



def load_data_set(filename):

    """

    Loads a dataset as a list of tuples: (text,label)

    Args:

       filename (str): the dataset filename 

    Returns:

       A pytorch compatible Dataset object

       list of tuples.

    """

    istream = open(filename)

    istream.readline()#skips header

    xydataset = [ ]

    for line in istream:

        fields = line.split(',')

        label  = fields[0]

        text   = ','.join(fields[1:])

        xydataset.append( (text,label) )

    istream.close()

    return SentimentDataset(xydataset)



train_set = load_data_set('../input/sentimentIMDB_train.csv')

print('Loaded %d examples as train set. '%(len(train_set)))



#This demonstrates the DataLoader basic usage

print('Train data Sample (1st batch only)')

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)

for text_batch, label_batch in train_loader:       #iterates over all batches

    for text,label in zip(text_batch, label_batch):#iterates over example in current batch 

        print('  ',text[:50],'...',label)

    break #stops displaying once first batch 



test_set = load_data_set('../input/sentimentIMDB_test.csv')

print('Loaded %d examples as test set. '%(len(test_set)))
def  make_w2idx(dataset):

    """

    Maps words to integers

    Returns:

    A dictionary mapping words to integers

    """

    wordset = set([])

    for text,label in dataset:

        words = text.split()

        wordset.update(words)

    return dict(zip(wordset,range(len(wordset))))   



def vectorize_text(text,w2idx):

    counts = Counter(text.split())

    xvec = torch.zeros(len(w2idx))

    for word in counts:

        if word in w2idx:       #manages unk words (ignored)

            xvec[w2idx[word]] = counts[word] 

    return xvec.squeeze()



def vectorize_target(ylabel):

     return torch.tensor(float(ylabel))
class SentimentAnalyzer(nn.Module): 

    

    def __init__(self):    

        super(SentimentAnalyzer, self).__init__()

        self.reset_structure(1,1)

        

    def reset_structure(self,vocab_size, num_labels):

        self.W = nn.Linear(vocab_size, num_labels)

            

    def forward(self, text_vec):    

        return torch.sigmoid(self.W(text_vec)) #sigmoid is the logistic activation

        

    def train(self,train_set,learning_rate,epochs):

            

        self.w2idx = make_w2idx(train_set)

        self.reset_structure(len(self.w2idx),1)

            

        #remind that minimizing Binary Cross Entropy <=> minimizing NLL

        loss_func   = nn.BCELoss() 

        optimizer   = optim.SGD(self.parameters(), lr=learning_rate)

        

        train_dataset, dev_dataset = random_split(train_set, [20000, 5000])

        data_loader = DataLoader(train_dataset, batch_size=len(train_set), shuffle=True)

        max_acc     = 0

        for epoch in range(epochs):

            global_logloss = 0.0

            for Xbatch,Ybatch in data_loader: #there is a single batch,this loop does a single iteration

                for X, Y in zip(Xbatch,Ybatch): 

                    self.zero_grad()

                    xvec            = vectorize_text(X,self.w2idx)

                    yvec            = vectorize_target(Y)

                    prob            = self(xvec).squeeze()

                    loss            = loss_func(prob,yvec)

                    loss.backward()

                    optimizer.step()

                    global_logloss += loss.item()

            validation_acc = self.eval_test(dev_dataset)

            print("Epoch %d, mean cross entropy = %f, Validation accurracy : %f"%(epoch,global_logloss/len(train_set),validation_acc))

            if validation_acc >= max_acc:

                torch.save(self.state_dict(), 'sentiment_model.wt')

                max_acc = validation_acc

        self.load_state_dict(torch.load('sentiment_model.wt'))

            

    def eval_test(self,dev_set):

        

        with torch.no_grad():

            data_loader = DataLoader(dev_set, batch_size=len(dev_set), shuffle=False)

            ncorrect    = 0

            N           = 0

            for Xbatch,Ybatch in data_loader:

                for X,Y in zip(Xbatch,Ybatch):

                    xvec = vectorize_text(X,self.w2idx)

                    prob = self(xvec).squeeze()

                    if int(prob > 0.5) == int(Y) :

                        ncorrect += 1

                    N += 1

            return float(ncorrect)/float(N)

                

    def run_test(self,test_set,pred_filename):

        

        with torch.no_grad():

            data_loader = DataLoader(test_set, batch_size=len(train_set), shuffle=False)

            idxList  = []

            sentList = []

            for Xbatch,idxbatch in data_loader:

                for X,idx in zip(Xbatch,idxbatch):

                    xvec = vectorize_text(X,self.w2idx)

                    prob = self(xvec).squeeze()

                    idxList.append(idx)

                    sentList.append(int(prob > 0.5))

            df = pa.DataFrame({'idx':idxList,'sentY':sentList})

            df.to_csv(pred_filename,index=False)

            print('done.')
sent = SentimentAnalyzer()

sent.train(train_set,0.01,50)
sent.run_test(test_set,'submission.csv')