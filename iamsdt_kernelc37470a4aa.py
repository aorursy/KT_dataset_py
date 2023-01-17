# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



from torchtext import data,datasets



import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.autograd import Variable

import torch

import sys
table = pd.read_csv("../input/imdb.csv", error_bad_lines=False)

table.size
table.columns
table.iloc[0:2]
table.iloc[0:10, 1:10]
from torchtext import data



Text = data.Field(lower=True, batch_first=True, fix_length=20)

Label = data.Field(sequential=False)
from torchtext import datasets

#train, test = datasets.IMDB.splits(Text, Label)
print(len(train))

#train.fields
#vars(train[0])
from torchtext.vocab import GloVe,FastText,CharNGram

#Text.build_vocab(train, vectors=GloVe(name='6B', dim=300),max_size=10000,min_freq=10)

#Label.build_vocab(train,)
#Text.vocab.freqs
#Text.vocab.vectors
#Text.vocab.stoi
#train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=128, device='cuda', shuffle=True)
#batch = next(iter(train_iter))
#batch.text
#batch.label
class EmbNet(nn.Module):

    def __init__(self,emb_size,hidden_size1,hidden_size2=200):

        super().__init__()

        self.embedding = nn.Embedding(emb_size,hidden_size1)

        self.fc = nn.Linear(hidden_size2,3)

        

    def forward(self,x):

        embeds = self.embedding(x).view(x.size(0),-1)

        out = self.fc(embeds)

        return F.log_softmax(out,dim=-1)
#model = EmbNet(len(Text.vocab.stoi),10)

#model = model.cuda()
#optimizer = optim.Adam(model.parameters(),lr=0.001)
def fit(epoch,model,data_loader,phase='training',volatile=False):

    if phase == 'training':

        model.train()

    if phase == 'validation':

        model.eval()

        volatile=True

    running_loss = 0.0

    running_correct = 0

    for batch_idx , batch in enumerate(data_loader):

        text , target = batch.text , batch.label

        text,target = text.cuda(),target.cuda()

        

        if phase == 'training':

            optimizer.zero_grad()

        output = model(text)

        loss = F.nll_loss(output,target)

        

        running_loss += F.nll_loss(output,target,size_average=False).item()

        preds = output.data.max(dim=1,keepdim=True)[1]

        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()

        if phase == 'training':

            loss.backward()

            optimizer.step()

    

    loss = running_loss/len(data_loader.dataset)

    accuracy = 100. * running_correct/len(data_loader.dataset)

    

    #print('{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')

    print("Phase: {}".format(phase),

         "\tLoss is: {}".format(loss),

         "\tAccuray: {}".format(accuracy))

    

    return loss,accuracy
train_losses , train_accuracy = [],[]

val_losses , val_accuracy = [],[]



for epoch in range(1,1):



    epoch_loss, epoch_accuracy = fit(epoch,model,train_iter,phase='training')

    val_epoch_loss , val_epoch_accuracy = fit(epoch,model,test_iter,phase='validation')

    train_losses.append(epoch_loss)

    train_accuracy.append(epoch_accuracy)

    val_losses.append(val_epoch_loss)

    val_accuracy.append(val_epoch_accuracy)