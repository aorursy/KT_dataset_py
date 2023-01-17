%matplotlib inline

import numpy as np 

import pandas as pd 

import torch

import torchtext

from torchtext import data

import spacy

import os

import re





os.environ['OMP_NUM_THREADS'] = '4'

my_tok = spacy.load('en')

my_stopwords = spacy.lang.en.stop_words.STOP_WORDS

my_stopwords.update(['wikipedia','article','articles','im','page'])



def spacy_tok(x):

    x= re.sub(r'[^a-zA-Z\s]','',x)

    x= re.sub(r'[\n]',' ',x)

    return [tok.text for tok in my_tok.tokenizer(x)]



# print(spacy_tok("I solve Kaggle"))

# print(spacy_tok("I solve Kaggle5"))

# print(spacy_tok("I slove Kagle 5 43,...."))



TEXT = data.Field(lower=True, tokenize=spacy_tok,eos_token='EOS',stop_words=my_stopwords,include_lengths=True)

LABEL = data.Field(sequential=False, 

                         use_vocab=False, 

                         pad_token=None, 

                            unk_token=None)



dataFields = [("id", None),

                 ("comment_text", TEXT), ("toxic", LABEL),

                 ("severe_toxic", LABEL), ("threat", LABEL),

                 ("obscene", LABEL), ("insult", LABEL),

                 ("identity_hate", LABEL)]



dataset= data.TabularDataset(path='../input/train.csv', 

                                            format='csv',

                                            fields=dataFields, 

                                            skip_header=True)
train,val= dataset.split()
TEXT.build_vocab(train,vectors='fasttext.simple.300d')
traindl, valdl = torchtext.data.BucketIterator.splits(datasets=(train, val),

                                            batch_sizes=(128,1024),

                                            sort_key=lambda x: len(x.comment_text),

                                            device=torch.device('cuda:0'),

                                            sort_within_batch=True

                                                     )
vectors= train.fields['comment_text'].vocab.vectors.cuda()
class BatchGenerator:

    def __init__(self, dl):

        self.dl = dl

        self.yFields= ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

        self.x= 'comment_text'

        

    def __len__(self):

        return len(self.dl)

    

    def __iter__(self):

        for batch in self.dl:

            X = getattr(batch, self.x)

            y = torch.transpose( torch.stack([getattr(batch, y) for y in self.yFields]),0,1)

            yield (X,y)
import torch

import torch.nn as nn

import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
class MyModel(nn.Module):

    def __init__(self,op_size,n_tokens,pretrained_vectors,nl=2,bidirectional=True,emb_sz=300,n_hiddenUnits=100):

        super(MyModel, self).__init__()

        self.n_hidden= n_hiddenUnits

        self.embeddings= nn.Embedding(n_tokens,emb_sz)

        self.embeddings.weight.data.copy_(pretrained_vectors)

#         self.embeddings.weight.requires_grad = False

        self.rnn= nn.LSTM(emb_sz,n_hiddenUnits,num_layers=2,bidirectional=True,dropout=0.2)

        self.lArr=[]

        if bidirectional:

            n_hiddenUnits= 2* n_hiddenUnits

        self.bn1 = nn.BatchNorm1d(num_features=n_hiddenUnits)

        for i in range(nl):

            if i==0:

                self.lArr.append(nn.Linear(n_hiddenUnits*3,n_hiddenUnits))

            else:

                self.lArr.append(nn.Linear(n_hiddenUnits,n_hiddenUnits))

        self.lArr= nn.ModuleList(self.lArr)

        self.l1= nn.Linear(n_hiddenUnits,op_size)

        

    def forward(self,data,lengths):

        torch.cuda.empty_cache()

        bs= data.shape[1]

        self.h= self.init_hidden(bs)

        embedded= self.embeddings(data)

        embedded= nn.Dropout()(embedded)

#         embedded = pack_padded_sequence(embedded, torch.as_tensor(lengths))

        rnn_out, self.h = self.rnn(embedded, (self.h,self.h))

#         rnn_out, lengths = pad_packed_sequence(rnn_out,padding_value=1)

        avg_pool= F.adaptive_avg_pool1d(rnn_out.permute(1,2,0),1).view(bs,-1)

        max_pool= F.adaptive_max_pool1d(rnn_out.permute(1,2,0),1).view(bs,-1)

        ipForLinearLayer= torch.cat([avg_pool,max_pool,rnn_out[-1]],dim=1)

        for linearlayer in self.lArr:

            outp= linearlayer(ipForLinearLayer)

            ipForLinearLayer= self.bn1(F.relu(outp))

            ipForLinearLayer= nn.Dropout(p=0.6)(ipForLinearLayer)

        outp = self.l1(ipForLinearLayer)

        del embedded;del rnn_out;del self.h;

        torch.cuda.empty_cache()

        return outp

        

    def init_hidden(self, batch_size):

        return torch.zeros((4,batch_size,self.n_hidden),device="cuda:0")
def getValidationLoss(valdl,model,loss_func):

    model.eval()

    runningLoss=0

    valid_batch_it = BatchGenerator(valdl)

    allPreds= []

    allActualPreds= []

    with torch.no_grad():

        for i,obj in enumerate(valid_batch_it):

            obj= ( (obj[0][0].cuda(),obj[0][1].cuda()),obj[1] )

            preds = model(obj[0][0],obj[0][1])

            loss = loss_func(preds,obj[1].float())

            runningLoss+= loss.item()

            allPreds.append(preds.detach().cpu().numpy())

            allActualPreds.append(obj[1].detach().cpu().numpy())

        rocLoss= roc_auc_score(np.vstack(allActualPreds),np.vstack(allPreds))

        return runningLoss/len(valid_batch_it),rocLoss
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

def oneEpoch(lr):

    train_batch_it = BatchGenerator(traindl)

    opt = optim.Adam(model.parameters(),lr)

    runningLoss= 0

    allPreds=[]

    allActualPreds=[]

    for i,obj in enumerate(train_batch_it):

        obj= ( (obj[0][0].cuda(),obj[0][1].cuda()),obj[1] )

        model.train()

        opt.zero_grad()

        preds = model(obj[0][0],obj[0][1])

        loss = loss_func(preds,obj[1].float())

        runningLoss+= loss.item()

        loss.backward()

        opt.step()

        allPreds.append(preds.detach().cpu().numpy())

        allActualPreds.append(obj[1].detach().cpu().numpy())

        del obj;del preds

    trainRocLoss= roc_auc_score(np.vstack(allActualPreds),np.vstack(allPreds))

    runningLoss= runningLoss/len(train_batch_it)

    valLoss,valRocLoss= getValidationLoss(valdl,model,loss_func)

    torch.cuda.empty_cache()

    return runningLoss,valLoss,trainRocLoss,valRocLoss
epochs= 10

trainLossArr=[]

valLossArr=[]

rocTrainLoss=[]

rocValLoss=[]

model= MyModel(6,len(TEXT.vocab),vectors,1)

loss_func= torch.nn.BCEWithLogitsLoss()

model = model.cuda()

for i in range(epochs):

    %time tLoss,vLoss,tRocLoss,vRocLoss= oneEpoch(1e-4)

    print(f"Epoch - {i}")

    print(f"Train Loss - {tLoss} vs Val Loss is {vLoss}")

    print(f"Train ROC - {tRocLoss} vs Val ROC is {vRocLoss}")

    trainLossArr.append(tLoss)

    valLossArr.append(vLoss)

    rocTrainLoss.append(tRocLoss)

    rocValLoss.append(vRocLoss)
import matplotlib.pyplot as plt 

plt.plot(trainLossArr,color='b')

plt.plot(valLossArr,color='g')

plt.plot(rocTrainLoss,color='r')

plt.plot(rocValLoss,color='c')

plt.show()
torch.save(model.state_dict(), "myFirstModel1")
dataFields = [("id", None),

                 ("comment_text", TEXT)

             ]



testDataset= data.TabularDataset(path='../input/test.csv', 

                                            format='csv',

                                            fields=dataFields, 

                                            skip_header=True)
test_iter1 = torchtext.data.Iterator(testDataset, batch_size=32, device=torch.device('cuda:0'), sort=False, sort_within_batch=False, repeat=False,shuffle=False)
testDF= pd.read_csv("../input/test.csv")
myPreds=[]

with torch.no_grad():

    model.eval()

    for obj in test_iter1:

#         print(torch.transpose(obj.comment_text[0],0,1)[:10].shape)

#         text= torch.transpose(obj.comment_text[0],0,1)[:2]

#         for t in text:

#             print( ' '.join([TEXT.vocab.itos[i] for i in t]) )

#         break

        torch.cuda.empty_cache()

        pred= model(obj.comment_text[0],obj.comment_text[1])

        pred= torch.sigmoid(pred)

        myPreds.append(pred.cpu().numpy())

        del pred;del obj;

        torch.cuda.empty_cache()
myPreds= np.vstack(myPreds)
for i, col in enumerate(["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]):

    testDF[col] = myPreds[:, i]
testDF.drop("comment_text", axis=1).to_csv("submission.csv", index=False)