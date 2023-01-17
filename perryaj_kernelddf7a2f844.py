import torch

import numpy as np

import pandas as pd

from torch import nn

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset

from pathlib import Path

from sklearn.model_selection import train_test_split

from torch.functional import F

import spacy

import string

import re

import numpy as np

from spacy.symbols import ORTH

from collections import Counter

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
batch_size = 100

embedding_size=50

hidden_dim=100

epochs=30

learning_rate=0.001
PATH = Path('../input/quora-question-pairs/quora-question-pairs/')
list(PATH.iterdir())
train_path = PATH/'train.csv'

val_path = PATH/'test.csv'

train = pd.read_csv(str(train_path))

train,val = train_test_split(train,test_size=0.2)

test = pd.read_csv(str(val_path))
train=train.sample(150_000)

val=val.sample(50_000)
train.dropna(inplace=True)

val.dropna(inplace=True)

test.dropna(inplace=True)
re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)

def sub_br(x):return re_br.sub("\n", x)

my_tok = spacy.load('en')

def spacy_tok(x): 



    return [tok.text for tok in my_tok.tokenizer(sub_br(x))]



#         #isnan

#         return []
train['question1']=train['question1'].apply(spacy_tok)

train['question2']=train['question2'].apply(spacy_tok)
val['question1']=val['question1'].apply(spacy_tok)

val['question2']=val['question2'].apply(spacy_tok)
counts = Counter()

for question_words in train['question1']:

    counts.update(question_words)

for question_words in train['question2']:

    counts.update(question_words)

for question_words in val['question1']:

    counts.update(question_words)

for question_words in val['question2']:

    counts.update(question_words)
len(counts)
for word in list(counts):

    if counts[word] < 3:

        del counts[word]
vocab2index = {"":0, "UNK":1}

words = ["", "UNK"]

for word in counts:

    vocab2index[word] = len(words)

    words.append(word)




# note that spacy_tok takes a while run it just once

def encode_sentence(word_list, vocab2index=vocab2index, N=embedding_size, padding_start=False):

    x = word_list

    enc = np.zeros(N, dtype=np.int32)

    enc1 = np.array([vocab2index.get(w, vocab2index["UNK"]) for w in x])

    l = min(N, len(enc1))

    if padding_start:

        enc[:l] = enc1[:l]

    else:

        enc[N-l:] = enc1[:l]

    return enc, l



train['question1']=train['question1'].apply(encode_sentence)

train['question2']=train['question2'].apply(encode_sentence)
val['question1']=val['question1'].apply(encode_sentence)

val['question2']=val['question2'].apply(encode_sentence)
num_words=len(words)

num_words
val.head()
class Question_Dataset(Dataset):

    def __init__(self,df,train):

    

        self.y = torch.Tensor(df['is_duplicate'].values)

        self.x1 = df['question1']

        self.x2 = df['question2']

        

        

    def __getitem__(self,idx):

        x1, s1 = self.x1.loc[idx]

        x2, s2 = self.x2.loc[idx]

        x1=torch.Tensor(x1)

        x2=torch.Tensor(x2)

        return({"x1":x1,'x2':x2,"s1":s1,'s2':s2,'y':self.y[idx]})

    def __len__(self):

        return len(self.y)
train.reset_index(inplace=True)

val.reset_index(inplace=True)
train_ds = Question_Dataset(train,train=True)

val_ds = Question_Dataset(val,train=True)
class Questionnaire(nn.Module):

    def __init__(self):

        super(Questionnaire,self).__init__()

        self.lstm = nn.LSTM(embedding_size, hidden_dim, batch_first=True).cuda()

        self.embedding =nn.Embedding(num_words,embedding_size, padding_idx=0)

        self.dropout = nn.Dropout(0.5)

    def forward(self,x,s):

        s, sort_index = torch.sort(s, 0,descending=True)

        s = s.long().cpu().numpy().tolist()

        x=self.embedding(x)

        x=self.dropout(x)

        x_pack = pack_padded_sequence(x.float(), list(s), batch_first=True)

        out_pack, (ht, ct) = self.lstm(x_pack)

        out=ht[-1]

        return torch.zeros_like(out).scatter_(0, sort_index.unsqueeze(1).expand(-1,out.shape[1]), out)

         
def val_metrics(model, valid_dl,eval_metric=F.nll_loss):

    model.eval()

    total = 0

    sum_loss = 0

    sum_loss2=0

    correct = 0 

    rand_int = np.random.randint(len(valid_dl),size=1)

    

    for i, input in enumerate(valid_dl):

        if i in rand_int:

            x1 = input['x1'].cuda().long()

            x2 = input['x2'].cuda().long()

            s1 = input['s1'].cuda().long()

            y = input['y'].cuda().float()



            s2 = input['s2'].cuda().long()

            y_hat_1 = model(x1,s1)

            y_hat_2 = model(x2,s2)

            DISTANCE = torch.exp(-torch.abs(y_hat_2-y_hat_1).sum(-1))

#             DISTANCE = DISTANCE.unsqueeze(1)

#             DISTANCE = torch.cat([1-DISTANCE,DISTANCE],1).float()

            xt1 = [words[int(x)] for x in x1[0]]

            xt2 = [words[int(x)] for x in x2[0]]

            loss = F.mse_loss(DISTANCE,y)



            print('Sentence 1: ',' '.join(xt1))

            print('Sentence 2:',' '.join(xt2))

            print('Prediction:',str(float(DISTANCE[0])))

            print('Actual:',str(float(y[0])))

        x1 = input['x1'].cuda().long()

        x2 = input['x2'].cuda().long()

        s1 = input['s1'].cuda().long()

        s2 = input['s2'].cuda().long()

        y_hat_1 = model(x1,s1)

        y_hat_2 = model(x2,s2)



        DISTANCE = torch.exp(-torch.abs(y_hat_2-y_hat_1).sum(-1))

        DISTANCE = DISTANCE.unsqueeze(1)

        DISTANCE = torch.cat([1-DISTANCE,DISTANCE],1).float()



        y = input['y'].cuda().long()



        loss = eval_metric(DISTANCE,y)

        batch=y.shape[0]



        sum_loss += batch*(loss.item())

        total += batch

    print("Validation Log Loss: ", sum_loss/total)

    return sum_loss/total
def train_routine(model,train_ds,valid_ds,epochs,eval_metric=F.mse_loss):

    model.train()

    optimizer = torch.optim.Adam(model.parameters(),learning_rate)

    train_dl = DataLoader(train_ds,batch_size,True)

    val_dl = DataLoader(val_ds,batch_size,True)

    valid_errors = []

    train_errors = []

    for epoch in range(epochs):

        model.train()



        sum_loss=0

        total=0

        for i, input in enumerate(train_dl):

            optimizer.zero_grad()

            x1 = input['x1'].cuda().long()

            x2 = input['x2'].cuda().long()

            s1 = input['s1'].cuda().long()

            s2 = input['s2'].cuda().long()



            y_hat_1 = model(x1,s1)

            y_hat_2 = model(x2,s2)

            DISTANCE = torch.exp(-torch.abs(y_hat_2-y_hat_1).sum(-1))

#             DISTANCE = DISTANCE.unsqueeze(1)

#             DISTANCE = torch.cat([1-DISTANCE,DISTANCE],1).float()

            

            y = input['y'].float().cuda()

            

            loss = eval_metric(DISTANCE,y)

            loss.backward()

            total+=y.shape[0]

            sum_loss+=loss.item()

            optimizer.step()

        print("Training Mean Squared error: ", sum_loss/total)

        train_errors.append(sum_loss/total)

        valid_errors.append(val_metrics(model,val_dl))

        print()

    return train_errors, valid_errors
model = Questionnaire().cuda()
model=model.cuda()
train_errors, val_errors = train_routine(model,train_ds,val_ds,epochs,eval_metric=F.mse_loss)
plt.plot(train_errors)



plt.show()

plt.plot(val_errors)

plt.show()
learning_rate=0.001

newtrain_errors, newval_errors = train_routine(model,train_ds,val_ds,epochs,eval_metric=F.mse_loss)
epochs=5

learning_rate=0.0001

newesttrain_errors, newestval_errors = train_routine(model,train_ds,val_ds,epochs)
plt.plot(train_errors+newtrain_errors+newesttrain_errors)
plt.plot(val_errors+newval_errors+newestval_errors)
test['question1']=test['question1'].apply(spacy_tok)

test['question2']=test['question2'].apply(spacy_tok)

testcounts = Counter()

for question_words in test['question1']:

    counts.update(question_words)

for question_words in test['question2']:

    counts.update(question_words)

for word in list(counts):

    if counts[word] < 3:

        del counts[word]

vocab2index = {"":0, "UNK":1}

words = ["", "UNK"]

for word in counts:

    vocab2index[word] = len(words)

    words.append(word)

test['question1']=test['question1'].apply(encode_sentence)

test['question2']=test['question2'].apply(encode_sentence)



predictions = []

for row in range(len(test)):

    x1, s1 = test['question1'][row]

    x2, s2 = test['question2'][row]

    x1=torch.Tensor(x1)

    y_hat_1 = model(x1,s1)

    y_hat_2 = model(x2,s2)

    prediction = torch.exp(-torch.abs(y_hat_2-y_hat_1).sum(-1))

    predictions.append(prediction)
my_submission = pd.DataFrame({'test_id': np.array( range(len(predictions))), 'is_duplicate':np.array( predictions)})

my_submission.to_csv('submission.csv', index=False)