import os



files=[]

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))
files
# !pip install kaggle
# !mkdir .kaggle
# !mkdir ~/.kaggle
# import json

# token = {'username':'','key':''}

# with open('/content/.kaggle/kaggle.json', 'w') as file:

#     json.dump(token, file)
# !cp /content/.kaggle/kaggle.json ~/.kaggle/kaggle.json
# !kaggle config set -n path -v{/content}
# !chmod 600 /root/.kaggle/kaggle.json
# !kaggle datasets list
# !kaggle datasets list -s sentiment
# !kaggle competitions download -c nlp-hack -p /content
# !unzip \*.zip
import numpy as np

import pandas as pd

import os

# files=['train.csv','test.csv','sample-submission.csv']
import numpy as np

import pandas as pd

import os

import string

import re

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

from torch.nn import functional as F

from torch.autograd import Variable

import torch.utils.data

from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences

from collections import Counter

import pickle
device = 'cuda' if torch.cuda.is_available() else 'cpu'

device
train = pd.read_csv(files[0],index_col=0)

test = pd.read_csv(files[1],index_col=0)

train.head()
len(train), len(test)
# pd.set_option('display.max_colwidth',-1)

# train['text'][:10]
# text = [text for text in train['text']]

# text[:3]
# Str = """aysha   \tkamal



# also this is two new\x0b\x0clines"""

# "".join([replacements.get(c,c) for c in Str])
# import re



# hum = re.compile("([\'!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~\\\])")
# Str = "\\\\\\\\\\\\"

# x = hum.sub(r' \1 ',Str);

# print(x)
# Str = "aysha aren't you the                 sweetest?+,-./:;text<=lol>?@[\kamal"

# x = hum.sub(r' \1 ',Str);

# print(x)
# re_apos = re.compile(r"n ' t ")    # n't

# re_bpos = re.compile(r" ' s ")     # 's

# re_mult_space = re.compile(r"  *") # replace multiple spaces with just one



# sent = re_apos.sub(r" n't ", x)

# sent = re_bpos.sub(r" 's ", sent)

# sent = re_mult_space.sub(' ', sent) ; sent
# print(sent)
# sent="AYSHA kamal Tanny TANny"

# ret = ['xbos']

# for w in sent.split():

#     if w.isupper():

#         ret.append('xwup')

#     elif w[0].isupper():

#         ret.append('xup')

#     ret.append(w.lower())

# ret
### All tokenizations combined

re_punc = re.compile("([\'!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~\\\])") # add spaces around punctuation

## () for capturing group, [] for one of the groups in the braces

re_apos = re.compile(r"n ' t ")    # n't

re_bpos = re.compile(r" ' s ")     # 's

re_mult_space = re.compile(r"  *") # replace multiple spaces with just one

replacements = {'\t':' ','\n':' ','\r':' ','\x0b':' ','\x0c':' '}



def simple_toks(sent):

    sent = "".join([replacements.get(c,c) for c in sent])

    sent = re_punc.sub(r" \1 ", sent) # \1 is the group that we have captured

    sent = re_apos.sub(r" n't ", sent)

    sent = re_bpos.sub(r" 's ", sent)

    sent = re_mult_space.sub(' ', sent)

    ret = ['xbos']

    for w in sent.split():

        if w.isupper():

            ret.append('xwup')

        elif w[0].isupper():

            ret.append('xup')

        ret.append(w.lower())

    return ret
# textlist = train['text'].apply(simple_toks).tolist()

# with open("tokenized.txt", "wb") as fp:   #Pickling

#     pickle.dump(textlist, fp)
# with open("tokenized.txt", "rb") as fp:   # Unpickling

#     textlist = pickle.load(fp)
# from collections import Counter

# full_vocab = Counter([w for row in textlist for w in row])
# ## 192 total characters

# len(full_vocab)
# itos = ['xunk']+[word for word,cnt in full_vocab.most_common(60000)]

# len(itos)
# stoi = {w:i for i,w in enumerate(itos)}

# len(stoi)
# with open("itos.txt", "wb") as fp:   #Pickling

#     pickle.dump(itos, fp)



# with open("stoi.txt", "wb") as fp:   #Pickling

#     pickle.dump(stoi, fp)
with open(files[3], "rb") as fp:   # Unpickling

    itos = pickle.load(fp)
len(itos)
# ## debug time

# original = textlist

# textlist = textlist[:10000]

# len(original), len(textlist)
# textlistfin=[[w if w in itos else itos[0] for w in sent] for sent in textlist]

# len(textlistfin), len(textlist)
# ## testing

# textlist_test = train['text'][:10].apply(simple_toks).tolist()

# nums_test = [[stoi[w] if w in itos else 0 for w in sent] for sent in textlist_test]
# nums = [[stoi[w] if w in itos else 0 for w in sent] for sent in textlist] ##so slow tho, NEED a way to parallel process

# len(nums)
# with open("tokenized_nums.txt", "wb") as fp:   #Pickling

#     pickle.dump(nums, fp)
with open(files[5], "rb") as fp:   # Unpickling

    nums = pickle.load(fp)
len(nums)
labels = train['target']/4
# ## debug time

# originalLabel = labels

# labels = labels[:10000]

# len(originalLabel), len(labels)
type(labels), labels.shape
labels = np.array(labels).reshape(-1,1)
max_length=50
padded = pad_sequences(nums, maxlen=max_length, padding='post', truncating='post', value=0)
padded.shape
xtrain, xtest, ytrain, ytest = train_test_split(padded,labels,test_size = 0.1,random_state = 42)
xTrain = torch.from_numpy(xtrain).type(torch.LongTensor).to(device)

yTrain = torch.from_numpy(ytrain).type(torch.FloatTensor).to(device)



xTest = torch.from_numpy(xtest).type(torch.LongTensor).to(device)

yTest = torch.from_numpy(ytest).type(torch.FloatTensor).to(device)
xTrain.shape, yTrain.shape, xTest.shape, yTest.shape
xTrain.dtype, yTrain.dtype, xTest.dtype, yTest.dtype
xTrain.is_cuda
batch_size = 100

num_epochs = 20



# dataset

train = torch.utils.data.TensorDataset(xTrain,yTrain)

test = torch.utils.data.TensorDataset(xTest,yTest)



# data loader

train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)

test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = True)
class memodel(nn.Module):

    def __init__(self, vocabsz, mxlen):

        super().__init__()

        self.vocab_size=vocabsz

        self.mx_len=mxlen

        ##config

        self.embd_size=100

        self.hidden_dim=50

        self.hidden_dim2=512

        self.layer_dim=2

        self.output_dim=1

        self.drp=0.8

        ##layers

        self.embeddings = nn.Embedding(self.vocab_size, self.embd_size)

        ##when using pretained

#         self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)

        self.lstm = nn.LSTM(input_size = self.embd_size,

                            hidden_size = self.hidden_dim,

                            num_layers = self.layer_dim,

                            dropout = self.drp, #keep

                            bidirectional = True)

        self.dropout = nn.Dropout(self.drp)

        self.fc = nn.Linear(2*self.hidden_dim*self.mx_len,self.hidden_dim2) #because bilinear

        self.fc2=nn.Linear(self.hidden_dim2,256)

        self.fc3=nn.Linear(256,64)

        self.out=nn.Linear(64,self.output_dim)

        self.out_act = nn.Sigmoid()

        self.act = nn.ReLU()

    def forward(self, x):

        # x.shape = (seq_len, batch_size)

        embedded_sent = self.embeddings(x)

#         print(embedded_sent.shape)

        # embedded_sent.shape = (seq_len, batch_size, embed_size)

        lstm_out, (h_n,c_n) = self.lstm(embedded_sent)

#         print(lstm_out.shape)

        # lstm_out.shape = (seq_len, batch_size, 2 * hidden_size)

        linear_output = self.act(self.dropout(self.fc(lstm_out.view(batch_size,-1))))

#         print(linear_output.shape)

        linear_output = self.act(self.dropout(self.fc2(linear_output)))

        linear_output = self.act(self.dropout(self.fc3(linear_output)))

        linear_output = self.out_act(self.out(linear_output))

        # print(linear_output.shape)

        return linear_output
model = memodel(len(itos),max_length).to(device)

error = nn.BCELoss()

opt = torch.optim.Adam(model.parameters(), lr=0.001)
# x,y = next(iter(train_loader))

# txt = Variable(x.reshape(-1,batch_size))

# lbl = Variable(y)



# print(txt.shape, lbl.shape)



# outputs = model(txt)
loss_list = []

iteration_list = []

accuracy_list = []

train_loss_list = []

train_acc_list = []

count=0
def train(ep):

  global loss_list

  global iteration_list

  global accuracy_list

  global train_loss_list

  global train_acc_list

  global count



  for epoch in range(ep): ##initially 2 epochs done

      train_loss=0

      train_total=0

      for txt,lbl in train_loader:

          model.train()

          txt = Variable(txt.reshape(-1,batch_size))

          lbl = Variable(lbl)

          opt.zero_grad()

          outputs = model(txt)

          loss = error(outputs, lbl)

          loss.backward()

          opt.step()

          count+=1

          train_loss+=loss.data

          train_total+=lbl.shape[0]

          if count%500==0: #500

              with torch.no_grad():

                  model.eval()

                  validation_loss=0

                  total=0

                  correct=0

                  for txt,lbl in test_loader:

                      txt = Variable(txt.reshape(-1,batch_size))

                      lbl = Variable(lbl)

                      outputs = model(txt)

                      valid_loss = error(outputs+1e-10, lbl)

                      ##loss

                      validation_loss+=valid_loss.data

                      ##accuracy:

                      prediction=outputs.data>=0.5

                      total+=lbl.shape[0] #not necessary it will be batch_size

                      correct+=(prediction==lbl.type(torch.uint8)).sum()

                  loss_list.append(validation_loss/total)

                  accuracy_list.append(100*correct/float(total))

                  iteration_list.append(count)

                  train_loss_list.append(train_loss/train_total)

                  train_loss=0

                  train_total=0



                  if count%2000==0: #2000

                      print('Iteration/Epoch: {}/{}  trainloss: {} Loss: {}  Accuracy: {} %'.format(count,epoch+1,train_loss_list[-1],loss_list[-1], accuracy_list[-1]))
train(2)
train(3)
len(iteration_list), len(train_loss_list), len(accuracy_list), len(loss_list)
# visualization loss 

plt.plot(iteration_list,loss_list,label='validation')

plt.plot(iteration_list,train_loss_list,label='train')

plt.xlabel("Number of iteration")

plt.ylabel("Loss")

plt.legend(loc='best')

plt.title("RNN: Loss vs Number of iteration")

plt.show()
# visualization accuracy 

plt.plot(iteration_list,accuracy_list,color = "red")

plt.xlabel("Number of iteration")

plt.ylabel("Accuracy")

plt.title("RNN: Accuracy vs Number of iteration")

plt.savefig('graph.png')

plt.show()
# torch.cuda.is_available()
# import pycuda.driver as cuda

# cuda.init()

# ## Get Id of default device

# torch.cuda.current_device()

# # 0

# cuda.Device(0).name() # '0' is the id of your GPU

# # Tesla K80
state = {

            'epoch': 5,

            'state_dict': model.state_dict(),

            'optimizer' : opt.state_dict()

        }
torch.save(state, 'model.pth.tar' )
# checkpoint = torch.load('model.pth.tar')

# start_epoch = checkpoint['epoch']+1

# model.load_state_dict(checkpoint['state_dict'])

# opt.load_state_dict(checkpoint['optimizer'])
test = pd.read_csv(files[1],index_col=0)

test.head()
# textlist = test['text'].apply(simple_toks).tolist()
# nums = [[stoi[w] if w in itos else 0 for w in sent] for sent in textlist]
with open(files[4], "rb") as fp:   # Unpickling

    nums = pickle.load(fp)
padded = pad_sequences(nums, maxlen=max_length, padding='post', truncating='post', value=0)
x = torch.from_numpy(padded).type(torch.LongTensor).to(device)

x = torch.utils.data.TensorDataset(x)

x_loader = torch.utils.data.DataLoader(x, batch_size = batch_size, shuffle = True)
preds = torch.zeros([len(test),1])

preds.shape
model.eval()

cnt=0

for txt in x_loader:

    bs=txt[0].shape[0]

    txt = Variable(txt[0].reshape(-1,bs))

    outputs = model(txt)

    ncnt=cnt+bs

    preds[cnt:ncnt]=(outputs.data>=0.5)

    cnt=ncnt
out = pd.Series(preds.numpy().reshape(-1))
out.index=test.index
out = out.astype(int)
outdf = pd.DataFrame(out*4)

outdf.head(10)
outdf.columns=['target']
outdf.head(10)
outdf.to_csv('test_output.csv')
ts = pd.read_csv('test_output.csv')
ts.head()
outdf
print('training loss {}, validation loss {}, validation accuracy {}'.format(train_loss_list[-1],loss_list[-1],accuracy_list[-1]))