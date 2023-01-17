!pip install -q bpe
import numpy as np

import pandas as pd

import os,sys

import glob

import re

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from tqdm import tqdm





import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader



import matplotlib

%matplotlib inline

from matplotlib import pyplot as plt



import torchtext

import nltk

from torch.nn.utils.rnn import pack_sequence,pad_sequence,pack_padded_sequence,pad_packed_sequence

import random

from sklearn.metrics import mean_absolute_error,log_loss



import multiprocessing





import time

from bpe import Encoder



data_path = "../input/sentencecorpus/SentenceCorpus/labeled_articles"

article_files = glob.glob(data_path+'/*')

print(len(article_files),' article files found')



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def clean_line(line):

    line = re.sub(r'\n','',line)

    return line



def remove_lines(article): #remove lines that start with ###

    ar = [ None if re.match('^\s*###.*',line) else line    for line in article ]

    ar = list(filter(None,ar))

    return ar



def read_file(filename):

    try:

        with open(filename) as f:

            return list(map(clean_line,f.readlines()))

    except Exception as e:

        print(e)



def article_to_pd(article,columns=['Category','Sentence']):

    try:

        article = [ re.match('^([A-Z]+)\s*(.*)',line).groups() for line in article ]

        return pd.DataFrame(article,columns=columns)

    except Exception as e:

        print(e,article)



def load_article_sentences(article_files):

    articles = list(map(read_file,article_files))

    articles = list(map(remove_lines,articles))

    pd_articles = pd.concat(list(map(article_to_pd,articles)),ignore_index=True)    

    return pd_articles



class PadSequence:

    def __call__(self, batch):

        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)

        sequences = [x[0] for x in sorted_batch]

        sequences_padded = pad_sequence(sequences, batch_first=True)

        lengths = torch.LongTensor([len(x) for x in sequences])

        labels = torch.LongTensor(map(lambda x: x[1], sorted_batch))

        return sequences_padded, lengths, labels



class bert_enc:

  def __init__(self):

    self.load()

  

  def fit(self,sentences):

        pass #for uniformity purposes

    

  def load(self):

    self.tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertTokenizer', 'bert-base-cased', do_basic_tokenize=False)

    self.model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertModel', 'bert-base-cased')

    self.model.eval()

  

  def enc_sent(self,sentence,index=-1):

    sep_token = ' [SEP] '

    sentence = sep_token+sentence+sep_token

    

    indexed_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sentence))

    tokens_tensor = torch.tensor([indexed_tokens])

    

    with torch.no_grad():

      encoded_layers, _ = self.model(tokens_tensor)

    

    return encoded_layers[index]

  

  def transform(self,doc,index=-1):

    sentences = nltk.sent_tokenize(doc)

    encs = []

    for sentence in sentences:

      encs.append(self.enc_sent(sentence,index=-1))

    if len(encs)==1:

        encs = encs[0]

    return encs# torch.cat(encs,dim=0)





def get_imp_wc(senteces_lst,important_words):

    def get_wc(sent):

        fd = dict(nltk.FreqDist(nltk.word_tokenize(sent)))

        fd_pd = pd.DataFrame(fd,index=[0])

        common = fd_pd[list(set(fd_pd.columns)&set(important_words))]

        return common



    pool = multiprocessing.Pool()

    map_func = [pool.map,map][1]

    print('Getting word frequencies')

    wcs = list(map_func(get_wc,senteces_lst))

    pool.close()

    print('Merging dataframes')

    return pd.concat(wcs,axis=0,ignore_index=True,sort=False)



def wc_name(filename):

    return '.'.join(os.path.basename(filename).split('.')[:-1])





def get_imp_word_count_data(sentences,word_lists_file="../input/sentencecorpus/SentenceCorpus/word_lists"):

    files = glob.glob(word_lists_file+'/*')

    word_types = dict(zip(list(map(wc_name,files)),list(map(read_file,files))))

    data_df = {}

    for d in word_types:

        data_df[d] = [word_types[d]]

    data_df = pd.DataFrame(data_df).drop(['stopwords'],axis=1)

    important_words = []

    for c in data_df.columns:

        important_words = important_words + data_df.loc[0,c]



    important_words = list(set(important_words))

    print('Found ',len(important_words),'important words')

    

    sentences_wc = list(sentences.Sentence)

    t1 = time.time()

    imp_word_count_data = get_imp_wc(sentences_wc,important_words).fillna(0)

    imp_word_count_data['Category'] = sentences.Category

    print(time.time()-t1)

    return imp_word_count_data



def split_labelled_sentences(sentences_data,device='cpu',encoder_to_use='bpe',pack=True):

    def enc_sentence(sentence):

        if encoder_to_use=='bpe':

            return torch.unsqueeze(torch.tensor(next(encoder.transform([sentence])),dtype=torch.float),dim=1)

        return encoder.transform(sentence)

    

    encoders = {'bpe':Encoder,'BERT':bert_enc}

    encoder = encoders[encoder_to_use]()

    encoder.fit(sentences.Sentence)

    t1 = time.time()

    X = list(map(enc_sentence,list(sentences_data.Sentence)))

    y = sentences.Category

    print(time.time()-t1,'Seconds for encoding sentences using ',encoder_to_use)

    

    if encoder_to_use=='BERT':

        X = list(map(lambda t:torch.squeeze(torch.tensor(t)),X))

    

    train_X,test_X,train_y,test_y = train_test_split(X,y)

    print('Lengths: train_X,train_y,test_X,test_y:',list(map(len,[train_X,train_y,test_X,test_y])))

    

    maxlen_found = torch.tensor(max(list(map(len,X)))).to(device)

    

    if pack:

        train_X = pack_sequence(sorted(train_X,key=len,reverse=True)).to(device)

        train_y = torch.tensor(list(train_y)).to(device)



        test_X = pack_sequence(sorted(test_X,key=len,reverse=True)).to(device)

        test_y = torch.tensor(list(test_y)).to(device)

    

    return train_X,train_y,test_X,test_y,maxlen_found
sentences = load_article_sentences(article_files)

print('Number of sentences ',len(sentences))



le = LabelEncoder()

le.fit(list(set(sentences.Category)))

sentences.Category = le.transform(list(sentences.Category))

output_size = len(le.classes_)

print('classes',le.classes_)



sentences.head()
imp_word_count_data = get_imp_word_count_data(sentences)

imp_word_count_data.head()
len(imp_word_count_data.columns)
from xgboost import XGBRegressor

wc_X = imp_word_count_data.drop(['Category'],axis=1).values

wc_y = imp_word_count_data[['Category']].values

wc_train_X,wc_test_X,wc_train_y,wc_test_y = train_test_split(wc_X,wc_y)

xgb_model = XGBRegressor(n_estimators=10000,learning_reate=0.01,n_jobs=10,max_depth=6)

xgb_model.fit(wc_train_X,wc_train_y,verbose=False,early_stopping_rounds=100, eval_set=[(wc_test_X, wc_test_y)])
predictions = xgb_model.predict(wc_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, wc_y)))

ns = sentences.copy()

ns.Category = le.inverse_transform(list(ns.Category))

ns['Category_pred'] = le.inverse_transform(list(map(lambda v:max(0,int(np.round(v).tolist())),predictions))).tolist()

ns.head()
list(range(1,2))
class SCFC(nn.Module):

    def __init__(self,input_size,hidden_size,output_size,num_layers=2):

        super(SCFC,self).__init__()

        self.input_size = input_size

        self.output_size = output_size

        self.num_layers = num_layers

        self.hidden_size = hidden_size

        self.fcs = []

        if num_layers<2:

            hidden_size = output_size

        self.fcs.append(nn.Linear(input_size,hidden_size))

        

        for i in range(1,num_layers):

            self.fcs.append(nn.Linear(hidden_size,hidden_size))

            

        if num_layers>1:

            self.fcs.append(nn.Linear(hidden_size,output_size))

        self.lsfmax = nn.LogSoftmax(dim=1)

    

    def forward(input):

        output=input

        for i in range(0,self.num_layers):

            output = F.relu(self.fcs[i](output))

        output = self.lsfmax(output)

        return output
device = torch.device('cuda')

scnn = SCFC(len(imp_word_count_data.columns),100,len(le.classes_),2).to(device)
def train(model,train_X,train_y,iters=100,lr=0.1):

    opz = optim.Adam(model.parameters(),lr=lr)

    loss_fn = nn.NLLLoss

    for i in iters:

        loss+=loss_fn(model(train_X),train_y)

        loss.backward()

        opz.step()

        print(loss.item())
train(scnn,wc_train_X,wc_train_y)
class RecurrentCell(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):

        super(RecurrentCell,self).__init__()

        self.input_size=input_size

        self.hidden_size=hidden_size

        self.rnn = nn.GRU(input_size,hidden_size, batch_first=True)

        self.LogSoftmax = nn.LogSoftmax(dim=1)

        self.output_fc = nn.Linear(hidden_size,output_size)

        

    def forward(self,batch):

#         x, x_lengths, _ = batch

#         x_pack = pack_padded_sequence(x, x_lengths, batch_first=True)

        output,hidden = self.rnn(batch)

        hidden = hidden.squeeze()

        result = self.LogSoftmax(self.output_fc(hidden))

        return result

    

    def init_hidden(self,shape):

        return torch.rand(shape)



def train(device, model,train_X,train_y,test_X,test_y, iters=100,lr=0.01,print_every=None):

    print_every = int(iters/10) if print_every is None else print_every

    opz = optim.Adam(model.parameters(),lr=lr)

    loss_func = nn.NLLLoss().to(device)

    train_losses = []

    test_losses = []

    model.zero_grad()

    opz.zero_grad()

    

    print('Iters  Train Loss       Test Loss')

    for i in range(1,iters+1):

        yhat = model(train_X)

        loss = loss_func(yhat,train_y)/maxlen_found

        loss.backward()

        opz.step()

        

        trnl = round(loss.cpu().item(),4)

        tstl = round(test(device,model,test_X,test_y),4)

        if i%print_every==0 or i==1:

            print(str(i).rjust(3,'0'),'\t',trnl,'\t',tstl)

        train_losses.append(trnl)

        test_losses.append(tstl)





    plt.plot(train_losses)

    plt.plot(test_losses)

    plt.legend(['train loss','test loss'])



def test(device,model,test_X,test_y):

    loss_func = nn.NLLLoss().to(device)

    with torch.no_grad():

        yhat = model(test_X)

        loss = loss_func(yhat,test_y)/maxlen_found

        return loss.item()

    

def num_words(sentence):

    return len(nltk.word_tokenize(sentence))





def evaluate(model,data):#,device=None):

    #device = torch.device('cpu') if device is None else device

    #model = model.to(device)

    tx = pack_sequence(sorted(data,key=len,reverse=True)).to(device)

    with torch.no_grad():

        yhat = model(tx)

        yh = []

        for t in yhat:

            val, index = t.max(0)

            yh.append(index.item())

        return yh
input_size = 1 #if encoder_to_use=='bpe' else 768

hidden_size = 5



train_X,train_y,test_X,test_y,maxlen_found = split_labelled_sentences(sentences,device,encoder_to_use='bpe')

maxlen_found/=10
model1 = RecurrentCell(input_size,hidden_size,output_size).to(device)

train(device,model1,train_X,train_y,test_X,test_y, iters=10000,lr=0.0001)
!pip install -q syft
import syft as sy

from syft.frameworks.torch.pointers import PointerTensor

from syft.frameworks.torch.tensors.decorators import LoggingTensor
def create_worker_objs(num_workers,hooks=None):

    workers = []

    if hooks is None:

        hooks = [ sy.TorchHook(torch) for _ in range(num_workers) ]

    workers = [sy.VirtualWorker(hooks[i],id="worker"+str(i+1)) for i in range(num_workers)]

    return workers



def train_workers(model,workers,iters,optimizer,lr):

    print_every = int(iters/10)

#     with tqdm(total=iters*len(workers)) as pbar:

    with tqdm(total=iters) as pbar:

        for j in range(iters):

            send_model(model,workers)

            for i in workers:

                workers[i]['opz'] = optimizer(workers[i]['model'].parameters(),lr)

                workers[i]['opz'].zero_grad()

                pred = workers[i]['model'](workers[i]['data']['train_X'])

                workers[i]['loss'] = F.nll_loss(pred,workers[i]['data']['train_y'])

                workers[i]['loss'].backward()

                workers[i]['opz'].step()

            pbar.update(1)

            pbar.set_description(str(round(workers[i]['loss'].cpu().item(),4)).rjust(6))

            #if j%print_every==0:

                #print(i, workers[i]['loss'].cpu().item())

            weights = aggregate_weights(model,workers)

            model.load_state_dict(weights) #update the model at end of each iteration

    return model



def get_params(worker_item):

    return list(worker_item[1]['model'].named_parameters())



def aggregate_weights(model,workers,eta=1):

    model_parameters_lst = list(map(get_params,workers.items()))

    num_params = len(model_parameters_lst[0])

    num_workers = len(workers)

    reference_params = list(model.named_parameters())

    

    avg_params = [] 

    for i in range(num_params):

        avg_params.append([model_parameters_lst[0][i][0],torch.zeros_like(model_parameters_lst[0][i][1])])



    for i in range(num_params):

        for j in range(num_workers):

            avg_params[i][1].add_(model_parameters_lst[j][i][1]-reference_params[i][1])

    

    for i in range(num_params):

        avg_params[i][1] = torch.div(avg_params[i][1],num_workers*eta)

        

    return dict(avg_params)

#     model.load_state_dict(dict(avg_params))

#     return model



def send_model(model,workers):

    for w in workers:

        workers[w]['model'] = model.copy()#.send(workers[w]['obj'])



def model_params_diff(params1,params2,num_params):

    diffs = []

    for i in range(num_params):

        diffs.append(params2[i]-params1[i])

    return diffs



def test_model(model,test_X,test_y,device=None):

    device = torch.device('cpu') if device is None else device

    loss_func = nn.NLLLoss().to(device)

    tx = pack_sequence(sorted(test_X,key=len,reverse=True)).to(device)#.send(worker_objs[i])

    ty = torch.tensor(list(test_y)).to(device)#.send(worker_objs[i])

    with torch.no_grad():

        yhat = model(tx)

        yh = []

        for t in yhat:

            val, index = t.max(0)

            yh.append(index.item())

        loss = loss_func(yhat,ty)

        return list(map(lambda v:round(v,4),[loss.item(),100*accuracy_score(test_y,yh)]))



        



# @sy.func2plan()

# def train_worker(woker,iters):

#     model,data,opz = workers['worker1']['model'],workers['worker1']['data'],workers['worker1']['model']

#     opz.zero_grad()

#     train_X = pack_sequence(sorted(data['train_X'],key=len,reverse=True))

#     train_y = torch.tensor(list(data['train_y']))

#     for i in range(iters):

#         pred = model(train_X)

#         loss = F.nll_loss(pred,train_y)

#         loss.backward()

#         opz.step()

#         if i%print_every==0:

#             print(loss.item())

device = torch.device('cpu')

train_X,train_y,test_X,test_y,maxlen_found = split_labelled_sentences(sentences,device,encoder_to_use='bpe',pack=False)
train_y,test_y = list(train_y),list(test_y)
train_y[0:116]
num_workers = 20

lr = 0.0001

data_share = int(len(train_X)/num_workers)

worker_objs = create_worker_objs(num_workers)

optimizer = optim.Adam

model2 = RecurrentCell(input_size,hidden_size,output_size)#.to(device)



workers = {}

for i in range(num_workers):

    tx = pack_sequence(sorted(train_X[i*data_share:(i+1)*data_share],key=len,reverse=True))#.send(worker_objs[i])

    ty = torch.tensor(list(train_y[(i*data_share):((i+1)*data_share)]))#.send(worker_objs[i])

    worker_data = {'train_X':tx,'train_y':ty}

    workers["worker"+str(i+1)] = {'obj':worker_objs[i],'data':worker_data}
workers[random.choice(list(workers.keys()))]
model2 = train_workers(model2,workers,10,optimizer,lr=0.0001)
test_model(model1,test_X,test_y,device),test_model(model2,test_X,test_y)