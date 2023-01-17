!pip install ../input/sacremoses/sacremoses-master/ > /dev/null



import os

import sys

import glob

import torch



sys.path.insert(0, "../input/transformers/transformers-master/")

import transformers



import torch

from transformers import *
import sys

package_dir_a = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"

sys.path.insert(0, package_dir_a)



from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
!ls ../input/emotion-baseline1/pytorch_Emotion_model.pkl
from transformers import *

import shutil

# Translate model from tensorflow to pytorch

WORK_DIR = "../working/"

BERT_MODEL_PATH = '../input/bert-pretrained-models/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12/'

convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(

    BERT_MODEL_PATH + 'bert_model.ckpt',

BERT_MODEL_PATH + 'bert_config.json',

WORK_DIR + 'pytorch_model.bin')



shutil.copyfile(BERT_MODEL_PATH + 'bert_config.json', WORK_DIR + 'config.json')
import pandas as pd

import numpy as np

from tqdm import tqdm

tqdm.pandas()

test_10k = pd.read_csv("../input/emotion-data-true/test_10k.csv").fillna('内容为空')

train_100k = pd.read_csv("../input/emotion-data-true/train_100k.csv")

train_labels=pd.get_dummies(train_100k['label'],prefix='label')

train_100k[list(train_labels.columns)]=train_labels

train_100k['label']=train_100k['label']+1
train_100k=train_100k[~train_100k['content'].isnull()]
words=['发烧','发热','感冒','肺炎']

for word in words:

    train_100k['content']=train_100k['content'].progress_apply(lambda x: ('***'+word+'***').join(x.split(word)) if word in x else x)

    test_10k['content']=test_10k['content'].progress_apply(lambda x: ('***'+word+'***').join(x.split(word)) if word in x else x)

train_100k.head(5)
import torch

#import torch.utils.data as data

from torchvision import datasets, models, transforms

from transformers import *

from sklearn.utils import shuffle

from sklearn.metrics import f1_score



import random



Target_nums=['label']

MAX_LEN = 150

SEP_TOKEN_ID = 102

DEVICE = 'cuda'





SEED=2019

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)

torch.backends.cudnn.deterministic = True

#create DataSet

class EmotionDataSet(torch.utils.data.Dataset):

    def __init__(self, df,train_mode=True, labeled=True):

        self.df = df

        self.train_mode = train_mode

        self.labeled = labeled

        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None)

        

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        token_ids, seg_ids = self.get_token_ids(row)

        

        if self.labeled:

            labels = self.get_label(row)

            return token_ids,seg_ids,labels

        return token_ids,seg_ids

        

    def select_tokens(self, tokens, max_num):

        if len(tokens) <= max_num:

            return tokens

        else:

            return tokens[:max_num]

        

    def get_seg_ids(self, ids):

        seg_ids = torch.zeros_like(ids)

        seg_idx = 0

        for i, e in enumerate(ids):

            seg_ids[i] = seg_idx

            if e == SEP_TOKEN_ID:

                seg_idx += 1

        max_idx = torch.nonzero(seg_ids == seg_idx)

        seg_ids[max_idx] = 0

        return seg_ids



    def get_label(self, row):

        return torch.tensor(row[Target_nums].values.astype(np.float32)).long()

    

    def get_token_ids(self,row):

        

        tokens= ['[CLS]'] + self.select_tokens(self.tokenizer.tokenize(row.content),MAX_LEN-2)+['[SEP]']

        

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        

        #padding

        if len(token_ids) < MAX_LEN:

            token_ids += [0] * (MAX_LEN- len(token_ids))

        #totensor

        ids = torch.tensor(token_ids)[:MAX_LEN]

        #segid

        seg_ids = self.get_seg_ids(ids)

        

        return ids,seg_ids



    def collate_fn(self, batch):

        token_ids = torch.stack([x[0] for x in batch])

        seg_ids = torch.stack([x[1] for x in batch])

        

    

        if self.labeled:

            labels = torch.stack([x[2] for x in batch])

            return token_ids, seg_ids, labels.squeeze()

        else:

            return token_ids, seg_ids

    





def get_train_val_loaders(df,batch_size=4, val_batch_size=4, val_percent=0.2):

    df = shuffle(df, random_state=1234)

    split_index = int(len(df) * (1-val_percent))

    

    df_train = df[:split_index]

    df_val = df[split_index:]



    print(df_train.shape)

    print(df_val.shape)



    ds_train = EmotionDataSet(df_train)

    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=ds_train.collate_fn, drop_last=True)

    train_loader.num = len(df_train)



    ds_val = EmotionDataSet(df_val, train_mode=False)

    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=val_batch_size, shuffle=False, num_workers=0, collate_fn=ds_val.collate_fn, drop_last=False)

    val_loader.num = len(df_val)

    val_loader.df = df_val



    return train_loader, val_loader





def get_test_loader(df,batch_size=4):

    

    ds_test = EmotionDataSet(df, train_mode=False, labeled=False)

    loader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=ds_test.collate_fn, drop_last=False)

    loader.num = len(df)

    

    return loader





def test_train_loader(train):

    loader, _ = get_train_val_loaders(train,2)

    for token_ids, seg_ids,labels in loader:

        print(token_ids)

        print(seg_ids)

        print(labels)

        break

def test_test_loader(test):

    loader = get_test_loader(test,2)

    for token_ids, seg_ids in loader:

        print(token_ids)

        print(seg_ids)

        break
# test_train_loader(train_100k)

# test_test_loader(test_10k)
# from transformers import *

# import torch

# import torch.nn as nn

# import torch.nn.functional as F

# import time

# from tqdm import tqdm_notebook



# from transformers import *

# import torch

# import torch.nn as nn

# import torch.nn.functional as F

# import time

# from tqdm import tqdm_notebook

# from scipy.stats import spearmanr





        

# class EmotionModel(nn.Module):

#     def __init__(self, n_classes=3):

#         super(EmotionModel, self).__init__()

#         self.model_name = 'EmotionModel'

#         self.bert_model = BertModel.from_pretrained("../working",cache_dir=None)

        

#         self.fcc= nn.Sequential(nn.Linear(768, n_classes),nn.LogSoftmax())

        

    

#     def forward(self,ids,seg_ids):

#         attention_mask = (ids > 0)

#         last_seq=self.bert_model(input_ids=ids, attention_mask=attention_mask)[0][:,0,:]

        

#         return self.fcc(last_seq)

    

    

    

# def test_model():

#     x = torch.tensor([[1,2,3,4,5, 0, 0], [1,2,3,4,5, 0, 0]])

        

#     seg_ids = torch.tensor([[1,0,0,0,0, 0, 0], [1,0,0,0,0, 0, 0]])

#     model = EmotionModel()

    

#     y = model(x, seg_ids)

#     print(y)

# netG = EmotionModel()

# print('# generator parameters:', sum(param.numel() for param in netG.parameters()))

from transformers import *

import torch

import torch.nn as nn

import torch.nn.functional as F

import time

import math

from tqdm import tqdm_notebook



from transformers import *

import torch

import torch.nn as nn

import torch.nn.functional as F

import time

from tqdm import tqdm_notebook

from scipy.stats import spearmanr





import torch

import torch.nn as nn

import numpy as np

class ScaledDotProductAttention(nn.Module):

    def __init__(self,attention_dropout=0.0):

        super(ScaledDotProductAttention, self).__init__()

        self.dropout = nn.Dropout(attention_dropout)

        self.softmax = nn.Softmax(dim=2)



    def forward(self,q,k,v,scale=None,attn_mask=None):

        """



        :param q: [B,Lq,D_q]

        :param k:[B,Lk,D_k]

        :param v:[B,Lv,D_v]

        :param scale: 缩放因子

        :param attn_mask:[B,Lq,Lk]

        :return:上下文张量，和attention张量

        """

        q=q.expand(k.size()[0],q.size()[0],q.size()[1])

#         print(q.size())



        #[B,s,s]

        attention=torch.bmm(q,k.transpose(1,2))



        if scale:

            attention=attention*scale

        #attn_mask:[B,sq,sk]

        if attn_mask!=None:

            attention=attention.masked_fill_(attn_mask,-np.inf)



        attention=self.softmax(attention)



        attention=self.dropout(attention)



        #[b,s,d]

        context=torch.bmm(attention,v)



        return context,attention

        

class EmotionModel(nn.Module):

    def __init__(self, n_classes=3):

        super(EmotionModel, self).__init__()

        self.model_name = 'EmotionModel'

        self.bert_model = BertModel.from_pretrained("../working",cache_dir=None,output_hidden_states=True)

        

        self.fcc= nn.Sequential(nn.Linear(768, n_classes),nn.LogSoftmax())

        

        self.Emotion_attention=ScaledDotProductAttention()

        self.query = torch.nn.Parameter(torch.empty(1,768))

        torch.nn.init.uniform_(self.query, a=0, b=1)



        

#     def get_seq_comapre(self,ids):

#         attention_mask = (ids > 0)

#         last_seq=self.bert_model(input_ids=ids, attention_mask=attention_mask)[0][:,0,:]

    

    def forward(self,ids,seg_ids):

        attention_mask = (ids > 0)

        last_seq,pooled_output,hidden_states=self.bert_model(input_ids=ids, attention_mask=attention_mask)

#         

#         print(len(hidden_states))

        

#         cls_seq=torch.cat([ h[:,0,:].unsqueeze(1) for h in hidden_states[1:]],1)

#         print(hidden_states[0].size())

        

#         pad_mask = ids.eq(0)

#         pad_mask = pad_mask.unsqueeze(1)

#         attout,_=self.Emotion_attention(self.query,cls_seq,cls_seq,scale=1/math.sqrt(768))

        

#         print(attout.squeeze().size())

        

        return self.fcc(pooled_output)

    

    

    

def test_model():

    x = torch.tensor([[1,2,3,4,5, 0, 0], [3,2,3,4,5, 0, 0]])

        

    seg_ids = torch.tensor([[1,0,0,0,0, 0, 0], [1,0,0,0,0,0,0]])

    model = EmotionModel()

    

    y = model(x, seg_ids)

    print(y)

netG = EmotionModel()

print('# generator parameters:', sum(param.numel() for param in netG.parameters()))

test_model()
logs='- epoch - {0:2d} - train_loss - {1:.4f} train_score - {2:.3f} - val_loss - {3:.4f} - val_score - {4:.3f} - best_loss - {5:.3f}'



def metric_fn(p, t):

    p=np.argmax(p,axis=1)

    return f1_score(t,p,average='macro')



@torch.no_grad()

def validation_fn(model, loader, loss_fn):

    y_pred, y_true, tloss = [], [], []

    for ids,seg_ids,target in loader:

        outputs = model(ids.cuda(DEVICE),seg_ids.cuda(DEVICE))

        loss = loss_fn(outputs, target.cuda(DEVICE))

        tloss.append(loss.item())

        y_true.append(target.detach().cpu().numpy())

        y_pred.append(outputs.detach().cpu().numpy())

        

    tloss = np.array(tloss).mean()

    y_pred = np.concatenate(y_pred)

    y_true = np.concatenate(y_true)

    metric = metric_fn(y_pred, y_true)

    return tloss, metric







def predict_model_test(model,loader):

    model.eval()

    y_pred=[]

    bar = tqdm_notebook(loader)

    for ids,seg_ids in bar:

        outputs = model(ids.cuda(DEVICE),seg_ids.cuda(DEVICE))

        y_pred.append(outputs.detach().cpu().numpy())

    y_pred = np.concatenate(y_pred)

    return y_pred







def train_model(model,train_loader,val_loader,epochs=4,model_save_path='pytorch_Emotion_model_true.pkl'):

    

    

    ########梯度累计

    accumulation_steps=2

    batch_size = accumulation_steps*16

    

    ########早停

    early_stop_epochs=2

    no_improve_epochs=0

    

    ########优化器 学习率

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [

            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.8},

            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},

#             {'params':model.query,'lr':1e-3}

            ]   

    

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)

    

    

    train_len=len(train_loader)

    

    loss_fn = nn.NLLLoss().cuda(DEVICE)

    best_vmetric=1.

    

    

    logss=[]

    

    

    

    for epoch in range(1,epochs+1):

        

        y_pred, y_true = [], []

        start_time = time.time

        tloss = []

        model.train()

        bar = tqdm_notebook(train_loader)

        for i,(ids, seg_ids,labels) in enumerate(bar):

#             print(i)

            outputs = model(ids.cuda(DEVICE),seg_ids.cuda(DEVICE))

            loss = loss_fn(outputs, labels.cuda(DEVICE))

#             print(labels)

            tloss.append(loss.item())

            loss.backward()

            #梯度累计

            if (i+1) % accumulation_steps == 0 or (i+1)==train_len:

                optimizer.step()

                optimizer.zero_grad()

            y_true.append(labels.detach().cpu().numpy())

            y_pred.append(outputs.detach().cpu().numpy())

            y_t = np.concatenate(y_pred)

            y_p = np.concatenate(y_true)

            metric = metric_fn(y_t,y_p)

            bar.set_postfix(loss=np.array(tloss).mean(),score=metric)

        

        tloss = np.array(tloss).mean()

        y_pred = np.concatenate(y_pred)

        y_true = np.concatenate(y_true)

        tmetric = metric_fn(y_pred, y_true)

        vloss, vmetric = validation_fn(model, val_loader, loss_fn)

        print(logs.format(epoch,tloss,tmetric,vloss,vmetric,best_vmetric))  

        logss.append(logs.format(epoch,tloss,tmetric,vloss,vmetric,best_vmetric))

        #save best model

        if vloss<=best_vmetric:

            torch.save(model.state_dict(),model_save_path)

            best_vmetric=vloss

            no_improve_epochs=0

            print('improve save model!!!')

        else:

            no_improve_epochs+=1

        ###for eary stop

        if no_improve_epochs==early_stop_epochs:

            print('no improve score !!! stop train !!!')

            break

        

    return logss
netG = EmotionModel().cuda(DEVICE)

train_loader, val_loader = get_train_val_loaders(train_100k,batch_size=16, val_batch_size=16, val_percent=0.1)

logss=train_model(netG,train_loader,val_loader,epochs=3)

# print(logss)

for loss in logss:

    print(loss)
test_loader=get_test_loader(test_10k,16)

Model_path='pytorch_Emotion_model_true.pkl'

model= EmotionModel().cuda(DEVICE)

model.load_state_dict(torch.load(Model_path))
test_preds= predict_model_test(model,test_loader)

test_label=np.argmax(test_preds,axis=1)-1
test_10k['y']=test_label
sub=test_10k[['user_id','y']]

sub.columns=['id','y']

sub['id']=sub['id'].apply(lambda  x: str(x)+' ')



sub.to_csv('sub_baseline_true.csv',index=False)