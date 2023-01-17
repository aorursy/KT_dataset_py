!pip install ../input/sacremoses/sacremoses-master/ > /dev/null



import os

import sys

import glob

import torch



sys.path.insert(0, "../input/transformers/transformers-master/")

import transformers

from transformers import *
import sys

package_dir_a = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"

sys.path.insert(0, package_dir_a)



from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from transformers import *

import shutil

# Translate model from tensorflow to pytorch

WORK_DIR = "../working/"

# BERT_MODEL_PATH = '../input/bert-pretrained-models/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12/'

BERT_MODEL_PATH ='../input/roberta-large/'

convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(

    BERT_MODEL_PATH + 'roberta_zh_large_model.ckpt',

BERT_MODEL_PATH + 'bert_config_large.json',

WORK_DIR + 'pytorch_model.bin')



shutil.copyfile(BERT_MODEL_PATH + 'bert_config_large.json', WORK_DIR + 'config.json')



print('yes')
import pandas as pd

import numpy as np

import json

import random

import copy

import re

from tqdm import tqdm

import collections

from random import shuffle

import warnings

warnings.filterwarnings("ignore")

tqdm.pandas()



#import torch.utils.data as data

from torchvision import datasets, models, transforms

from transformers import *

from sklearn.utils import shuffle

from sklearn.metrics import f1_score

import random



!pip install pytorch-crf
! ls ../input/baidu-competition/
def load_data(filename):

    D = []

    with open(filename) as f:

        for l in f:

            l = json.loads(l)

            #遍历每一个事件类型

            event_list=[]#一个文本有多个事件

            for event in l['event_list']:

                trigger=event['trigger'] #触发词（裁员）

                event_type=event['event_type'] #事件类型（组织关系-裁员）

                trigger_start_index=event['trigger_start_index'] #触发词的开始 59

                enent_key=(event_type,trigger,trigger_start_index)

                #每一个事件的argument

                arguments = collections.defaultdict(list)

                for argument in event['arguments']:

                    value = argument['argument'] #雀巢

                    role=argument['role'] ##裁员方

                    argument_start_index=argument['argument_start_index'] #在原文开始的位置

                    key = role

                    #有重复的论元

                    arguments[key].append((argument_start_index,value))

                

                event_list.append((enent_key,arguments))

            D.append([[l['text'],l['id']],event_list])

    return D



def load_test_data(filename):

    D = []

    with open(filename) as f:

        for l in f:

            l = json.loads(l)

            #遍历每一个事件类型

            event_list=[]#一个文本有多个事件

            l['event_list']=['null']

            for event in l['event_list']:

                trigger='null' #触发词（裁员）

                event_type='null' #事件类型（组织关系-裁员）

                trigger_start_index='null' #触发词的开始 59

                enent_key=(event_type,trigger,trigger_start_index)

                #每一个事件的argument

                arguments = collections.defaultdict(list)

                event_list.append((enent_key,arguments))

            D.append([[l['text'],l['id']],event_list])

    return D

data_path='../input/baidu-competition/'

test_data = load_test_data(data_path+'test1.json')

train_data = load_data(data_path+'train.json')

valid_data = load_data(data_path+'dev.json')

print(len(train_data))

print(len(valid_data))



with open(data_path+'event_schema.json') as f:

    id2label, label2id, n = {}, {}, 0

    for l in f:

        l = json.loads(l)

        for role in l['role_list']:

            key = (l['event_type'], role['role'])

            id2label[n] = key

            label2id[key] = n

            n += 1

    num_labels = len(id2label) * 2 + 1

print('nums labels:',num_labels)
for t in train_data:

    print(t)

    print('*'*50)

    break
event_type_roles=collections.defaultdict(list)

with open(data_path+'event_schema.json') as f:

    for l in f:

        l = json.loads(l)

        key=l['event_type']

        for role in l['role_list']:

            event_type_roles[key].append(role['role'])
class_type_map={'财经/交易':0,'产品行为':1,'交往':2,'竞赛行为':3,'人生':4,'司法行为':5,'灾害/意外':6,'组织关系':7,'组织行为':8}

class_type_map_65=dict()

index=0

for _,key in id2label.items():

    if key[0] not in class_type_map_65:

        class_type_map_65[key[0]]=index

    else:

        continue

    index+=1

    

class_type_map_reverse=dict()

for k,v in class_type_map_65.items():

    class_type_map_reverse[v]=k
m=0

for k,v in class_type_map_65.items():

    if len(k)>m:

        m=len(k)

m
# # !ls ../input/right-recall1-0862

# import pickle

# with open('../input/shijianchouquqa/ans_recall_large_val.pkl','rb') as f:

#     val_recall_data=pickle.load(f)

# with open('../input/shijianchouquqa/ans_recall_large_test.pkl','rb') as f:

#     test_recall_data=pickle.load(f)
val_recall_data=collections.defaultdict(list)

test_recall_data=collections.defaultdict(list)

import pickle

for i in range(1,6):

    with open('../input/cv-recall/val_recall_{}.pkl'.format(i),'rb') as f:

        temp=pickle.load(f)

        for k,v in temp.items():

           val_recall_data[k].extend(v)

           val_recall_data[k]=list(set(val_recall_data[k]))

    with open('../input/cv-recall/test_recall_{}.pkl'.format(i),'rb') as f:

        temp=pickle.load(f)

        for k,v in temp.items():

            test_recall_data[k].extend(v)

            test_recall_data[k]=list(set(test_recall_data[k]))
for k,v in val_recall_data.items():

    new_v=[]

    for vi in v:

        if len(vi[0])>=2:

            new_v.append(vi)

    val_recall_data[k]=new_v
val_recall_data


MAX_TEXT_LEN = 160

MAX_QUESTION_LEN=15

MAX_LEN=180

SEP_TOKEN_ID = 102

DEVICE = 'cuda'









def seed_everything(seed):

    """

    Seeds basic parameters for reproductibility of results

    

    

    Arguments:

        seed {int} -- Number of the seed

    """

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False





SEED=2020

seed_everything(SEED)

import unicodedata

def _is_control(ch):

    """控制类字符判断

    """

    return unicodedata.category(ch) in ('Cc', 'Cf')



def _is_special(ch):

    """判断是不是有特殊含义的符号

    """

    return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')



def stem(token):

    """获取token的“词干”（如果是##开头，则自动去掉##）

    """

    if token[:2] == '##':

        return token[2:]

    else:

        return token

def rematch(text,tokens,_do_lower_case=True):

    '''

    返回的是token后标签与原始的映射

    '''

    normalized_text, char_mapping = '', []

    #规范化样本

    for i, ch in enumerate(text):

        if _do_lower_case:

            ch = unicodedata.normalize('NFD', ch)

            ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])

            ch = ch.lower()

        ch = ''.join([

                c for c in ch

                if not (ord(c) == 0 or ord(c) == 0xfffd or _is_control(c))

            ])

        normalized_text += ch

        char_mapping.extend([i] * len(ch))



    text, token_mapping, offset = normalized_text, [], 0

    for i,token in enumerate(tokens):

        if _is_special(token):

            token_mapping.append([offset])

            offset+=1

        else:

            token = stem(token)

            start = text[offset:].index(token) + offset

            end = start + len(token)

            token_mapping.append(char_mapping[start:end])

            offset = end



    return token_mapping

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None)

text='雀巢裁员4000人：时代抛弃你时，连招呼都不会打！'

tokens=tokenizer.tokenize(text)

print(tokens)

mapping=rematch(text,tokens)
def search(pattern, sequence):

    """从sequence中寻找子串pattern

    如果找到，返回第一个下标；否则返回-1。

    """

    n = len(pattern)

    for i in range(len(sequence)):

        if sequence[i:i + n] == pattern:

            return i

    return -1



def search_list(pattern, sequence):

    """从sequence中寻找子串pattern

    如果找到，返回第一个下标；否则返回-1。

    """

    n = len(pattern)

    ans=[]

    for i in range(len(sequence)):

        if sequence[i:i + n] == pattern:

            ans.append(i)

    return ans



class Feature(object):

    def __init__(self,item_id,or_text,enent_text,trigger_and_start,mapping,mapping_off,token_ids,question,labels,data,split_off):

        self.id=item_id

        self.or_text=or_text

        self.enent_text=enent_text

        self.trigger_and_start=trigger_and_start

        self.mapping=mapping

        self.mapping_off=mapping_off

        self.token_ids=token_ids

        self.question=question

        self.labels=labels

        self.data=data

        self.split_off=split_off

        

        

        

        

    def __str__(self):

        return self.__repr__()



    def __repr__(self):

        s = ""

        s += "id: %s" % (str(self.id))

        s += "or_text: %s" % (str(self.or_text))

        s += "enent_text: %s" % (str(self.enent_text))

        s += "mapping: %s" % (str(self.mapping))

        s += "mapping_off: %s" % (str(self.mapping_off))

        s += "token_ids: %s" % (str(self.token_ids))

        s += "question: %s" % (str(self.question))

        s += "labels: %s" % (str(self.labels))

        s += "data: %s" % (str(self.data))

        s += "split_off: %s" % (str(self.split_off))

        return s



class zy_DataSet(torch.utils.data.Dataset):

    def __init__(self, data_list,train_mode=False, val_mode=False,test_mode=False,labeled=True,recall=False):

        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None)

        self.train_mode = train_mode

        self.val_mode=val_mode

        self.test_mode=test_mode

        self.labeled = labeled

        self.recall=recall

        if recall and self.val_mode:

            print(recall)

            self.features =self.get_recall_features(data_list,val_recall_data)

        elif recall and self.test_mode:

            self.features =self.get_recall_features(data_list,test_recall_data)

        elif self.train_mode:

            self.features =self.get_train_features(data_list)

        elif self.val_mode:

            self.features =self.get_train_features(data_list)

        elif self.test_mode:

            self.features =self.get_train_features(data_list)

        else:

            print('no features !!!')

            

    def get_feat_qa(self,q_text,text):

        pass

    

    def get_recall_features(self,data_list,recall):

        neg=0

        features=[]

        ###滑动窗口

        print(len(data_list))

        split_data_list=[]

        for index in range(len(data_list)):

            data=data_list[index]

            text=data[0][0]

            data.append(0)

            #长度小于最大长度

            if len(text)<=MAX_TEXT_LEN-2:

                split_data_list.append(data)

                continue

            #切分句子

            texts=re.split(r'。|；|！|，',text)

            if len(texts[-1])==0:

                texts=texts[:-1]

            split_off=0

            new_ts=[]

            i=0

            split_text=''

            windows=4 #转移为两句话

            begin=0

            current_len=0

            all_split_text=[]

#             print('*'*50)

#             print(data)

#             print(text)

#             print(texts)

            while i<len(texts):

                s_t=texts[i]

                tag=text[len(''.join(texts[:i+1]))+i] if len(''.join(texts[:i+1]))+i<len(text) else ''

                if len(split_text+s_t)+1>=MAX_TEXT_LEN-2:

                    if not any([split_text in i for i in all_split_text]):

                        data[2]=split_off

                        data[0][0]=split_text

                        split_data_list.append(copy.deepcopy(data))

                        all_split_text.append(split_text)

#                         print('-'*50)

#                         print(split_text)

                    split_text='' #重新清空

                    split_off+=len(''.join(texts[begin:begin+windows]))+windows

                    begin=begin+windows

                    i=begin

                else:

                    split_text=split_text+s_t+tag

                    i=i+1

            if not any([split_text in i for i in all_split_text]):

#                 print('-'*50)

#                 print(split_text)

                data[2]=split_off

                data[0][0]=split_text[:] #加上该标点符号

                split_data_list.append(copy.deepcopy(data))

                all_split_text.append(split_text)

        print(len(split_data_list))

        print('*'*50)

        for data in tqdm(split_data_list):

            text=data[0][0] #文本

            item_id=data[0][1] #id

            split_off=data[2] #切分起点

            enent_key_argument=data[1]

            trigger_start=recall[item_id]

            #遍历每一个事件

            for trigger,trigger_start_index in trigger_start:

#                 enent_type=event_key[0]

#                 trigger=event_key[1]

#                 trigger_start_index=event_key[2]

                enent_text=text

                if trigger_start_index>=split_off and trigger_start_index<split_off+len(text):

                    assert text[trigger_start_index-split_off:trigger_start_index-split_off+len(trigger)]==trigger

                    #对每一个事件建立一个特定的文本，这个文本去掉了与非触发词相同的词

                    text_before=text[:trigger_start_index-split_off]

                    text_end=text[trigger_start_index-split_off+len(trigger):]

                    enent_text=text_before+"$"+trigger+"$"+text_end

                    t_tokens=self.tokenizer.tokenize(enent_text)

                    assert len(t_tokens)<=MAX_TEXT_LEN+10

                    mapping=rematch(enent_text,t_tokens) #获得原始的map

                    question=trigger

                    q_tokens= self.tokenizer.convert_tokens_to_ids(['[CLS]']+self.tokenizer.tokenize(question)+['[SEP]']) #20

                    if len(q_tokens)>MAX_QUESTION_LEN:

                            print('big')

                            print(len(q_tokens))

                            print(question)

                    text_token_ids = self.tokenizer.convert_tokens_to_ids(t_tokens+['[SEP]'])

                    token_ids=q_tokens+text_token_ids

                    assert len(token_ids)<=MAX_LEN

                    if len(token_ids) < MAX_LEN:

                        token_ids += [0] * (MAX_LEN- len(token_ids))

                    mapping_off=len(q_tokens)

                    labels=0

                    

                    feature=Feature(item_id=item_id,

                            or_text=text,

                            enent_text=enent_text,

                            trigger_and_start=(trigger,trigger_start_index),

                            mapping=mapping,

                            mapping_off=mapping_off,

                            token_ids=token_ids,

                            question=question,

                            labels=labels,

                            data=data[1],

                            split_off=split_off,

                            )

                    features.append(feature)



        return features

    def get_train_features(self,data_list,recall=None):

        neg=0

        features=[]

        ###滑动窗口

        print(len(data_list))

        split_data_list=[]

        for index in range(len(data_list)):

            data=data_list[index]

            text=data[0][0]

            data.append(0)

            #长度小于最大长度

            if len(text)<=MAX_TEXT_LEN-2:

                split_data_list.append(data)

                continue

            #切分句子

            texts=re.split(r'。|；|！|，',text)

            if len(texts[-1])==0:

                texts=texts[:-1]

            split_off=0

            new_ts=[]

            i=0

            split_text=''

            windows=4 #转移为两句话

            begin=0

            current_len=0

            all_split_text=[]

#             print('*'*50)

#             print(data)

#             print(text)

#             print(texts)

            while i<len(texts):

                s_t=texts[i]

                tag=text[len(''.join(texts[:i+1]))+i] if len(''.join(texts[:i+1]))+i<len(text) else ''

                if len(split_text+s_t)+1>=MAX_TEXT_LEN-2:

                    if not any([split_text in i for i in all_split_text]):

                        data[2]=split_off

                        data[0][0]=split_text

                        split_data_list.append(copy.deepcopy(data))

                        all_split_text.append(split_text)

#                         print('-'*50)

#                         print(split_text)

                    split_text='' #重新清空

                    split_off+=len(''.join(texts[begin:begin+windows]))+windows

                    begin=begin+windows

                    i=begin

                else:

                    split_text=split_text+s_t+tag

                    i=i+1

            if not any([split_text in i for i in all_split_text]):

#                 print('-'*50)

#                 print(split_text)

                data[2]=split_off

                data[0][0]=split_text[:] #加上该标点符号

                split_data_list.append(copy.deepcopy(data))

                all_split_text.append(split_text)

        print(len(split_data_list))

        print('*'*50)

        for data in tqdm(split_data_list):

            text=data[0][0] #文本

            item_id=data[0][1] #id

            split_off=data[2] #切分起点

            enent_key_argument=data[1]

            #遍历每一个事件

            for event_key,arguments in enent_key_argument:

                enent_type=event_key[0]

                trigger=event_key[1]

                trigger_start_index=event_key[2]

                enent_text=text

                if trigger_start_index>=split_off and trigger_start_index<split_off+len(text):

                    assert text[trigger_start_index-split_off:trigger_start_index-split_off+len(trigger)]==trigger

                    #对每一个事件建立一个特定的文本，这个文本去掉了与非触发词相同的词

                    text_before=text[:trigger_start_index-split_off]

                    text_end=text[trigger_start_index-split_off+len(trigger):]

                    enent_text=text_before+"$"+trigger+"$"+text_end

                    t_tokens=self.tokenizer.tokenize(enent_text)

                    assert len(t_tokens)<=MAX_TEXT_LEN+10

                    mapping=rematch(enent_text,t_tokens) #获得原始的map

                    question=trigger

                    q_tokens= self.tokenizer.convert_tokens_to_ids(['[CLS]']+self.tokenizer.tokenize(question)+['[SEP]']) #20

                    if len(q_tokens)>MAX_QUESTION_LEN:

                            print('big')

                            print(len(q_tokens))

                            print(question)

                    text_token_ids = self.tokenizer.convert_tokens_to_ids(t_tokens+['[SEP]'])

                    token_ids=q_tokens+text_token_ids

                    assert len(token_ids)<=MAX_LEN

                    if len(token_ids) < MAX_LEN:

                        token_ids += [0] * (MAX_LEN- len(token_ids))

                    mapping_off=len(q_tokens)

                    labels=class_type_map_65[enent_type]

                    

                    feature=Feature(item_id=item_id,

                            or_text=text,

                            enent_text=enent_text,

                            trigger_and_start=(trigger,trigger_start_index),

                            mapping=mapping,

                            mapping_off=mapping_off,

                            token_ids=token_ids,

                            question=question,

                            labels=labels,

                            data=data[1],

                            split_off=split_off,

                            )

                    features.append(feature)



        return features

                        

    

    def __len__(self):

        return len(self.features)

    def select_tokens(self, tokens, max_num):

        if len(tokens) <= max_num:

            return tokens

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

    def __getitem__(self,index):

        feature=self.features[index]

        token_ids=torch.tensor(feature.token_ids)

        seg_ids=self.get_seg_ids(token_ids)

        labels=torch.tensor(np.array(feature.labels).astype(np.float32)).long()

        

        return token_ids,seg_ids,labels

    

    def collate_fn(self, batch):

        token_ids = torch.stack([x[0] for x in batch])

        seg_ids = torch.stack([x[1] for x in batch])

        labels=torch.stack([x[2] for x in batch])

        return token_ids, seg_ids, labels

    

    

def get_loader(df,batch_size=16,train_mode=False,val_mode=False,test_mode=False,train_val=True,recall=False):

    ds_df = zy_DataSet(copy.deepcopy(df),train_mode=train_mode,val_mode=val_mode,test_mode=test_mode,labeled=train_val,recall=recall)

    loader = torch.utils.data.DataLoader(ds_df, batch_size=batch_size, shuffle=train_mode, num_workers=0, collate_fn=ds_df.collate_fn, drop_last=train_mode)

    loader.num = len(ds_df)

    

    return loader,ds_df.features



def debug_loader():

    loader,features=get_loader(train_data[:64],train_mode=True)

    for token_ids, seg_ids,labels in loader:

        print(token_ids)

        print(seg_ids)

        print(labels)

        break

    print(len(features))
train_data[:1]
# train_loader,train_features=get_loader(train_data,train_mode=True)

# train_features[10000]
# len(train_features)
debug_loader()
from transformers import *

from torchcrf import CRF

import torch

import torch.nn as nn

import torch.nn.functional as F

import time

from tqdm import tqdm_notebook



from transformers import *

import torch

import torch.nn as nn

import torch.nn.functional as F

import collections

import time

from tqdm import tqdm_notebook
class PositionalWiseFeedForward(nn.Module):



    def __init__(self, model_dim=768, ffn_dim=2048, dropout=0.0):

        super(PositionalWiseFeedForward, self).__init__()





        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)

        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)







    def forward(self, x):

        #[b,e,s]

        output = x.transpose(1, 2)



        output = self.w2(F.relu(self.w1(output)))

        output = self.dropout(output.transpose(1, 2))



        # add residual and norm layer

        output = self.layer_norm(x + output)

        return output





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



class zy_Model(nn.Module):

    def __init__(self, n_classes=3):

        num_labels=3

        super(zy_Model, self).__init__()

        self.model_name = 'zy_Model'

        self.bert_model = BertModel.from_pretrained("../working",cache_dir=None,output_hidden_states=True)

        self.hidden_fc= nn.Sequential(nn.Linear(1024, len(class_type_map_65)))

#         self.zy_hidden_fc= nn.Sequential(nn.Linear(1024+2, num_labels),nn.ReLU(True))

#         self.zy_hidden_ans_label=nn.Sequential(nn.Linear(1024, 1))

#         self.zy_left_right=nn.Sequential(nn.Linear(1024, 4))

#         self.crf=CRF(num_labels,batch_first=True)

    

    def mask_mean(self,x,mask):

        mask_x=x*(mask.unsqueeze(-1))

        x_sum=torch.sum(mask_x,dim=1)

        re_x=torch.div(x_sum,torch.sum(mask,dim=1).unsqueeze(-1))

        return re_x

    def forward(self,ids,seg_ids,labels,is_test=False):

        attention_mask = (ids > 0)

        last_seq,pooled_output,hidden_state=self.bert_model(input_ids=ids,token_type_ids=seg_ids,attention_mask=attention_mask)

        



        outs=self.hidden_fc(pooled_output)

    

        if not is_test:

            c_left_right=nn.CrossEntropyLoss()

            loss=c_left_right(outs,labels)

            return loss

        else:

#             decode=self.crf.decode(emissions,attention_mask)

            return outs



def debug_label():

    loader,features=get_loader(train_data[:64],train_mode=True,batch_size=2)

    model=zy_Model(num_labels)

    for token_ids, seg_ids,labels in loader:

        print(token_ids.size())

        print(labels.size())

        y = model(token_ids, seg_ids,labels ,is_test=False)

        print(y)

        y = model(token_ids, seg_ids,labels ,is_test=True)

        print(y)

        print(len(y))

        break



        
# debug_label()
# model=zy_Model(num_labels)

# list(model.named_parameters())
def get_text_f1(pred_text,text):

    common_number=0

    for ch in pred_text:

        if ch in text:

            common_number+=1

    p_len=len(pred_text)

    t_len=len(text)

    P=common_number/p_len if p_len>0 else 0.

    R=common_number/t_len if t_len>0 else 0.

    

    return (2*P*R)/(P+R) if (P+R)>0 else 0.

    

def metric_fn(results,features):

    totol_number=0

    predict_number=0

    predict_score=0

    for index,feat in enumerate(features):

        true_dicts=feat.answers #真实的answers

        pred_dicts=results[index] #预测的answers

        totol_number+=len(set(list(true_dicts.values()))) #去掉重复的

        predict_number+=len(set(list(pred_dicts.values())))

        

        #为了多个表述，改为dict list

        true_dicts_list=collections.defaultdict(list)

        predict_dicts_list=collections.defaultdict(list)

        for answer,key in pred_dicts.items():

            predict_dicts_list[key].append(answer)

        for answer,key in true_dicts.items():

            true_dicts_list[key].append(answer)

        #计算论文f1，如果有多个论元，选择分数最高那一个

        for key,answer in predict_dicts_list.items():

            if key in true_dicts_list:

                true_list=true_dicts_list[key]

                ans_list=answer

                s=0.

                for t in true_list:

                    for a in ans_list:

                        s=max(s,get_text_f1(a,t))

                predict_score+=s

    P=predict_score/predict_number if predict_number>0 else 0.

    R=predict_score/totol_number if totol_number>0 else 0.

    

    f1=(2*P*R)/(P+R) if (P+R)>0 else 0.

    

    return f1,P,R

    

    

def compute_list_score(preds,trues):

    score_dict={}

    for i in range(len(preds)):

        for j in range(len(trues)):

            score_dict[(i,j)]=get_text_f1(preds[i],trues[j])

    number=min(len(preds),len(trues))

    score_dict= sorted(score_dict.items(), key=lambda d:d[1], reverse = True)

    aready1={}

    aready2={}

    s=0.

    for k,v in score_dict:

        if number>0:

            if k[0] not in aready1 and k[1] not in aready2:

                s+=v

                aready1[k[0]]=''

                aready2[k[1]]=''

                number-=1

        else:

            break

    return s

            

    

    

def metric_fn_qa(results,label_results):

    totol_number=0

    predict_number=0

    predict_score=0

    for item_id,feat in tqdm_notebook(label_results.items()):

        #feat p_feat是一个字典

        p_feat=results[item_id]

        for k,v in p_feat.items():

            predict_number+=len(list(set(v)))

        print('predict:',p_feat)

        print('real:',feat)

        print('*'*30)

        for e_role,answers in feat.items():

            totol_number+=len(list(set(answers)))

            if e_role not in p_feat:

                continue

            p_answers=list(set(p_feat[e_role])) #预测的list

            #answer是一个list

            if len(answers)==0:

                if len(p_feat[e_role])==0:

                    predict_score+=1

                continue

            answers=list(set(answers))

            predict_score+=compute_list_score(p_answers,answers)

            

    print(predict_number)

    print(totol_number)

    P=predict_score/predict_number if predict_number>0 else 0.

    R=predict_score/totol_number if totol_number>0 else 0.

    

    f1=(2*P*R)/(P+R) if (P+R)>0 else 0.

    

    return f1,P,R

    

    

def metric_fn_word(results,label_results,log=False):

    totol_number=0

    predict_number=0

    predict_score=0

    for item_id,feat in label_results.items():

        if log:

            print('id:',item_id)

            print('type_labels:',label_results[item_id])

            print('type_predict:',results[item_id])

            print('*'*50)

        #feat p_feat是一个list

        p_feat=results[item_id]

        predict_number+=len(p_feat)

        totol_number+=len(feat)

        for word in feat:

            if word in p_feat:

                predict_score+=1

            

    print(predict_number)

    print(totol_number)

    P=predict_score/predict_number if predict_number>0 else 0.

    R=predict_score/totol_number if totol_number>0 else 0.

    

    f1=(2*P*R)/(P+R) if (P+R)>0 else 0.

    

    return f1,P,R



def validation_fn(model,val_loader,val_features,is_test=False):

    model.eval()

    cls_preds=[]

    bar = tqdm_notebook(val_loader)

    for i,(ids,seg_ids,labels) in enumerate(bar):

        preds= model(ids.cuda(DEVICE),seg_ids.cuda(DEVICE),labels.cuda(DEVICE),is_test=True)

        cls_preds.append(preds.detach().cpu().numpy())

    cls_preds=np.concatenate(cls_preds)

    results=collections.defaultdict(list) #预测的结果

    label_results=collections.defaultdict(list) #真实的结果

    for index in range(len(cls_preds)):

        pred=cls_preds[index] #预测结果

        feat=val_features[index]

        item_id=feat.id

        text=feat.enent_text

        off=feat.mapping_off

        mapping=feat.mapping #映射

        data=feat.data

        trigger_and_start=feat.trigger_and_start #trigger和开始

        

        #获取真实结果

        if len(label_results[item_id])==0:

            dict_list=collections.defaultdict(list)

            for event_key,arguments in data:

                label_results[item_id].append(event_key)

        #获取预测结果

        p_enent_type=class_type_map_reverse[np.argmax(pred)]

        results[item_id].append((p_enent_type,trigger_and_start[0],trigger_and_start[1]))

        results[item_id]=list(set(results[item_id]))

    if not is_test:

        return metric_fn_word(results,label_results,True)

    else:

        return results

        

        

    

def train_model(model,train_loader,val_loader,val_features,val_loader_recall=None,val_features_recall=None,accumulation_steps=2,early_stop_epochs=2,epochs=4,model_save_path='pytorch_zy_model_true.pkl'):  

    

    losses=[]

    ########梯度累计

    batch_size = accumulation_steps*32

    

    ########早停

    early_stop_epochs=2

    no_improve_epochs=0

    

    ########优化器 学习率

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    

#     crf_p=[n for n, p in param_optimizer if (str(n).find('crf')!=-1 or str(n).find('zy')!=-1) and str(n).find('bias')==-1 ]

#     crf_p_bias=[n for n, p in param_optimizer if (str(n).find('crf')!=-1 or str(n).find('zy')!=-1) and str(n).find('bias')!=-1 ]

#     print(crf_p)

#     print(crf_p_bias)

    optimizer_grouped_parameters = [

            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.8},

            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},

#             {'params': [p for n, p in param_optimizer if n in crf_p], 'lr': 1e-4, 'weight_decay': 0.8},

#             {'params': [p for n, p in param_optimizer if n in crf_p_bias], 'lr': 1e-4,'weight_decay': 0.0}

            ]   

    

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)

    

    train_len=len(train_loader)

    

    best_vmetric=-np.inf

    tloss = []

    ans_losses=[]

    for epoch in range(1,epochs+1):

        model.train()

        bar = tqdm_notebook(train_loader)

        for i,(ids,seg_ids,labels) in enumerate(bar):

            loss= model(ids.cuda(DEVICE),seg_ids.cuda(DEVICE),labels.cuda(DEVICE),is_test=False)

            sloss=loss

            sloss.backward()

            tloss.append(loss.item())

            ans_losses.append(loss.item())

            if (i+1) % accumulation_steps == 0 or (i+1)==train_len:

                optimizer.step()

                optimizer.zero_grad()

            bar.set_postfix(loss=np.array(tloss).mean(),ans_loss=np.array(ans_losses).mean())

        

        #val

        val_f1,val_p,val_recall=validation_fn(model,val_loader,val_features)

        if val_features_recall:

            s=validation_fn(model,val_loader_recall,val_features_recall)

            print('recall f1:',s)

            losses.append(str(s))

        losses.append( 'train_loss:%.5f, f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %

            (np.array(tloss).mean(),val_f1, val_p, val_recall, best_vmetric))

        print(losses[-1])

        if val_f1>=best_vmetric:

            torch.save(model.state_dict(),model_save_path)

            best_vmetric=val_f1

            no_improve_epochs=0

            print('improve save model!!!')

        else:

            no_improve_epochs+=1

        if no_improve_epochs==early_stop_epochs:

            print('no improve score !!! stop train !!!')

            break

    return losses

            
train_loader,train_features=get_loader(train_data,batch_size=16,train_mode=True,train_val=True)

val_loader,val_features=get_loader(valid_data,batch_size=8,val_mode=True,train_val=True,recall=False)

val_loader_recall,val_features_recall=get_loader(valid_data,batch_size=8,val_mode=True,train_val=True,recall=True)
print(len(train_features))

print(len(val_features))

print(len(val_features_recall))
# model=zy_Model(num_labels).cuda(DEVICE)

# losses=train_model(model,train_loader,val_loader,val_features,val_loader_recall,val_features_recall,accumulation_steps=2,early_stop_epochs=2,epochs=7,model_save_path='right_model_ans_label.pkl')
# for l in losses:

#     print(l)
model=zy_Model(num_labels).cuda(DEVICE)

model_save_path='../input/roberta-recall-windows-right/right_model_ans_label.pkl'

# model_save_path='right_model_ans_label.pkl'

model.load_state_dict(torch.load(model_save_path))


val_f1,val_p,val_recall=validation_fn(model,val_loader,val_features)
val_f1,val_p,val_recall
val_f1,val_p,val_recall=validation_fn(model,val_loader_recall,val_features_recall)
val_f1,val_p,val_recall


test_loader,test_features=get_loader(test_data,batch_size=8,test_mode=True,train_val=False,recall=True)
len(test_features)
test_preds=validation_fn(model,test_loader,test_features,is_test=True)
val_preds=validation_fn(model,val_loader_recall,val_features_recall,is_test=True)
import pickle

with open('final_recall_large_val_cv.pkl','wb') as f:

    pickle.dump(val_preds,f)

with open('final_recall_large_test_cv.pkl','wb') as f:

    pickle.dump(test_preds,f)
test_preds