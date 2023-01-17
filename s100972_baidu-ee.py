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

import re

import copy

import random

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

                    key = role

                    #有重复的论元

                    arguments[key].append(value)

                

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
train_data[2]
event_type_roles=collections.defaultdict(list)

with open(data_path+'event_schema.json') as f:

    for l in f:

        l = json.loads(l)

        key=l['event_type']

        for role in l['role_list']:

            event_type_roles[key].append(role['role'])
event_type_roles
class_type_map={'财经/交易':0,'产品行为':1,'交往':2,'竞赛行为':3,'人生':4,'司法行为':5,'灾害/意外':6,'组织关系':7,'组织行为':8}

class_type_map_65=dict()

index=0

for _,key in id2label.items():

    if key[0] not in class_type_map_65:

        class_type_map_65[key[0]]=index

    else:

        continue

    index+=1

    

map_65_class_type=dict()

for k,v in class_type_map_65.items():

    map_65_class_type[v]=k

map_65_class_type
label2id
train_data[:2]


MAX_LEN=160

SEP_TOKEN_ID = 102

DEVICE = 'cuda'





# SEED=2020

# np.random.seed(SEED)

# torch.manual_seed(SEED)

# torch.cuda.manual_seed(SEED)

# torch.backends.cudnn.deterministic = True





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

    def __init__(self,item_id,or_text,mapping,mapping_off,token_ids,labels,entype_trigger_indexs,cls_label_65,cls_label_6,cls_label_number,cls_type,slip_off,crf_trigger,qa_start,qa_end):

        self.id=item_id

        self.or_text=or_text

        self.mapping=mapping

        self.mapping_off=mapping_off

        self.token_ids=token_ids

        self.entype_trigger_indexs=entype_trigger_indexs

        self.labels=labels

        self.cls_label_65=cls_label_65

        self.cls_label_6=cls_label_6

        self.cls_label_number=cls_label_number

        self.cls_type=cls_type

        self.slip_off=slip_off

        self.crf_trigger=crf_trigger

        self.qa_start=qa_start

        self.qa_end=qa_end

        

    def __str__(self):

        return self.__repr__()



    def __repr__(self):

        s = ""

        s += "id: %s" % (str(self.id))

        s += "or_text: %s" % (str(self.or_text))

        s += "mapping: %s" % (str(self.mapping))

        s += "mapping_off: %s" % (str(self.mapping_off))   

        s += "token_ids: %s" % (str(self.token_ids)) 

        s += "entype_trigger_indexs: %s" % (str(self.entype_trigger_indexs))

        s += "labels: %s" % (str(self.labels))

        s += "cls_label_65: %s" % (str(self.cls_label_65))

        s += "cls_label_6: %s" % (str(self.cls_label_6))

        s += "cls_label_number: %s" % (str(self.cls_label_number))

        s += "cls_type: %s" % (str(self.cls_type))

        s += "slip_off: %s" % (str(self.slip_off))

        s += "crf_trigger: %s" % (str(self.crf_trigger))

        s += "qa_start: %s" % (str(self.qa_start))

        s += "qa_end: %s" % (str(self.qa_end))

        return s



class zy_DataSet(torch.utils.data.Dataset):

    def __init__(self, data_list,train_mode=False, val_mode=False,test_mode=False,labeled=True):

        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None)

        self.train_mode = train_mode

        self.val_mode=val_mode

        self.test_mode=test_mode

        self.labeled = labeled

        self.data_list=data_list

        if self.train_mode:

            self.features =self.get_train_features(self.data_list,val_mode)

        elif self.val_mode:

            self.features =self.get_train_features(self.data_list,val_mode)

        elif self.test_mode:

            self.features =self.get_test_features(self.data_list)

        else:

            print('no features !!!')

            

    def get_feat_qa(self,q_text,text):

        pass

    

    def get_test_features(self,data_list):

        features=[]

        neg_max=20000

        neg_number=0

        all_enent_types=list(set([i for i in list(event_type_roles.keys())]))

        

        split_data_list=[]

        ###滑动窗口

#         print(len(data_list))

        for index in range(len(data_list)):

            data=data_list[index]

            text=data[0][0]

            data.append(0)

            #长度小于最大长度

            if len(text)<=MAX_LEN-2:

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

                if len(split_text+s_t)+1>=MAX_LEN-2:

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

#         print(len(split_data_list))

#         print('*'*50)

        for data in tqdm(split_data_list):

            text=data[0][0] #文本

            item_id=data[0][1] #id

            split_off=data[2]

            enent_key_argument=data[1] #[('组织关系-裁员', '裁员', 13),dict]

            cls_type=0

            cls_label_65=[0]*len(class_type_map_65)

            cls_label_6=[0]*len(class_type_map)

            cls_number=0

            #遍历每一个事件

            texts=[]

            trigger_start_indexs=[]

            triggers=[]

            #先对文本内容token

            tokens=self.tokenizer.tokenize(text)[:MAX_LEN-2]

            mapping=rematch(text,tokens)

            mapping_off=1

            tokens=['[CLS]'] +tokens +['[SEP]']

            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            assert len(token_ids)<=MAX_LEN

            if len(token_ids) < MAX_LEN:

                token_ids += [0] * (MAX_LEN- len(token_ids))

            labels=[0]*len(token_ids)

            crf_trigger=[0]*len(token_ids)

            qa_start=None

            qa_end=None

            entype_trigger_indexs=[]

            feature=Feature(item_id=item_id,

                            or_text=text,

                            mapping=mapping,

                            mapping_off=mapping_off,

                            token_ids=token_ids,

                            entype_trigger_indexs=entype_trigger_indexs,

                            labels=labels,

                            cls_label_65=cls_label_65,

                            cls_label_6=cls_label_6,

                           cls_label_number=cls_number,

                           cls_type=cls_type,

                           slip_off=split_off,

                           crf_trigger=crf_trigger,

                           qa_start=qa_start,

                           qa_end=qa_end)

            features.append(feature)

        return features

    

    def get_train_features(self,data_list,val_mode=False):

        features=[]

        neg_max=20000

        neg_number=0

        all_enent_types=list(set([i for i in list(event_type_roles.keys())]))

        split_data_list=[]

        ###滑动窗口

#         print(len(data_list))

        for index in range(len(data_list)):

            data=data_list[index]

            text=data[0][0]

            data.append(0)

            #长度小于最大长度

            if len(text)<=MAX_LEN-2:

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

                if len(split_text+s_t)+1>=MAX_LEN-2:

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

#         print(len(split_data_list))

#         print('*'*50)

        for data in tqdm(split_data_list):

            text=data[0][0] #文本

            item_id=data[0][1] #id

#             if item_id=='a9031ecc164079fc578dfa9ae59a0149':

#                 print(data)

            split_off=data[2] #切分起点

            enent_key_argument=data[1] #[('组织关系-裁员', '裁员', 13),dict]

            cls_type=0

            cls_label_65=[0]*len(class_type_map_65)

            cls_label_6=[0]*len(class_type_map)

            cls_label_number=0

            #遍历每一个事件

            texts=[]

            trigger_start_indexs=[]

            triggers=[]

            #先对文本内容token

            tokens=self.tokenizer.tokenize(text)[:MAX_LEN-2]

            mapping=rematch(text,tokens)

            mapping_off=1

            tokens=['[CLS]'] +tokens +['[SEP]']

            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            assert len(token_ids)<=MAX_LEN

            if len(token_ids) < MAX_LEN:

                token_ids += [0] * (MAX_LEN- len(token_ids))

            labels=[0]*len(token_ids)

            crf_trigger=[0]*len(token_ids)

            entype_trigger_indexs=[]

            type_set=[]

            qa_start=0

            qa_end=0

            #遍历每一个事件

            for event_key,arguments in enent_key_argument:

                enent_type=event_key[0]

#                 type_set.append(enent_type)

                trigger=event_key[1]

                trigger_start_index=event_key[2]

                trigger_token_ids=self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(trigger))

                start=None

                end=None

                for index in range(len(mapping)):

                    if mapping[index][0]+split_off==trigger_start_index:

                        assert text[mapping[index][0]:].startswith(trigger)

                        start=index+1 #因为有cls

                        break

                if start:

                    cls_label_65[class_type_map_65[enent_type]]=1

                    cls_label_6[class_type_map[enent_type.split('-')[0]]]=1

                    cls_label_number+=1

                    type_set.append(enent_type)

                    entype_trigger_indexs.append(event_key)

                    end=start+len(trigger_token_ids)

                    labels[start]=class_type_map_65[enent_type]*2+1

                    for i in range(start+1,end):

                        labels[i]=class_type_map_65[enent_type]*2+2

                    crf_trigger[start]=1

                    for i in range(start+1,end):

                        crf_trigger[i]=2

                    qa_start=start

                    qa_end=end-1

            cls_type=len(set(type_set))

            if cls_type>2:

                cls_type=2

#             cls_type

            if cls_label_number>2:

                cls_label_number=2

            feature=Feature(item_id=item_id,

                            or_text=text,

                            mapping=mapping,

                            mapping_off=mapping_off,

                            token_ids=token_ids,

                            entype_trigger_indexs=entype_trigger_indexs,

                            labels=labels,

                            cls_label_65=cls_label_65,

                            cls_label_6=cls_label_6,

                           cls_label_number=cls_label_number,

                           cls_type=cls_type,

                           slip_off=split_off,

                           crf_trigger=crf_trigger,

                           qa_start=qa_start,

                           qa_end=qa_end

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

        cls_label_65=torch.tensor(np.array(feature.cls_label_65).astype(np.float32))

        cls_label_6=torch.tensor(np.array(feature.cls_label_6).astype(np.float32))

        cls_label_number=torch.tensor(np.array(feature.cls_label_number).astype(np.float32)).long()

        cls_label_type=torch.tensor(np.array(feature.cls_type).astype(np.float32)).long()

        crf_trigger=torch.tensor(np.array(feature.crf_trigger).astype(np.float32)).long()

        qa_start=torch.tensor(np.array(feature.qa_start).astype(np.float32)).long()

        qa_end=torch.tensor(np.array(feature.qa_end).astype(np.float32)).long()

        return token_ids,seg_ids,labels,cls_label_65,cls_label_6,cls_label_number,cls_label_type,crf_trigger,qa_start,qa_end

    

    def collate_fn(self, batch):

        token_ids = torch.stack([x[0] for x in batch])

        seg_ids = torch.stack([x[1] for x in batch])

        labels=torch.stack([x[2] for x in batch])

        cls_label_65=torch.stack([x[3] for x in batch])

        cls_label_6=torch.stack([x[4] for x in batch])

        cls_label_number=torch.stack([x[5] for x in batch])

        cls_label_type=torch.stack([x[6] for x in batch])

        crf_trigger=torch.stack([x[7] for x in batch])

        qa_start=torch.stack([x[8] for x in batch])

        qa_end=torch.stack([x[8] for x in batch])

        return token_ids, seg_ids, labels,cls_label_65,cls_label_6,cls_label_number,cls_label_type,crf_trigger,qa_start,qa_end

    

    

def get_loader(df,batch_size=16,train_mode=False,val_mode=False,test_mode=False,train_val=True):

    ds_df = zy_DataSet(copy.deepcopy(df),train_mode=train_mode,val_mode=val_mode,test_mode=test_mode,labeled=train_val)

    loader = torch.utils.data.DataLoader(ds_df, batch_size=batch_size, shuffle=train_mode, num_workers=0, collate_fn=ds_df.collate_fn, drop_last=train_mode)

    loader.num = len(ds_df)

    

    return loader,ds_df.features



def debug_loader(train_data):

    loader,features=get_loader(train_data,train_mode=True)

    for token_ids, seg_ids,labels,cls_label_65,cls_label_6,cls_number,cls_type,crf_trigger,qa_start,qa_end in train_loader:

        print(token_ids)

        print(seg_ids)

        print(labels)

        print(cls_label_65)

        print(cls_number)

        print(crf_trigger)

        print(qa_start)

        print(qa_end)

        break

    print(len(features))
train_data[:1]


train_loader,train_features=get_loader(train_data[:64],train_mode=True)

len(train_features)
m={}

for t in train_features:

    if t.cls_type not in m:

        m[t.cls_type]=0

    m[t.cls_type]+=1

m
val_loader,val_features=get_loader(valid_data[:64],train_mode=True)

val_features[:20]
m_dict={}

mi=100

for f in train_features:

    if f.cls_label_number in m_dict:

        m_dict[f.cls_label_number]+=1

    else:

        m_dict[f.cls_label_number]=1

m_dict
m_dict={}

mi=100

for f in val_features:

    if f.cls_label_number in m_dict:

        m_dict[f.cls_label_number]+=1

    else:

        m_dict[f.cls_label_number]=1

m_dict
type_dict={}

for f in train_features:

    if f.cls_type>=4:

        print(f)

    if f.cls_type not in type_dict:

        type_dict[f.cls_type]=1

    else:

        type_dict[f.cls_type]+=1

type_dict
type_dict={}

for f in val_features:

    if f.cls_type not in type_dict:

        type_dict[f.cls_type]=1

    else:

        type_dict[f.cls_type]+=1

type_dict
len(train_features)
debug_loader(train_data[:64])
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




class zy_Model(nn.Module):

    def __init__(self, n_classes=3):

        num_labels=len(class_type_map_65)*2+1

        super(zy_Model, self).__init__()

        self.model_name = 'zy_Model'

        dim=1024

        self.bert_model = BertModel.from_pretrained("../working",cache_dir=None,output_hidden_states=True)

#         self.qa_hidden_fc= nn.Sequential(nn.Linear(dim, 2))

        self.hidden_fc_crf2= nn.Sequential(nn.Linear(dim, 3),nn.ReLU(True))

        self.hidden_fc= nn.Sequential(nn.Linear(dim+3, num_labels),nn.ReLU(True))

        self.zy_cls_65_hidden= nn.Sequential(nn.Linear(dim+num_labels, len(class_type_map_65)),nn.ReLU(True),nn.Linear(len(class_type_map_65), len(class_type_map_65)))

        self.crf1=CRF(num_labels,batch_first=True)

        self.zy_cls_6_hidden= nn.Sequential(nn.Linear(dim+num_labels, len(class_type_map)),nn.ReLU(True),nn.Linear(len(class_type_map), len(class_type_map)))

        self.zy_cls_number_hidden= nn.Sequential(nn.Linear(dim+num_labels, 3),nn.ReLU(True),nn.Linear(3, 3))

        self.zy_cls_type_hidden= nn.Sequential(nn.Linear(dim+num_labels, 300),nn.ReLU(True),nn.Linear(300, 3))

        self.crf2=CRF(3,batch_first=True)

        

    def mask_mean(self,x,mask):

        mask_x=x*(mask.unsqueeze(-1))

        x_sum=torch.sum(mask_x,dim=1)

        re_x=torch.div(x_sum,torch.sum(mask,dim=1).unsqueeze(-1))

        return re_x

    def forward(self,ids,seg_ids,labels,cls_label_65,cls_label_6,cls_number,cls_type,crf_trigger,qa_start,qa_end,is_test=False):

        attention_mask = (ids > 0)

        last_seq,pooled_output,hidden_state=self.bert_model(input_ids=ids,token_type_ids=seg_ids,attention_mask=attention_mask)

        

#         outs=self.qa_hidden_fc(last_seq)

#         p_start,p_end=outs.split([1,1], dim=-1)

#         p_end=p_end.squeeze()

#         p_start=p_start.squeeze()

        

        emissions2=self.hidden_fc_crf2(last_seq)

        emissions=self.hidden_fc(torch.cat((last_seq,emissions2.detach()),2))

        #分类问题

        crf_feat=self.mask_mean(emissions.detach(),attention_mask*1)

        #65

        cls_hid=torch.cat([pooled_output,crf_feat],1)

        out_65=self.zy_cls_65_hidden(cls_hid).sigmoid()

        #6

        cls_hid_6=torch.cat([pooled_output,crf_feat],1)

        out_6=self.zy_cls_6_hidden(cls_hid_6).sigmoid()

        #number

        number_feat=self.mask_mean(last_seq,attention_mask*1)

        cls_hid_number=torch.cat([number_feat,crf_feat],1)

        out_number=self.zy_cls_number_hidden(cls_hid_number)

        

        #type

        type_feat=self.mask_mean(last_seq,attention_mask*1)

        type_feat_hid_number=torch.cat([type_feat,crf_feat],1)

        out_type=self.zy_cls_type_hidden(type_feat_hid_number)

        

        if not is_test:

            criten=nn.BCELoss()

            loss=-self.crf1(emissions, labels, mask=attention_mask,reduction='mean')

            loss_crf=-self.crf2(emissions2, crf_trigger, mask=attention_mask,reduction='mean')

            cls_loss=0.

            cls_6_loss=0.

            for i in range(cls_label_65.size()[1]):

                cls_loss+=criten(out_65[:,i]**2,cls_label_65[:,i])

            for i in range(cls_label_6.size()[1]):

                cls_6_loss+=criten(out_6[:,i]**2,cls_label_6[:,i])

            c_number=nn.CrossEntropyLoss()

            c_number_loss=c_number(out_number,cls_number)

            c_type_loss=c_number(out_type,cls_type)

#             c_qaloss=(c_number(p_start,qa_start)+c_number(p_end,qa_end))/2

            return loss,cls_loss/cls_label_65.size()[1]+cls_6_loss/cls_label_6.size()[1]+c_number_loss+c_type_loss+loss_crf

        else:

            decode=self.crf1.decode(emissions,attention_mask)

            decode2=self.crf2.decode(emissions2,attention_mask)

            return decode,out_65,decode2

def debug_label():

    loader,features=get_loader(train_data[:64],train_mode=True,batch_size=2)

    model=zy_Model(num_labels)

    for token_ids, seg_ids,labels,cls_label_65,cls_label_6,cls_number,cls_type,crf_trigger,qa_start,qa_end in loader:

        print(token_ids.size())

        print(cls_number.size())

        loss,cls_loss = model(token_ids, seg_ids,labels,cls_label_65,cls_label_6,cls_number,cls_type,crf_trigger,qa_start,qa_end,is_test=False)

        print(loss)

        print(cls_loss)

        y= model(token_ids, seg_ids,labels,cls_label_65,cls_label_6,cls_number,cls_type,crf_trigger,qa_start,qa_end,is_test=True)

        print(y)

#         print(out)

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

    

def metric_fns(results,label_results,results_crf2,label_results_crf2,log=False):

    totol_number=0

    predict_number=0

    predict_score=0

    for item_id,feat in label_results.items():

#         if log:

#             print('id:',item_id)

#             print('type_labels:',label_results[item_id])

#             print('type_predict:',results[item_id])

#             print('type_labes_crf2:',label_results_crf2[item_id])

#             print('type_predict_crf2:',results_crf2[item_id])  

#             print('*'*50)

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



def is_same_crf(out,c2):

    word=out[0]

    index=out[1]

    for c in c2:

        pword=c[0]

        pindex=c[1]

        if word==pword and index==pindex:

            return True

#         elif word in pword or pword in word:

#             return (len(pword)-len(word))==(pindex-index)

    return False

        



def validation_fn(model,val_loader,val_features,is_test=False):

    model.eval()

#     torch.manual_seed(2020)

#     torch.cuda.manual_seed(2020)

    cls_65_preds=[]

    decodes=[]

    decodes2=[]

    bar = tqdm_notebook(val_loader)

    num=0

    for i,(ids,seg_ids,labels,cls_label_65,cls_label_6,cls_number,cls_type,crf_trigger,qa_start,qa_end) in enumerate(bar):

        decode,cls_pred,decode2= model(ids.cuda(DEVICE),seg_ids.cuda(DEVICE),labels.cuda(DEVICE),cls_label_65.cuda(DEVICE),cls_label_6.cuda(DEVICE),cls_number.cuda(DEVICE),cls_type.cuda(DEVICE),crf_trigger.cuda(DEVICE),qa_start.cuda(DEVICE),qa_end.cuda(DEVICE),is_test=True)

        cls_65_preds.append(cls_pred.detach().cpu().numpy())

#         print(len(decode))

#         for de in decode:

#             print(num)

#             num+=1

#             print(de)

        decodes.extend(decode)

        decodes2.extend(decode2)

    cls_65_preds=np.concatenate(cls_65_preds)

    results=collections.defaultdict(list) #预测的结果

    label_results=collections.defaultdict(list) #真实的结果

    word_results=collections.defaultdict(list) #预测的结果

    label_word_results=collections.defaultdict(list) #预测的结果

    results_crf2=collections.defaultdict(list)

    label_results_crf2=collections.defaultdict(list)

    for index in range(len(decodes)):

        preds=cls_65_preds[index] #预测概率

        decode=decodes[index] #解码序列

        decode2=decodes2[index] #解码序列

        feat=val_features[index]

        item_id=feat.id

        text=feat.or_text

        off=feat.mapping_off

        split_off=feat.slip_off

        mapping=feat.mapping #映射

        entype_trigger_indexs=feat.entype_trigger_indexs #[(type,'word','index')]

        p_entype_trigger_indexs=[] #预测的[(type,'word','index')]

        p_entype_crf2_indexs=[]

        starting=False

        off=1

        

        #crf2

        p_crf2_words=[]

        for i,label in enumerate(decode2[off:len(mapping)+off]):

            if label > 0:

                ch = text[mapping[i][0]:mapping[i][-1] + 1] #当前预测的字

                #是实体开始

                if label==1:

                    starting = True

                    p_crf2_words.append(([i],'trigger')) #事件类型

                elif starting:

                    p_crf2_words[-1][0].append(i) #是预测中间的字

                else:

                    starting = False

            else:

                starting=False

        #获取预测二元组

        for w,l in p_crf2_words:

            p_entype_crf2_indexs.append((text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1],split_off+mapping[w[0]][0]))

        label_results_crf2[item_id].extend([(i[1],i[2]) for i in entype_trigger_indexs])

        results_crf2[item_id].extend(p_entype_crf2_indexs)

        label_results_crf2[item_id]=list(set(label_results_crf2[item_id]))

        results_crf2[item_id]=list(set(results_crf2[item_id]))

        #crf2 end

        p_triiger_words=[]

        starting=False

        for i,label in enumerate(decode[off:len(mapping)+off]):

            if label > 0:

                ch = text[mapping[i][0]:mapping[i][-1] + 1] #当前预测的字

                #是实体开始

                if label%2==1:

                    starting = True

                    p_triiger_words.append(([i],map_65_class_type[label//2])) #事件类型

                elif starting:

                    p_triiger_words[-1][0].append(i) #是预测中间的字

                else:

                    starting = False

            else:

                starting=False

        #获取预测结果三元组

        ans_triigger=[]

        for w,l in p_triiger_words:

            p_entype_trigger_indexs.append((l,text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1],split_off+mapping[w[0]][0]))

        label_word_results[item_id].extend(entype_trigger_indexs)

        word_results[item_id].extend(p_entype_trigger_indexs)

        label_word_results[item_id]=list(set(label_word_results[item_id]))

        word_results[item_id]=list(set(word_results[item_id]))

        ### types

        pred_types=[]

        for pi,p in enumerate(preds):

            if p>0.5:

                pred_types.append(map_65_class_type[pi])

        results[item_id].extend(pred_types)

        results[item_id]=list(set(results[item_id]))

        label_results[item_id].extend([i[0] for i in entype_trigger_indexs])

        label_results[item_id]=list(set(label_results[item_id]))

    #只要在3中出现过的

    results_with2=collections.defaultdict(list) 

    for id,out in word_results.items():

        crf2_out=results_crf2[id]

        right=[]

        for o in out:

            if is_same_crf((o[1],o[2]),crf2_out):

                right.append(o)

        results_with2[id].extend(right)

    res=metric_fns(word_results,label_word_results,results_crf2,label_results_crf2,False)

    if not is_test:

#         res_with2=metric_fn_word(results_with2,label_word_results)

#         print('cls:',metric_fn_word(results,label_results)) #分类的结果

#         print('crf2:',metric_fn_word(results_crf2,label_results_crf2)) #crf2的结果

#         print('without crf2:',res)

        return res

    else:

        return results,word_results,label_word_results,res

        

        

    

def train_model(model,train_loader,val_loader,val_features,accumulation_steps=2,early_stop_epochs=2,epochs=4,model_save_path='pytorch_zy_model_true.pkl'):  

    

    losses=[]

    ########梯度累计

    batch_size = accumulation_steps*32

    

    ########早停

    no_improve_epochs=0

    

    ########优化器 学习率

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    

    crf_p=[n for n, p in param_optimizer if str(n).find('crf1')!=-1]

    crf_p2=[n for n, p in param_optimizer if str(n).find('crf2')!=-1 and str(n).find('hidden_fc_crf')==-1]

    crf2_bias=[n for n, p in param_optimizer if str(n).find('zy')!=-1 and str(n).find('bias')!=-1]

    crf2_w=[n for n, p in param_optimizer if str(n).find('zy')!=-1 and str(n).find('bias')==-1]

#     crf_p=[n for n, p in param_optimizer if str(n).find('crf')!=-1]

#     print(crf_p)

#     print(crf_p2)

#     print(crf2_bias)

#     print(crf2_w)

    optimizer_grouped_parameters = [

            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and n not in crf_p+crf2_bias+crf_p2+crf2_w], 'weight_decay': 0.8},

            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and n not in crf_p+crf2_bias+crf_p2+crf2_w], 'weight_decay': 0.0},

            {'params': [p for n, p in param_optimizer if n in crf_p], 'lr': 15e-4,'weight_decay': 0.8},

            {'params': [p for n, p in param_optimizer if n in crf_p2], 'lr': 1e-4,'weight_decay': 0.8},

            {'params': [p for n, p in param_optimizer if n in crf2_bias], 'lr': 15e-4,'weight_decay': 0.0},

            {'params': [p for n, p in param_optimizer if n in crf2_w], 'lr': 15e-4,'weight_decay': 0.8}

            ]   

    

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)

    

    train_len=len(train_loader)

    

    best_vmetric=-np.inf

    tloss = []

    cls_65_loss=[]

    for epoch in range(1,epochs+1):

        model.train()

        bar = tqdm_notebook(train_loader)

        for i,(ids,seg_ids,labels,cls_label_65,cls_label_6,cls_number,cls_type,crf_trigger,qa_start,qa_end) in enumerate(bar):

            loss,cls_loss_65 = model(ids.cuda(DEVICE),seg_ids.cuda(DEVICE),labels.cuda(DEVICE),cls_label_65.cuda(DEVICE),cls_label_6.cuda(DEVICE),cls_number.cuda(DEVICE),cls_type.cuda(DEVICE),crf_trigger.cuda(DEVICE),qa_start.cuda(DEVICE),qa_end.cuda(DEVICE),is_test=False)

            sloss=loss+cls_loss_65

            sloss.backward()

            tloss.append(loss.item())

            cls_65_loss.append(cls_loss_65.item())

            if (i+1) % accumulation_steps == 0 or (i+1)==train_len:

                optimizer.step()

                optimizer.zero_grad()

            bar.set_postfix(tloss=np.array(tloss).mean(),cls_65_loss=np.array(cls_65_loss).mean())

        

        #val

        val_f1,val_p,val_recall=validation_fn(model,val_loader,val_features)

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
from sklearn.model_selection import KFold

FOLD=5

kf = KFold(n_splits=FOLD, shuffle=True,random_state=2019)

test_loader,test_features=get_loader(test_data,batch_size=8,test_mode=True,train_val=False) #测试集

val_loader,val_features=get_loader(valid_data,batch_size=8,val_mode=True,train_val=True) #验证集

log_losses=[]



train_recall=collections.defaultdict(list)

train_recall_real=collections.defaultdict(list)

val_recall=collections.defaultdict(list)

val_recall_real=collections.defaultdict(list)



test_recall=collections.defaultdict(list)

for i,(train_index , test_index) in enumerate(kf.split(train_data)):

    if i+1<4:

        continue

    print(str(i+1),'*'*50)

    tra=[train_data[index] for index in train_index]

    valid=[train_data[index] for index in test_index]

    print(len(tra))

    print(len(valid))

    

    tra_loader,tra_features=get_loader(tra,batch_size=16,train_mode=True,train_val=True)

    valid_loader,valid_features=get_loader(valid,batch_size=8,val_mode=True,train_val=True)

    

    model=zy_Model(num_labels).cuda(DEVICE)

    model_save_path='../input/shijianchouquqa/cv_recall{}.pkl'.format(i+1)

#     losses=train_model(model,tra_loader,valid_loader,valid_features,accumulation_steps=2,early_stop_epochs=2,epochs=20,model_save_path='cv_recall{}.pkl'.format(i+1))

#     log_losses.extend(losses)

    #加载最好模型

#     model_save_path='cv_recall{}.pkl'.format(i+1)

    model.load_state_dict(torch.load(model_save_path))

    

    #off 预测

#     _,off_recall,off_real,_=validation_fn(model,valid_loader,valid_features,is_test=True)

#     for item,words in off_recall.items():

#         train_recall[item].extend(words)

#         train_recall_real[item].extend(off_real[item])

#         train_recall_real[item]=list(set(train_recall_real[item]))

#         train_recall[item]=list(set(train_recall[item]))

    

#     #val 预测

#     _,val_off_recall,val_off_real,valid_score=validation_fn(model,val_loader,val_features,is_test=True)

#     log_losses.append(str(valid_score))

#     print('val score:',log_losses[-1])

#     for item,words in val_off_recall.items():

#         val_recall[item].extend(words)

#         val_recall_real[item].extend(val_off_real[item])

#         val_recall_real[item]=list(set(val_recall_real[item]))

#         val_recall[item]=list(set(val_recall[item]))

#     n=0

#     for k,v in val_recall.items():

#         n+=len(v)

#     print('val number:',n)

    #test预测

    _,test_off_recall,_,_=validation_fn(model,test_loader,test_features,is_test=True)

    for item,words in test_off_recall.items():

        test_recall[item].extend(words)

        test_recall[item]=list(set(test_recall[item]))

    print(str(i+1),'-'*50)

    torch.cuda.empty_cache()

#     if i+1>3:

#         break
metric_fn_word(train_recall,train_recall_real)
metric_fn_word(val_recall,val_recall_real)
for l in log_losses:

    print(l)
val_recall
test_recall
import pickle

# with open('cv_recall_large_train.pkl','wb') as f:

#     pickle.dump(train_recall,f)

# with open('cv_recall_large_train_real.pkl','wb') as f:

#     pickle.dump(train_recall_real,f)

# with open('cv_recall_large_val.pkl','wb') as f:

#     pickle.dump(val_recall,f)

with open('cv_recall_large_test.pkl','wb') as f:

    pickle.dump(test_recall,f)
# train_loader,train_features=get_loader(train_data,batch_size=16,train_mode=True,train_val=True)

# val_loader,val_features=get_loader(valid_data,batch_size=8,val_mode=True,train_val=True)
# len(train_features)
# model=zy_Model(num_labels).cuda(DEVICE)

# losses=train_model(model,train_loader,val_loader,val_features,accumulation_steps=2,early_stop_epochs=2,epochs=20,model_save_path='right_recall1_2decode_windows5.pkl')
# for l in losses:

#     print(l)
# model=zy_Model(num_labels).cuda(DEVICE)

# model_save_path='right_recall1_2decode_windows5.pkl'

# model.load_state_dict(torch.load(model_save_path))
# val_f1,val_p,val_recall=validation_fn(model,val_loader,val_features)
# val_f1,val_p,val_recall
# val_result,val_word_result=validation_fn(model,val_loader,val_features,is_test=True)
# val_word_result
# for k,v in val_word_result.items():

#     if len(v)==0:

#         print(k)

#         print(val_result[k])

#         if len(val_result[k])!=0:

#             for typ in val_result[k]:

#                 word=typ.split('-')[1]

#                 val_word_result[k].append((typ,word,-1))
# import pickle

# with open('right_recall1_val_large.pkl','wb') as f:

# #     pickle.dump(val_word_result,f)
# for k,v in test_words_result.items():

#     if len(v)==0:

#         print(k)

#         print(test_result[k])

#         if len(test_result[k])!=0:

#             for typ in test_result[k]:

#                 word=typ.split('-')[1]

#                 test_words_result[k].append((typ,word,-1))
# test_words_result
# test_final_result=collections.defaultdict(list) 

# for id,type in test_result.items():

#     if len(type)==0:

#         print(id)

#         print(type)

#     words=test_words_result[id]

#     types=[]

#     for k,_ in type.items():

#         types.append(k)

#     test_final_result[id].append(types)

#     test_final_result[id].append(words)
# import pickle

# with open('right_recall1_test_large.pkl','wb') as f:

#     pickle.dump(test_words_result,f)