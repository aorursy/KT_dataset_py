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

BERT_MODEL_PATH = '../input/bert-pretrained-models/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12/'

convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(

    BERT_MODEL_PATH + 'bert_model.ckpt',

BERT_MODEL_PATH + 'bert_config.json',

WORK_DIR + 'pytorch_model.bin')



shutil.copyfile(BERT_MODEL_PATH + 'bert_config.json', WORK_DIR + 'config.json')
import pandas as pd

import numpy as np

import json

import random

from tqdm import tqdm

import collections

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

            D.append(((l['text'],l['id']),event_list))

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

            D.append(((l['text'],l['id']),event_list))

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


MAX_TEXT_LEN = 170

MAX_LEN=185

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



seed = 2020

seed_everything(seed)
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
!ls ../input/right-recall1/

import pickle

with open('../input/right-recall1-0862/right_recall1_val.pkl','rb') as f:

    val_recall_data=pickle.load(f)

with open('../input/right-recall1-0862/right_recall1_test.pkl','rb') as f:

    test_recall_data=pickle.load(f)
print(len(val_recall_data))

print(len(test_recall_data))
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

    def __init__(self,item_id,or_text,enent_text,trigger_and_start,mapping,mapping_off,event_type_role,token_ids,start,end,question,ans,data,is_answer):

        self.id=item_id

        self.or_text=or_text

        self.enent_text=enent_text

        self.trigger_and_start=trigger_and_start

        self.mapping=mapping

        self.mapping_off=mapping_off

        self.event_type_role=event_type_role

        self.token_ids=token_ids

        self.question=question

        self.start=start

        self.end=end

        self.ans=ans

        self.data=data

        self.is_answer=is_answer

        

    def __str__(self):

        return self.__repr__()



    def __repr__(self):

        s = ""

        s += "id: %s" % (str(self.id))

        s += "or_text: %s" % (str(self.or_text))

        s += "enent_text: %s" % (str(self.enent_text))

        s += "mapping: %s" % (str(self.mapping))

        s += "mapping_off: %s" % (str(self.mapping_off))

        s += "event_type_role: %s" % (str(self.event_type_role)) 

        s += "token_ids: %s" % (str(self.token_ids)) 

        s += "question: %s" % (str(self.question))

        s += "start: %s" % (str(self.start))

        s += "end: %s" % (str(self.end))

        s += "data: %s" % (str(self.data))

        s += "is_answer: %s" % (str(self.is_answer))

        return s



class zy_DataSet(torch.utils.data.Dataset):

    def __init__(self, data_list,train_mode=False, val_mode=False,test_mode=False,labeled=True,recall_tag=False):

        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None)

        self.train_mode = train_mode

        self.val_mode=val_mode

        self.test_mode=test_mode

        self.labeled = labeled

        self.recall_tag=recall_tag

        if self.train_mode:

            self.features =self.get_train_features(data_list)

        elif self.val_mode:

            if self.recall_tag:

                self.features=self.get_recall_features(data_list,val_recall_data)

            else:

                self.features =self.get_train_features(data_list)

        elif self.test_mode:

            self.features =self.get_recall_features(data_list,test_recall_data)

        

    def get_recall_features(self,data_list,recall_data):

        features=[]

        for data in tqdm(data_list):

            text=data[0][0] #文本

            item_id=data[0][1] #id

            enent_key_argument=recall_data[item_id]

            for event_key in enent_key_argument:

                enent_type=event_key[0]

                trigger=event_key[1]

                trigger_start_index=event_key[2]

                t_tokens=None

                mapping=None

                if trigger_start_index!=-1:

                    assert text[trigger_start_index:trigger_start_index+len(trigger)]==trigger

                    #对每一个事件建立一个特定的文本，这个文本去掉了与非触发词相同的词

                    text_before=text[:trigger_start_index].replace(trigger,'事件')

                    text_end=text[trigger_start_index+len(trigger):].replace(trigger,'事件')

                    enent_text=text_before+"$"+trigger+"$"+text_end

                    if len(enent_text)>156:

                        enent_text="$"+trigger+"$"

                        if len(text_end)<=78:

                            enent_text=enent_text+text_end

                            text_before=text_before[-156+len(enent_text):]

                            enent_text=text_before+enent_text

                        elif len(text_before)<=78:

                            enent_text=text_before+enent_text

                            text_end=text_end[:156-len(enent_text)]

                            enent_text=enent_text+text_end

                        else:

                            enent_text=text_before[-78:]+trigger+text_end[:78]

                            text_before=text_before[-78:]

                            text_end=text_end[:78]

                    t_tokens=self.tokenizer.tokenize(enent_text)[:MAX_TEXT_LEN]

                    mapping=rematch(enent_text,t_tokens) #获得原始的map

                else:

                    enent_text=text

                    t_tokens=self.tokenizer.tokenize(enent_text)[:MAX_TEXT_LEN]

                    mapping=rematch(enent_text,t_tokens) #获得原始的map

                roles=event_type_roles[enent_type]

                for role in roles:

                    question=enent_type+'-'+role

                    q_tokens=self.tokenizer.tokenize(question)

                    tokens= ['[CLS]'] +q_tokens +['[SEP]']+t_tokens[:MAX_LEN-len(q_tokens)-3]+['[SEP]']

                    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                    assert len(token_ids)<=MAX_LEN

                    if len(token_ids) < MAX_LEN:

                        token_ids += [0] * (MAX_LEN- len(token_ids))

                    mapping_off=len(q_tokens)+2 #映射转移，方便解码

                    start=0

                    end=0

                    ans=[]

                    is_answer=0

                    feature=Feature(item_id=item_id,

                            enent_text=enent_text,

                            or_text=text,

                            mapping=mapping,

                            mapping_off=mapping_off,

                            trigger_and_start=(trigger,trigger_start_index),

                            token_ids=token_ids,

                            event_type_role=(enent_type,role),

                            start=start,

                            end=end,

                            question=question,

                            ans=ans,

                            data=data[1],

                            is_answer=is_answer)

                    features.append(feature)

        return features

        

    

    def get_train_features(self,data_list):

        features=[]

        for data in tqdm(data_list):

            text=data[0][0] #文本

            item_id=data[0][1] #id

            enent_key_argument=data[1] #[('组织关系-裁员', '裁员', 13),dict]

            

            #遍历每一个事件

            for event_key,arguments in enent_key_argument:

                enent_type=event_key[0]

                trigger=event_key[1]

                trigger_start_index=event_key[2]

                

                assert text[trigger_start_index:trigger_start_index+len(trigger)]==trigger

                #对每一个事件建立一个特定的文本，这个文本去掉了与非触发词相同的词

                text_before=text[:trigger_start_index].replace(trigger,'事件')

                text_end=text[trigger_start_index+len(trigger):].replace(trigger,'事件')

                enent_text=text_before+"$"+trigger+"$"+text_end

                if len(enent_text)>156:

                    enent_text="$"+trigger+"$"

                    if len(text_end)<=78:

                        enent_text=enent_text+text_end

                        text_before=text_before[-156+len(enent_text):]

                        enent_text=text_before+enent_text

                    elif len(text_before)<=78:

                        enent_text=text_before+enent_text

                        text_end=text_end[:156-len(enent_text)]

                        enent_text=enent_text+text_end

                    else:

                        enent_text=text_before[-78:]+trigger+text_end[:78]

                        text_before=text_before[-78:]

                        text_end=text_end[:78]

                

                

                

                t_tokens=self.tokenizer.tokenize(enent_text)[:MAX_TEXT_LEN]

                mapping=rematch(enent_text,t_tokens) #获得原始的map

                

                #获取这个事件可能的论元，roles

                roles=event_type_roles[enent_type]

                #对每一个role建立一个实例

                for role in roles:

                    question='触发词是'+trigger+'-'+enent_type+'-'+role

                    q_tokens=self.tokenizer.tokenize(question)

                    tokens= ['[CLS]'] +q_tokens +['[SEP]']+t_tokens[:MAX_LEN-len(q_tokens)-3]+['[SEP]']

                    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                    assert len(token_ids)<=MAX_LEN

                    if len(token_ids) < MAX_LEN:

                        token_ids += [0] * (MAX_LEN- len(token_ids))

                    mapping_off=len(q_tokens)+2 #映射转移，方便解码

                    start=0

                    end=0

                    ans=arguments[role]

                    is_answer=0

                    #打标

                    if len(ans)>1:

                        continue

                    for an in ans:

                        a_token_ids=self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(an))

                        start_index=search(a_token_ids,token_ids)

                        if start_index!=-1:

                            start=start_index

                            end=start_index+len(a_token_ids)-1

                            is_answer=1

                            break

                    

                    feature=Feature(item_id=item_id,

                            enent_text=enent_text,

                            or_text=text,

                            mapping=mapping,

                            mapping_off=mapping_off,

                            trigger_and_start=(trigger,trigger_start_index),

                            token_ids=token_ids,

                            event_type_role=(enent_type,role),

                            start=start,

                            end=end,

                            question=question,

                            ans=ans,

                            data=data[1],

                            is_answer=is_answer)

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

        start=torch.tensor(np.array(feature.start).astype(np.float32)).long()

        end=torch.tensor(np.array(feature.end).astype(np.float32)).long()

        is_answer=torch.tensor(np.array(feature.is_answer).astype(np.float32))

        return token_ids,seg_ids,start,end,is_answer

    

    def collate_fn(self, batch):

        token_ids = torch.stack([x[0] for x in batch])

        seg_ids = torch.stack([x[1] for x in batch])

        start=torch.stack([x[2] for x in batch])

        end=torch.stack([x[3] for x in batch])

        is_answer=torch.stack([x[4] for x in batch])

    

        return token_ids, seg_ids, start ,end,is_answer

    

    

def get_loader(df,batch_size=16,train_mode=False,val_mode=False,test_mode=False,recall_tag=False):

    ds_df = zy_DataSet(df,train_mode=train_mode,val_mode=val_mode,test_mode=test_mode, labeled=train_mode,recall_tag=recall_tag)

    loader = torch.utils.data.DataLoader(ds_df, batch_size=batch_size, shuffle=train_mode, num_workers=0, collate_fn=ds_df.collate_fn, drop_last=train_mode)

    loader.num = len(ds_df)

    

    return loader,ds_df.features



def debug_loader(df):

    loader,features=get_loader(train_data,batch_size=64,train_mode=True,recall_tag=False)

    for token_ids, seg_ids,start,end,is_answer in train_loader:

        print(token_ids)

        print(seg_ids)

        print(start)

        print(end)

        print(is_answer)

        break

    print(len(features))
train_data[:1]
train_loader,train_features=get_loader(train_data[:64],train_mode=True)

train_features[:20]
len(train_features)
debug_loader(train_loader)
! pip install pytorch-crf
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

        super(zy_Model, self).__init__()

        self.model_name = 'zy_Model'

        self.bert_model = BertModel.from_pretrained("../working",cache_dir=None,output_hidden_states=True)

        self.hidden_fc= nn.Sequential(nn.Linear(768, 2))

        self.zy_hidden_is_answer= nn.Sequential(nn.Linear(768, 1))

    

    def mask_mean(self,x,mask):

        mask_x=x*(mask.unsqueeze(-1))

        x_sum=torch.sum(mask_x,dim=1)

        re_x=torch.div(x_sum,torch.sum(mask,dim=1).unsqueeze(-1))

        return re_x

    def forward(self,ids,seg_ids,start,end,is_answer,is_test=False):

        attention_mask = (ids > 0)

        last_seq,pooled_output,hidden_state=self.bert_model(input_ids=ids, attention_mask=attention_mask)

        outs=self.hidden_fc(last_seq)

        p_start,p_end=outs.split([1,1], dim=-1)

        p_end=p_end.squeeze()

        p_start=p_start.squeeze()

        

        out_is_answer=self.zy_hidden_is_answer(pooled_output).sigmoid()

#         print(out_is_answer.size())

#         print(is_answer.size())

        #add

        if not is_test:

            #add

            criterion = nn.CrossEntropyLoss()

            loss=criterion(p_start,start)+criterion(p_end,end)

            criterion2=nn.BCELoss()

            loss2=criterion2(out_is_answer,is_answer)

            

            return loss/2+loss2

        else:

            return p_start,p_end



def debug_label():

    loader,features=get_loader(train_data[:64],train_mode=True,recall_tag=False)

    model=zy_Model(num_labels)

    for token_ids, seg_ids,start,end,is_answer in train_loader:

        print(token_ids.size())

#         print(type(is_answer))

        y = model(token_ids, seg_ids,start,end ,is_answer,is_test=False)

        print(y)

        y = model(token_ids, seg_ids,start,end ,is_answer,is_test=True)

        print(y)

        print(len(y))

        break
debug_label()
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

            

    



def n_softmax(x):

    x_row_max = x.max(axis=-1)

    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])

    x = x - x_row_max

    x_exp = np.exp(x)

    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])

    softmax = x_exp / x_exp_row_sum

    return softmax





def get_ans(starts,ends,feat):

    max_len=feat.mapping_off+len(feat.mapping)

    starts=n_softmax(np.array(starts[:max_len]))

    ends=n_softmax(np.array(ends[:max_len]))

    start_end, score = None, -1

    for start, p_start in enumerate(starts):

        for end, p_end in enumerate(ends):

            if end >= start:

                if p_start * p_end > score:

                    start_end = (start, end)

                    score = p_start * p_end

    start, end = start_end

    return start,end



def metric_cls_all(pred,true):

    pred=(pred)>0.5*1

    return len([1 for i in range(len(true)) if pred[i]==true[i]])/len(true)





def metric_fn_qa(results,label_results):

    totol_number=0

    predict_number=0

    predict_score=0

    for item_id,feat in tqdm_notebook(label_results.items()):

        #feat p_feat是一个字典

        p_feat=results[item_id]

        for k,v in p_feat.items():

            predict_number+=len(list(set(v)))

#         print('predict:',p_feat)

#         print('real:',feat)

#         print('*'*30)

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





def validation_fn(model,val_loader,val_features,is_test=False):

    model.eval()

#     torch.manual_seed(2020)

#     torch.cuda.manual_seed(2020)

    p_ss=[] #预测的答案下标概率

    p_ee=[] #预测的答案末尾概率

    bar = tqdm_notebook(val_loader)

    for i,(ids,seg_ids,start,end,is_answer) in enumerate(bar):

        starts,ends = model(ids.cuda(DEVICE),seg_ids.cuda(DEVICE),start.cuda(DEVICE),end.cuda(DEVICE),is_answer.cuda(DEVICE),is_test=True)

        starts=starts.detach().cpu().numpy()

        ends=ends.detach().cpu().numpy()

        p_ss.append(starts)

        p_ee.append(ends)

    #获取答案下标

    p_ss=np.concatenate(p_ss)

    p_ee=np.concatenate(p_ee)

    

    p_starts=[] #预测的下标

    p_ends=[]

    for index in tqdm_notebook(range(p_ss.shape[0])):

        st,et=get_ans(p_ss[index,:],p_ee[index,:],val_features[index])

        p_starts.append(st)

        p_ends.append(et)

    

    results=collections.defaultdict(dict) #预测的结果

    label_results=collections.defaultdict(dict) #真实的结果

    for index in tqdm_notebook(range(len(p_ends))):

        feat=val_features[index]

        item_id=feat.id

        text=feat.enent_text

        off=feat.mapping_off

        mapping=feat.mapping #映射

        event_type_role=feat.event_type_role #(enent_type,role)

        ans=feat.ans

        data=feat.data

        p_st=p_starts[index] #预测的开始

        p_en=p_ends[index]  #预测的结束

        p_answer=[]

        #在答案区间

        if p_st-off>=0 and len(mapping)>p_en-off:

            s=p_st-off

            e=p_en-off

            if s>=0:

                p_answer.append(text[mapping[s][0]:mapping[e][-1]+1])

        #有答案

        if len(p_answer)>0:

            if len(results[item_id])>0:

                results[item_id][event_type_role].extend(p_answer)

            #没有创建答案字典

            else:

                dict_list=collections.defaultdict(list)

                dict_list[event_type_role]=p_answer

                results[item_id]=dict_list

        #获取真实结果

        if len(label_results[item_id])==0:

            dict_list=collections.defaultdict(list)

            for event_key,arguments in data:

                for role,argument in arguments.items():

                    if len(argument)!=0:

                        dict_list[(event_key[0],role)].extend(argument)

            label_results[item_id]=dict_list

        

    

    if not is_test:

        return metric_fn_qa(results,label_results)

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

    

#     crf_p=[n for n, p in param_optimizer if str(n).find('crf')!=-1]

#     print(crf_p)

    optimizer_grouped_parameters = [

            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.8},

            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},

#             {'params': [p for n, p in param_optimizer if n in crf_p], 'lr': 1e-3}

            ]   

    

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)

    

    train_len=len(train_loader)

    

    best_vmetric=-np.inf

    tloss = []

    

    for epoch in range(1,epochs+1):

        model.train()

        bar = tqdm_notebook(train_loader)

        for i,(ids,seg_ids,start,end,is_answer) in enumerate(bar):

            loss = model(ids.cuda(DEVICE),seg_ids.cuda(DEVICE),start.cuda(DEVICE),end.cuda(DEVICE),is_answer.cuda(DEVICE),is_test=False)

            loss.backward()

            tloss.append(loss.item())

            if (i+1) % accumulation_steps == 0 or (i+1)==train_len:

                optimizer.step()

                optimizer.zero_grad()

            bar.set_postfix(loss=np.array(tloss).mean(),score='not_compute')

        

        #val

        val_f1,val_p,val_recall=validation_fn(model,val_loader,val_features)

        if val_loader_recall:

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

            
train_loader,train_features=get_loader(train_data,batch_size=32,train_mode=True,recall_tag=False)

val_loader,val_features=get_loader(valid_data,batch_size=16,val_mode=True,recall_tag=False)

val_loader_recall,val_features_recall=get_loader(valid_data,batch_size=16,val_mode=True,recall_tag=True)
model=zy_Model(num_labels).cuda(DEVICE)

losses=train_model(model,train_loader,val_loader,val_features,val_loader_recall=val_loader_recall,val_features_recall=val_features_recall,accumulation_steps=1,early_stop_epochs=2,epochs=20,model_save_path='pytorch_zy_model_qa_new1.pkl')
for l in losses:

    print(l)
model=zy_Model(num_labels).cuda(DEVICE)

model_save_path='pytorch_zy_model_qa_new1.pkl'

model.load_state_dict(torch.load(model_save_path))
val_f1,val_p,val_recall=validation_fn(model,val_loader_recall,val_features_recall)

val_f1,val_p,val_recall
val_f1,val_p,val_recall=validation_fn(model,val_loader,val_features)

val_f1,val_p,val_recall
test_loader,test_features=get_loader(test_data,batch_size=16,test_mode=True,recall_tag=False)
test_features[:6]
len(test_features)
test_preds=validation_fn(model,test_loader,test_features,is_test=True)
for a in test_preds.items():

    print(a)
test_preds['6eab49e6643d5a5f50037be79937d7cb']
len(test_preds)
def predict_to_file_qa(results,in_file,out_file):

    """预测结果到文件，方便提交

    """

    fw = open(out_file, 'w', encoding='utf-8')

    with open(in_file) as fr:

        for index,l in enumerate(fr):

            l = json.loads(l)

            event_list = []

            arguments=results[l['id']]

            for k, v in arguments.items():

                v=list(set(v))

#                 print(k)

#                 print(v)

                for vi in v:

                    event_list.append({

                        'event_type': k[0],

                        'arguments': [{

                            'role': k[1],

                            'argument': vi

                        }]

                    })

            l['event_list'] = event_list

#             print(l)

            l = json.dumps(l, ensure_ascii=False)

            fw.write(l + '\n')

    fw.close()
predict_to_file_qa(test_preds,data_path+'test1.json','right_predict_qa1.json')