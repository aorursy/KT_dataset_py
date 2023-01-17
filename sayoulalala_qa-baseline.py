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
!ls ../input/
import pandas as pd

data_path='../input/qa-4013/'

A_train_recall = pd.read_csv(data_path+"A_train_recall.csv")

NCPPolicies_context = pd.read_csv(data_path+"NCPPolicies_context.csv")

NCPPolicies_train_20200301 = pd.read_csv(data_path+"NCPPolicies_train_20200301.csv",sep='\t')



train=A_train_recall.merge(NCPPolicies_train_20200301[['id','question','answer']],on='id',how='left')

train=train.merge(NCPPolicies_context,on='docid',how='left')



print(train.label.value_counts())

print(train.shape)



train.head()
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None)

a=tokenizer.tokenize('我爱你中国，，，1975123   ')

print(a)

# a=tokenizer.convert_tokens_to_ids(a)

# print(a)

# a=tokenizer.convert_ids_to_tokens(a,skip_special_tokens=True)

# print(a)

a.append('[UNK]')

a=tokenizer.convert_tokens_to_string(a)

print(a)
Max_query_len=59

Max_answer_len=250

Split_windows=150

class Split_doc:

    def __init__(self,df):

        self.query=df.question

        self.answer=df.answer

        self.docs=df.text

        self.query_in_answer=df.label

        self.doc_num=1

        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None)

        

    def split(self):

        

        self.docs=self.prepare(self.docs)

        docs_token=self.tokenizer.tokenize(self.docs)

        answer_token=self.tokenizer.tokenize(self.answer)[-Max_answer_len:]

        query_token=self.tokenizer.tokenize(self.query)[:Max_query_len]

        

        len_docs_token=len(docs_token)

        len_answer_token=len(answer_token)

        len_query_token=len(query_token)

        

        #找到answer开始位置

        answer_token_s=' '.join(answer_token)

        answer_start=-1

        answer_end=-1

        for s in range(len_docs_token-len_answer_token):

            if ' '.join(docs_token[s:s+len_answer_token])==answer_token_s:

                answer_start=s

                answer_end=s+len_answer_token-1

                break

        

        

        #开始返回

        split_docs=[]

        split_query=[]

        split_answer=[]

        doc_start=[]

        doc_end=[]

        

        begin=0

        end=len_answer_token

        Max_docs_len=512-Max_query_len-3

        #文本比较短

        if len_docs_token<=Max_docs_len:

            split_docs.append(' '.join(docs_token))

            split_query.append(' '.join(query_token))

            split_answer.append(' '.join(answer_token))

            doc_start.append(0)

            doc_end.append(len_docs_token)

        

        ###文本比较长，滑动切分

        else:

            begin=0

            end=Max_docs_len

            while end<=len_docs_token:

                doc_start.append(begin)

                doc_end.append(end-1)

                split_docs.append(' '.join(docs_token[begin:end]))

                begin=begin+Split_windows

                end=end+Split_windows

                

                assert len_query_token+len(split_docs[-1].split())<=509

            

            split_query=[' '.join(query_token)]*len(split_docs)

            

            split_answer=[' '.join(answer_token)]*len(split_docs)

        return split_query,split_docs,split_answer,[answer_start]*len(split_docs),[answer_end]*len(split_docs),doc_start,doc_end

    

    def prepare(self,text):

        text=text.replace(' ','')

        return text
split_train_list=[[],[],[],[],[],[],[],[],[],[]]

for index,row in train.iterrows():

    split_doc=Split_doc(row)

    #split_query,split_docs,split_answer,[start]*len(split_docs),[end]*len(split_docs),doc_start,doc_end

    for i,retur in enumerate(split_doc.split()):

        split_train_list[i].extend(retur)

    ##docid id label

    split_train_list[i+1].extend([row.docid]*len(retur))

    split_train_list[i+2].extend([row.id]*len(retur))

    split_train_list[i+3].extend([row.label]*len(retur))

    if index%1000==0:

        print(index)
new_train_dict={str(i):split_train_list[i] for i  in range(len(split_train_list))}

new_train=pd.DataFrame(new_train_dict)

new_train.columns=['question','split_docs','answer','answer_start','answer_end','doc_start','doc_end','docid','id','label']

new_train.to_csv('train_recall_split_4013windows.csv',index=False)
new_train.shape
# for i in range(len(split_train_list[0])):

#     for j in range(len(split_train_list)):

#         print(split_train_list[j][i])

#         if i==10:

#             break
!ls ../input/
import pandas as pd

A_test_recall_5 = pd.read_csv("../input/qa-4013-test5/A_test_recall.csv")

test = pd.read_csv(data_path+"NCPPolicies_test.csv",sep='\t')

NCPPolicies_context = pd.read_csv(data_path+"NCPPolicies_context.csv")

print(A_test_recall_5.shape)

test=pd.merge(A_test_recall_5,test,on='id',how='left')

test=test.merge(NCPPolicies_context,on='docid',how='left')

test['answer']='none'

test['label']=0

test.head()
split_test_list=[[],[],[],[],[],[],[],[],[],[],[]]

for index,row in test.iterrows():

    split_doc=Split_doc(row)

    #split_query,split_docs,split_answer,[start]*len(split_docs),[end]*len(split_docs),doc_start,doc_end

    for i,retur in enumerate(split_doc.split()):

        split_test_list[i].extend(retur)

    ##docid id label text

    split_test_list[i+1].extend([row.docid]*len(retur))

    split_test_list[i+2].extend([row.id]*len(retur))

    split_test_list[i+3].extend([row.label]*len(retur))

    split_test_list[i+4].extend([row.text]*len(retur))

    if index%1000==0:

        print(index)
new_test_dict={str(i):split_test_list[i] for i  in range(len(split_test_list))}

new_test=pd.DataFrame(new_test_dict)

new_test.columns=['question','split_docs','answer','answer_start','answer_end','doc_start','doc_end','docid','id','label','text']

new_test.to_csv('test5_recall_split_4013windows.csv',index=False)
# new_test
# import pandas as pd

# A_test_recall_10 = pd.read_csv("../input/A_test_recall_10.csv")
# import pandas as pd

# A_train_recall = pd.read_csv("../input/A_train_recall.csv")
# import pandas as pd

# A_train_recall = pd.read_csv("../input/qa-4013/A_train_recall.csv")

# NCPPolicies_context = pd.read_csv("../input/qa-4013/NCPPolicies_context.csv")

# NCPPolicies_test = pd.read_csv("../input/qa-4013/NCPPolicies_test.csv")

# NCPPolicies_train_20200301 = pd.read_csv("../input/qa-4013/NCPPolicies_train_20200301.csv")
import pandas as pd

A_test_recall = pd.read_csv("../input/A_test_recall.csv")