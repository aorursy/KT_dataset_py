import os

from argparse import Namespace

import json

import re

import string

from collections import Counter



import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import Dataset,DataLoader



from sklearn.metrics import accuracy_score

from tqdm import tqdm_notebook
os.listdir("../input/")
df=pd.read_csv("../input/news_with_splits.csv")

df.head()
class Vocabulary(object):

    def __init__(self,token_to_idx=None):

        if token_to_idx is None:

            self._token_to_idx={}

        else:

            self._token_to_idx=token_to_idx

        self._idx_to_token={idx:token for token,tdx in self._token_to_idx.items()}

        

    def add_token(self,token):

        if token in self._token_to_idx:

            index=self._token_to_idx[token]

        else:

            index=len(self._token_to_idx)

            self._token_to_idx[token]=index

            self._idx_to_token[index]=token

        return index 

    

    def look_token(self,token):

        return self.add_token(token)

        

    def look_index(self,index):

        if index not in self._idx_to_token:

            raise KeyError("the index (%d) is not in the Vocabulary" % index)

        return self._idx_to_token[index]  
class SequenceVocabulary(Vocabulary):

    def __init__(self,token_to_idx=None,unk_token="<UNK>",

                mask_token="<MASK>",begin_seq_token="<BEGIN>",end_seq_token="<END>"):

        super(SequenceVocabulary,self).__init__()

        self._mask_token=mask_token

        self._unk_token=unk_token

        self._begin_seq_token=begin_seq_token

        self._end_seq_token=end_seq_token

        

        self._mask_index=self.add_token(self._mask_token)

        self._unk_index=self.add_token(self._unk_token)

        self._begin_seq_index=self.add_token(self._begin_seq_token)

        self._end_seq_index=self.add_token(self._end_seq_token)

        

    def look_token(self,token):

        if self._unk_index>=0:

            return self._token_to_idx.get(token,self._unk_index)

        else:

            return self.add_token(token)
class Vectorizer(object):

    def __init__(self,title_vocab,category_vocab):

        self.title_vocab=title_vocab

        self.category_vocab=category_vocab

        

    def vectorize(self,title,max_len=-1):

        indices=[self.title_vocab._begin_seq_index]

        indices.extend([self.title_vocab.look_token(token) for token in title.split(" ")])

        indices.append(self.title_vocab._end_seq_index)

        

        if max_len<0:

            max_len=len(indices)

        vector=np.zeros(max_len,dtype=np.int64)

        vector[:len(indices)]=indices

        vector[len(indices):]=self.title_vocab._mask_index

        

        return vector

        

    @classmethod

    def transform_df(cls,news_df,cutoff=25):

        category_vocab=Vocabulary()

        for category in news_df.category:

            category_vocab.add_token(category)

            

        word_counts=Counter()

        for title in news_df.title:

            for token in title.split(" "):

                if token not in string.punctuation:

                    word_counts[token]+=1

        

        title_vocab=SequenceVocabulary()

        for word,word_count in word_counts.items():

            if word_count>cutoff:

                title_vocab.add_token(word)

        

        return cls(title_vocab,category_vocab)
class NewsDataset(Dataset):

    def __init__(self,news_df,vectorizer,colname):

        self.news_df=news_df

        self._vectorizer=vectorizer

        

        self.max_len=max(self.news_df.title.apply(lambda x:len(x.split(" "))))+2

        

        self.train_df=self.news_df[self.news_df.split=="train"]

        self.train_size=len(self.train_df)

        self.val_df=self.news_df[self.news_df.split=="val"]

        self.val_size=len(self.val_df)

        self.test_df=self.news_df[self.news_df.split=="test"]

        self.test_size=len(self.test_df)

                 

        self.look_dict={

            "train":(self.train_df,self.train_size),

            "val":(self.val_df,self.val_size),

            "test":(self.test_df,self.test_size)

        }

        

        # class weight 

        class_counts=self.news_df.category.value_counts().to_dict()

        def sort_key(item):

            return self._vectorizer.category_vocab.look_token(item[0])

        sorted_counts=sorted(class_counts.items(),key=sort_key)

        frequenies=[count for _,count in sorted_counts]

        self.class_weights=1.0/torch.tensor(frequenies,dtype=torch.float32)

        

        self.set_split(colname)

        

    def set_split(self,colname):

        self._target_df,self._target_size=self.look_dict[colname]

        

    def __len__(self):

        return self._target_size

    

    def __getitem__(self,index):

        row=self._target_df.iloc[index]

        title_vector=self._vectorizer.vectorize(row.title,self.max_len)

        category_index=self._vectorizer.category_vocab.look_token(row.category)

        return title_vector,category_index
train_dataset=NewsDataset(df,Vectorizer.transform_df(df),"train")

val_dataset=NewsDataset(df,Vectorizer.transform_df(df),"val")

test_dataset=NewsDataset(df,Vectorizer.transform_df(df),"test")
train_dataset.__getitem__(0)
args=Namespace(

    use_glove=True,

    embedding_size=100,

    hidden_dim=100,

    nun_channels=100,

    learning_rate=0.001,

    dropout_p=0.1,

    n_epochs=100,

    batch_size=128,

    num_classes=4

)
train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)

val_loader=DataLoader(val_dataset,batch_size=args.batch_size,shuffle=True)

test_loader=DataLoader(test_dataset,batch_size=1)
class NewsClassifier(nn.Module):

    def __init__(self,vocabulary_num,embedding_num,num_channels,dropout_p,

                 hidden_dim,num_classes,pretained_embedding=None,padding_idx=0):

        super(NewsClassifier,self).__init__()

        

        if pretained_embedding is None:

            self.embedding=nn.Embedding(vocabulary_num,embedding_num,padding_idx)

        else:

            pretained_embedding=torch.from_numpy(pretained_embedding).float()

            self.embedding=nn.Embedding(vocabulary_num,embedding_num,padding_idx,_weight=pretained_embedding)

            

        self.connet=nn.Sequential(

            nn.Conv1d(in_channels=embedding_num,out_channels=num_channels,kernel_size=3),

            nn.ELU(),

            nn.Conv1d(in_channels=num_channels,out_channels=num_channels,kernel_size=3,stride=2),

            nn.ELU(),

            nn.Conv1d(in_channels=num_channels,out_channels=num_channels,kernel_size=3,stride=2),

            nn.ELU(),

            nn.Conv1d(in_channels=num_channels,out_channels=num_channels,kernel_size=3),

            nn.ELU()

        )

        

        self._dropout_p=dropout_p

        self.fc1=nn.Linear(num_channels,hidden_dim)

        self.fc2=nn.Linear(hidden_dim,num_classes)

        

    def forward(self,x_in,apply_softmax=False):

        x=self.embedding(x_in).permute(0,2,1)

        x=self.connet(x)

        remaing_size=x.size(dim=2)

        x=F.avg_pool1d(x,remaing_size).squeeze(dim=2)  # 对最小一维取平均

        x=F.dropout(x,p=self._dropout_p)

        

        x=self.fc1(x)

        x=F.dropout(x,p=self._dropout_p)

        x=F.relu(x)

        output=self.fc2(x)

        

        if apply_softmax:

            output=F.softmax(output)

        return output
with open("../input/glove.6B.100d.txt","r",encoding="utf-8") as f:

    word_to_index={}

    embedding_glove=[]

    for index,line in enumerate(f):

        word_to_index[line.split(" ")[0]]=index

        embedding_i=[float(x) for x in line.split(" ")[1:]]

        embedding_glove.append(embedding_i)
vec=Vectorizer.transform_df(df)
n=len(vec.title_vocab._token_to_idx)

final_embedding=np.ones((n,100))

for word,i in vec.title_vocab._token_to_idx.items():

    word=word.lower()

    if word in word_to_index:

        idx=word_to_index[word]

        final_embedding[i,:]=embedding_glove[idx]

    else:

        embedding_ii=torch.ones(1,100)

        torch.nn.init.xavier_uniform_(embedding_ii)

        final_embedding[i,:]=embedding_ii
classifier=NewsClassifier(vocabulary_num=n,embedding_num=args.embedding_size,num_channels=args.nun_channels,dropout_p=0.1,

                          hidden_dim=args.hidden_dim,num_classes=args.num_classes,pretained_embedding=final_embedding)

print(classifier)
loss_func=nn.CrossEntropyLoss()

optimizer=optim.Adam(params=classifier.parameters(),lr=args.learning_rate)
train_state={"train_loss":[],"val_loss":[]}

for epoch in range(args.n_epochs):

    running_train_loss=0.0

    classifier.train()

    

    for batch_index,batch_data in enumerate(train_loader):

        x,y=batch_data

        classifier.zero_grad()

        

        y_pred=classifier(x)

        loss=loss_func(y_pred,y)

        loss_train_batch=loss.item()

        running_train_loss+=(loss_train_batch-running_train_loss)/(batch_index+1)

        loss.backward()

        optimizer.step()

    train_state["train_loss"].append(running_train_loss)

    

    running_val_loss=0.0

    classifier.eval()

    for batch_index,batch_data in enumerate(val_loader):

        x,y=batch_data

        y_pred=classifier(x)

        loss=loss_func(y_pred,y)

        loss_val_batch=loss.item()

        running_val_loss+=(loss_val_batch-running_val_loss)/(batch_index+1)

    train_state["val_loss"].append(running_val_loss)

    print("Epoch: ",epoch,"Train Loss: ",running_train_loss,"Valid Loss",running_val_loss)
em=nn.Embedding(num_embeddings=n,embedding_dim=100,padding_idx=0,_weight=torch.from_numpy(final_embedding))
for x,y in train_loader:

    print(x)

    o=em(x)

    print(o.permute(0,2,1).size())

    break