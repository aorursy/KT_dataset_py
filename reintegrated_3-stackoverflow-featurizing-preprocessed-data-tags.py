# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pickle

with open('../input/stackoverflow-2/data1.txt','rb') as fh:

    s=pickle.load(fh)

data=pd.read_csv("../input/stackoverflow-tag-prediction/data.csv")

data.drop(columns=['Unnamed: 0','Id'],axis=1,inplace=True)

data['sentences']=s

data.drop(columns=['Title','Body'],inplace=True)

data.head(5)
from datetime import datetime

start=datetime.now()

from sklearn.feature_extraction.text import TfidfVectorizer

vec=TfidfVectorizer(min_df=0.00009, max_features=200000, smooth_idf=True, norm="l1", \

                             tokenizer = lambda x: x.split(), sublinear_tf=False, ngram_range=(1,2))

x_train_vec=vec.fit_transform(data.iloc[:int(data.shape[0]*0.8),1])

x_test_vec=vec.transform(data.iloc[int(data.shape[0]*0.8):,1])

# with open('train_unigram.npy','wb') as f:

#     np.save(f,x_train_vec)

# with open('test_unigram.npy','wb') as f:

#     np.save(f,x_test_vec)

datetime.now()-start
from scipy import sparse

sparse.save_npz("train_bi_gram.npz",x_train_vec)

sparse.save_npz("test_ni_gram.npz",x_test_vec)
from sklearn.feature_extraction.text import CountVectorizer

vec=CountVectorizer(tokenizer=lambda x: x.split(),binary=True)

tags_vec=vec.fit_transform(data.Tags)
type(tags_vec)
def choose_tags(n):

    t=np.array(tags_vec.sum(axis=0).A1)

    tag_index=t.argsort()[::-1]

    return tags_vec[:,tag_index[:n]]

def questions_explained(n):

    tags_choosen=choose_tags(n)

    x=tags_choosen.sum(axis=1)

    return np.count_nonzero(x==0)
total_tags=tags_vec.shape[1]

total_qs=tags_vec.shape[0]

qs_explained=[]

for no_tags in range(500,total_tags,100):

    qs_explained.append(np.round(((total_qs-questions_explained(no_tags))/total_qs)*100,3))
import matplotlib.pyplot as plt

labels=list(range(500,total_tags,100))

plt.plot(labels,qs_explained,'r--')

plt.grid()

plt.show()

print("percentages of questions explained by ",labels[45],"Tags is",qs_explained[45])
print('from total no of tags =',tags_vec.shape[1],'we have used',labels[45]*100/tags_vec.shape[1],"% of total tags =",labels[45])

trunc_tags=choose_tags(5000)
train_size=int(tags_vec.shape[0]*0.8)

y_train=trunc_tags[:train_size,:]

y_test=trunc_tags[train_size:,:]

print(y_train.shape,"train_shape")

print(y_test.shape,"test_shape")
from scipy import sparse

sparse.save_npz('tags_vec_train.npz',y_train)

sparse.save_npz('tags_vec_test.npz',y_test)