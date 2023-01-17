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
train=pd.read_csv('/kaggle/input/facebook-recruiting-iii-keyword-extraction/Train.zip')
train.shape
train.head()
train.shape
train1=train.iloc[1:50000,:]
del train
train1.columns
train1.isnull().sum()
y=train1.Tags
x=train1.drop('Tags',axis=1)
x.head()
del train1
import re
def check(x):
    s=re.sub('<[^>]*>','',x)
    s=re.sub('[^\w\s]','',s)
    return s
x1=x.Body.apply(check)
x.Body=x1
del x1
y.head()
from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(x,y,test_size=.20,random_state=20)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
ct=CountVectorizer()
tf=TfidfVectorizer()
ct.fit_transform(xtr,ytr)
import pickle