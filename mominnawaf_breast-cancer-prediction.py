# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

import time

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df.info()
df.isna().sum()
df=df.drop(['Unnamed: 32'],axis=1)
df.info()
df.head()
df['diagnosis'] = df['diagnosis'].apply(lambda x:'1' if x=='M' else '0')
df.head()
df.plot(kind='density', subplots=True, layout=(10,4), sharex=False, legend=False, fontsize=1)

plt.show()
import seaborn as sns

corrmat=df.corr()

corrmat
top_cor_feature=corrmat.index
g=sns.heatmap(df[top_cor_feature].corr(),annot=True,cmap='RdYlGn')
x=df.drop(['diagnosis'],axis=1)

y=df['diagnosis']
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=21)
models_list = []

models_list.append(('CART', DecisionTreeClassifier()))

models_list.append(('SVM', SVC())) 

models_list.append(('NB', GaussianNB()))

models_list.append(('KNN', KNeighborsClassifier()))
num_fold = 10

result = []

names = []

for name , model in models_list:

    kfold = KFold(n_splits=num_fold,random_state=123)

    start = time.time()

    cv_result = cross_val_score(model,xtrain,ytrain,cv=kfold,scoring='accuracy')

    end= time.time()

    result.append(cv_result)

    print('%s:%f runtime:(%f)'% (name,cv_result.mean(),end-start))