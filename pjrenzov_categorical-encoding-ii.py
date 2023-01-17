# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import category_encoders as ce   

from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

test =  pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')
train = train.drop({'id'},axis =1)
for i, key in enumerate(train.columns):

    print(key , i , len(train[key].unique()))

    print(train[key].unique())
train = train.drop({'nom_5','nom_6', 'nom_7', 'nom_8' , 'nom_9'},axis=1)
train.info()
for i, key in enumerate(train.columns):

    print(key , i , len(train[key].unique()))

    print(train[key].unique())
train_d = train

train_d = train.sort_values(by=['target'])
train_d = train_d.reset_index().drop({'index'},axis =1)
for i in range(len(train_d['target'])):

    if train_d.iloc[i,-1] == 1:

        print(i)

        break
train_d.iloc[487678,-1]
train_d_1 = train_d.iloc[487677:,:]

train_d_0 = train_d.iloc[:487677,:]
replaces_0=[]

replaces_1=[]

for col in train_d_1.columns:

    replaces_1.append(train_d_1[col].value_counts().idxmax())

for col in train_d_0.columns:

    replaces_0.append(train_d_0[col].value_counts().idxmax())
for col in range(len(train_d_1.columns)):

    train_d_1.iloc[:,col] = train_d_1.iloc[:,col].fillna(replaces_1[col])

for col in range(len(train_d_0.columns)):

    train_d_0.iloc[:,col] = train_d_0.iloc[:,col].fillna(replaces_0[col])
train_d = pd.concat([train_d_0,train_d_1])
train_d.info()
for i, key in enumerate(train_d.columns):

    print(key , i , len(train_d[key].unique()))

    print(train_d[key].unique())
train_f = train_d
concat_data = np.zeros(shape = (190,19))

concat_data = pd.DataFrame(concat_data, columns= train_f.columns)
ord_0 = [ 'Novice','Contributor', 'Expert','Master', 'Grandmaster' ]

ord_1 = [ 'Freezing','Cold','Warm','Hot', 'Boiling Hot','Lava Hot'  ]

concat_data.iloc[:5,11] = ord_0

concat_data.iloc[:6,12] = ord_1

concat_data.iloc[:15,13]  = sorted(train_f['ord_3'].unique(), reverse =True)

concat_data.iloc[:26,14]  = sorted(train_f['ord_4'].unique(), reverse =True)

concat_data.iloc[:190,15]  = sorted(train_f['ord_5'].unique(), reverse =True)
train_f = pd.concat([concat_data,train_f])
ce_ord = ce.OrdinalEncoder(cols = ['ord_1','ord_2','ord_3','ord_4','ord_5'])

train_f = ce_ord.fit_transform(train_f)

ce_one_hot = ce.OneHotEncoder(cols = ['bin_3','bin_4','nom_0','nom_1','nom_2','nom_3','nom_4'])

train_f = ce_one_hot.fit_transform(train_f)
train_f = train_f[190:]
test = test.drop({'id','nom_5','nom_6', 'nom_7', 'nom_8' , 'nom_9'},axis=1)
test.info()
concat_data = concat_data.drop({'target'},axis=1)
for col in test.columns:

    test[col]=test[col].fillna(method='ffill')
test = pd.concat([concat_data,test],)
ce_ord = ce.OrdinalEncoder(cols = ['ord_1','ord_2','ord_3','ord_4','ord_5'])

test = ce_ord.fit_transform(test)

ce_one_hot = ce.OneHotEncoder(cols = ['bin_3','bin_4','nom_0','nom_1','nom_2','nom_3','nom_4'])

test = ce_one_hot.fit_transform(test)

test = test[190:]
target = train_f['target']

train_f = train_f.drop({'target'},axis =1)
from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()

lg.fit(train_f,target)

y_pred = lg.predict(test)
submission = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')
submission['target'] = y_pred
submission.to_csv('submission.csv',index=False)