# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd
tr=pd.read_csv('../input/train (1).csv')
## first 5 rows of training data

tr.head()
## Number of rows and columns of training data

tr.shape

## column names

tr.columns
## Number of distinct observations

tr.nunique()
## find missing values in each column

tr.isna().sum()
## Basic statistical details of data

tr.describe()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
## fit_transform function is one the feature extraction for text analysis.

tr['ID_status'] = le.fit_transform(tr['ID_status'])

tr['active'] = le.fit_transform(tr['active'])

tr['count_reassign'] = le.fit_transform(tr['count_reassign'])

tr['count_opening'] = le.fit_transform(tr['count_opening'])



tr['type_contact'] = le.fit_transform(tr['type_contact'])

tr['category_ID'] = le.fit_transform(tr['category_ID'])

tr['Doc_knowledge'] = le.fit_transform(tr['Doc_knowledge'])

tr['notify'] = le.fit_transform(tr['notify'])



tr['confirmation_check'] = le.fit_transform(tr['confirmation_check'])

tr['impact'] = le.fit_transform(tr['impact'])

tr['ID'] = le.fit_transform(tr['ID'])

tr['count_updated'] = le.fit_transform(tr['count_updated'])



tr['ID_caller'] = le.fit_transform(tr['ID_caller'])

tr['opened_by'] = le.fit_transform(tr['opened_by'])

tr['updated_by'] = le.fit_transform(tr['updated_by'])

tr['opened_time'] = le.fit_transform(tr['opened_time'])



tr['Created_by'] = le.fit_transform(tr['Created_by'])

tr['created_at'] = le.fit_transform(tr['created_at'])

tr['updated_at'] = le.fit_transform(tr['updated_at'])

tr['opened_time'] = le.fit_transform(tr['opened_time'])



tr['location'] = le.fit_transform(tr['location'])

tr['user_symptom'] = le.fit_transform(tr['user_symptom'])

tr['Support_group'] = le.fit_transform(tr['Support_group'])

tr['support_incharge'] = le.fit_transform(tr['support_incharge'])



tr['problem_ID'] = le.fit_transform(tr['problem_ID'])

tr['change_request'] = le.fit_transform(tr['change_request'])



## first 5 rows of training data after preprocessing

tr.head()
col =['ID','ID_status','active','count_reassign','count_opening','count_updated','ID_caller','opened_by',

   'created_at','opened_time','Created_by','updated_by','updated_at','type_contact','location','category_ID',

     'user_symptom','Support_group','support_incharge','Doc_knowledge','confirmation_check','impact','notify','problem_ID','change_request']
tr = tr[col]

tr.head()
## Number of rows and columns of training data

tr.shape
## impact is to predict

y = tr['impact']

y
## drop the columns impact and Created_by from training data

## here we drop impact, because it is taken as predict

## we drop Creared_by, because incident is not created by customer

tr.drop(['impact','Created_by'],axis=1,inplace=False)
from sklearn import linear_model

import numpy as np

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(tr,y,test_size=0.1,random_state=420)
model = linear_model.LogisticRegression()

mod = model.fit(xtrain,ytrain)

preds = mod.predict(xtest)
preds
from sklearn.metrics import accuracy_score

accuracy_score(ytest,preds)