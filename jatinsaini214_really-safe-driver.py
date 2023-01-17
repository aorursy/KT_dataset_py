# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

from sklearn import preprocessing as ppr

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

submission = pd.read_csv("../input/sample_submission.csv")

print("Train shape : ", train_df.shape)

print("Test shape : ", test_df.shape)
dtype_df = train_df.dtypes.reset_index()

#print (dtype_df)

dtype_df.columns = ["Count", "Column Type"]

dtype_df.groupby("Column Type").aggregate('count').reset_index()
missing_df = train_df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.loc[missing_df['missing_count']>0]

missing_df = missing_df.sort_values(by='missing_count')

missing_df
list(submission)

#y=train_df['ps_calc_04']

#print (y)#+train_df['ps_calc_06']
from scipy.stats.stats import pearsonr

train_y = train_df['target']

train_x = train_df.drop(["id","target"],axis=1)

i=0

l=[]

f=[]

#k=list(list(train_x))

for y in list(train_x):

    #print (train_x[y])

    z= pearsonr(train_x[y],train_y)[1]

    if z>0.4:

        l.append(y)

        f.append(z)

        #print (k[i])

        i=i+1

#for a in range(len(l)):

#    print ((l[a]),(f[a]*100))

x=pd.DataFrame(f,l)

print (len(x))    
used=set()

unique = [x for x in train_df['ps_calc_04'] if x not in used and (used.add(x) or True)]

print ((unique))

#print (type(train_df['ps_calc_17_bin']))
# split data into train and test sets

X=pd.DataFrame(train_x,columns=l)

Y=train_y

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=3)

#[float(i) for i in x_train]

#list(map(int,x_train))

#[float(j) for j in x_test]

#[float(k) for k in y_train]

#[float(l) for l in y_test]
xgb = XGBClassifier()

xgb.fit(x_train, y_train)

y_pred = xgb.predict_proba(x_test)[:,1]

# make predictions for test data

#predictions = [value for value in y_pred]

print ((y_pred))

# evaluate predictions

test_x = pd.DataFrame(test_df,columns=l)

test_y = xgb.predict_proba(test_x)[:,1]

test_x=pd.DataFrame(test_df,columns=['id'])

test_x = test_x.reset_index()

#test_x['key'] = test_x.index

test_y=pd.DataFrame(test_y)

test_y = test_y.reset_index()

predicted=test_x.merge(test_y,on='index')

#m=pd.DataFrame(test_y)

#for i in range(y_pred):

 #   if (predictions>0):

  #      print (test_x(i,"ps_calc_04"))

#print (test_y.size)

#k=pd.DataFrame(test_df,columns=['id'])

#predict=pd.concat([k,test_y],axis=1)

predicted=predicted.drop(['index'],axis=1)

predicted['target']=predicted[0]

predicted=pd.DataFrame(predicted,columns=['id','target'])

#submission=submission.merge(test_y,on='id')

print ((predicted))
predicted.to_csv('predicted.csv',header=True,index=False)