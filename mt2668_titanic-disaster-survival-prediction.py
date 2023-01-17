# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

submission=pd.read_csv('../input/gender_submission.csv')
train.head()
df_test=test.merge(submission,on='PassengerId')
df_test.head()
df=pd.concat([train,df_test],sort=True)
df.head()
df.drop(['Cabin','Name','Ticket'],axis=1,inplace=True)
def col_to_numeric(col):

    arr=col.unique()

    np_col=col.values

    di={}

    j=0

    if col.dtype=='O' or col.dtype=='str':

        for i in arr:

            if i not in di:

                di[i]=j

                j+=1

        for i in range(len(np_col)):

            np_col[i]=di[np_col[i]]

        df_col=pd.DataFrame(np_col)

        return df_col

    else:

        return pd.DataFrame(np_col)



def data_to_numeric(df):

    for i in list(df):

        df[i]=col_to_numeric(df[i])

    return df
df=data_to_numeric(df)
df.set_index('PassengerId',inplace=True)

df.head()
df=df.interpolate()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



X=df.drop(['Survived'],axis=1)

y=df['Survived']



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)



rfc=RandomForestClassifier(n_estimators=100)

rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)



from sklearn.metrics import accuracy_score



acc=accuracy_score(y_pred,y_test)

print('Accuracy during training : ',acc)
test=data_to_numeric(test)
test.set_index('PassengerId',inplace=True)

test=test.drop(['Name','Ticket','Cabin'],axis=1)

test.head()
test = test.sort_index(axis=1)

test.head()
test=test.interpolate()
y_main_test=submission['Survived']
y_main_pred=rfc.predict(test)

acc_main=accuracy_score(y_main_pred,y_main_test)

print(acc_main)
submission['Survived']=pd.DataFrame({'Survived':y_main_pred})

submission.head()
submission.to_csv('My_Submission.csv',index=False)