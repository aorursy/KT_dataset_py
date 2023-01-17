import pandas as pd

import numpy as np

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
varlist=['Male','Married','BankCustomer','EducationLevel','Ethnicity','Citizen','PriorDefault','Employed','DriversLicense','Approved']





def binary_map(x):

    return x.map({'t': 1,'a':1 ,'b':0,'f': 0,'u':0,'y':1,'l':2,'t':3,'g':0,'p':1,'gg':2,'c':0,'d':1,'cc':2,'i':3,'j':4,'k':5,'m':6,'r':7,'q':8,'w':9,'x':10,'e':11,'aa':12,'ff':13,'v':0,'h':1,'bb':2,'n':3,'z':5,'dd':6,'ff':7,'o':8,'s':3,'+':1,'-':0})





train[varlist] = train[varlist].apply(binary_map)

train.isnull().info()
train.isnull().sum()
train=train.dropna()
train.isnull().sum()
train.info()
train[train['Age']=='?'].count()
#age has only ? so we can replace it with mean
train['Age']=train['Age'].replace('?',0)
train[train['Age']=='?'].count()
train['Age']=train['Age'].astype(float)
c=train['Age'].mean()
train['Age']=train['Age'].replace(0,c)
train.info()
X_train=train.drop(['Key','Approved','ZipCode'],1)
X_train.head(10)
y_train=train['Approved']

y_train[y_train=='-']=0

y_train[y_train=='+']=1

y_train=y_train.astype(int)
y_train.unique()
test.head()
varlist=['Male','Married','BankCustomer','EducationLevel','Ethnicity','Citizen','PriorDefault','Employed','DriversLicense']



def binary_map(x):

    return x.map({'t': 1,'a':1 ,'b':0,'f': 0,'u':0,'y':1,'l':2,'t':3,'g':0,'p':1,'gg':2,'c':0,'d':1,'cc':2,'i':3,'j':4,'k':5,'m':6,'r':7,'q':8,'w':9,'x':10,'e':11,'aa':12,'ff':13,'v':0,'h':1,'bb':2,'n':3,'z':5,'dd':6,'ff':7,'o':8,'s':3,'+':1,'-':0})





test[varlist] = test[varlist].apply(binary_map)

test.isnull().sum()
test=test.fillna(0)
for i in test.columns:

    print(test[i].unique())
X_test=test.drop(['Key','ZipCode'],1)
from sklearn.linear_model import LogisticRegression

rog=LogisticRegression()
rog.fit(X_train,y_train)
y_pred=rog.predict(X_test)
y_pred
y_pred=y_pred.astype(str)

y_pred[y_pred=='1']='+'

y_pred[y_pred=='0']='-'

y_pred
key=test['Key']

key
c=pd.Series(y_pred)

d=pd.Series(key)

# c[c==0]='+'
c[c==0]='-'

c[c==1]='+'

c
prediction = pd.DataFrame(c, columns=['predictions']).to_csv('prediction.csv')