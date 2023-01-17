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
import pandas as pd

data=pd.read_csv('../input/cat-in-the-dat/train.csv')
data.head()
data.columns
data.dtypes
data=data.astype({'bin_0':'object','bin_1':'object','bin_2':'object'})
data['bin_3']=data['bin_3'].astype('category')

data['b3_']=data['bin_3'].cat.codes

data=data.drop('bin_3',axis=1)

data.head()
data['bin_4']=data['bin_4'].astype('category')

data['b4_']=data['bin_4'].cat.codes

data=data.drop('bin_4',axis=1)

data.head()
data['nom_0'].value_counts()
data['nom_1'].value_counts()
data['nom_2'].value_counts()
data['nom_3'].value_counts()
data['nom_4'].value_counts()
data['nom_5'].value_counts()
data=pd.get_dummies(data,columns=['nom_0','nom_1','nom_2','nom_3','nom_4'])

data.head()
temp=data['nom_5'].value_counts()

temp.head()
data['5nom']=data['nom_5'].apply(lambda x: temp[x])
data=data.drop('nom_5',axis=1)

data.head()
data['nom_6'].value_counts()
temp=data['nom_6'].value_counts()

data['nom6']=data['nom_6'].apply(lambda x: temp[x])
data=data.drop('nom_6',axis=1)

data.head()
data['nom_7'].value_counts()
temp=data['nom_7'].value_counts()

data['nom7']=data['nom_7'].apply(lambda x: temp[x])
data=data.drop('nom_7',axis=1)

data.head()
data['nom_8'].value_counts()
temp=data['nom_8'].value_counts()

data['nom8']=data['nom_8'].apply(lambda x: temp[x])
data=data.drop('nom_8',axis=1)

data.head()
data['nom_9'].value_counts()
temp=data['nom_9'].value_counts()

data['nom9']=data['nom_9'].apply(lambda x: temp[x])
data=data.drop('nom_9',axis=1)

data.head()
data['ord_1'].value_counts()
data['ord_0'].value_counts()
data['ord_2'].value_counts()
data['ord_3'].value_counts()
data['ord_4'].value_counts()
data['ord_5'].value_counts()
temp=data['ord_5'].value_counts()

data['ord5']=data['ord_5'].apply(lambda x: temp[x])
data=data.drop('ord_5',axis=1)

data.head()
data=pd.get_dummies(data,columns=['ord_0','ord_1','ord_2','ord_3','ord_4'])

data.head()
data.columns
data=data.astype({'day':'object','month':'object'})
data['day'].value_counts()
data['month'].value_counts()
data=pd.get_dummies(data,columns=['day','month'])

data.head()
y=data['target']

x=data.drop('target',axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,stratify=y,test_size=0.2)
from sklearn.linear_model import LogisticRegression

log=LogisticRegression()

log.fit(x_train,y_train)

log.predict(x_train)
log.score(x_train,y_train)
prob=log.predict_proba(x_train)

prob[:10]
prob[:10,0]
ans1=log.predict(x_test)

ans1
log.score(x_test,y_test)
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
accuracy_score(ans1,y_test)
confusion_matrix(y_test,ans1)
datay=pd.read_csv('../input/cat-in-the-dat/test.csv')

datay.head()
datay.shape
datay=datay.astype({'bin_0':'object','bin_1':'object','bin_2':'object'})

datay['bin_3']=datay['bin_3'].astype('category')

datay['b3_']=datay['bin_3'].cat.codes

datay=datay.drop('bin_3',axis=1)

datay['bin_4']=datay['bin_4'].astype('category')

datay['b4_']=datay['bin_4'].cat.codes

datay=datay.drop('bin_4',axis=1)

datay=datay.astype({'day':'object','month':'object'})

datay=pd.get_dummies(datay,columns=['nom_0','nom_1','nom_2','nom_3','nom_4','ord_0','ord_1','ord_2','ord_3','ord_4','day','month'])

datay.head()
temp=datay['nom_5'].value_counts()

datay['nom5']=datay['nom_5'].apply(lambda x: temp[x])

datay.drop('nom_5',axis=1)

datay.head()
datay=datay.drop('nom_5',axis=1)

datay.head()
temp=datay['nom_6'].value_counts()

datay['nom6']=datay['nom_6'].apply(lambda x: temp[x])

datay=datay.drop('nom_6',axis=1)

datay.head()
temp=datay['nom_7'].value_counts()

datay['nom7']=datay['nom_7'].apply(lambda x: temp[x])

datay=datay.drop('nom_7',axis=1)

datay.head()
temp=datay['nom_8'].value_counts()

datay['nom8']=datay['nom_8'].apply(lambda x: temp[x])

datay=datay.drop('nom_8',axis=1)

datay.head()
temp=datay['nom_9'].value_counts()

datay['nom9']=datay['nom_9'].apply(lambda x: temp[x])

datay=datay.drop('nom_9',axis=1)

datay.head()
temp=datay['ord_5'].value_counts()

datay['ord5']=datay['ord_5'].apply(lambda x: temp[x])

datay=datay.drop('ord_5',axis=1)

datay.head()
predictions=log.predict_proba(datay)
predictions[:10]
predictions=predictions[:,0]

predictions
Submission=pd.DataFrame( {'id' : datay['id'] , 'target' : predictions} )

Submission.to_csv('Submission.csv',index=False)