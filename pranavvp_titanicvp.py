import pandas as pd

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

tr=pd.read_csv('../input/titanic-me/train.csv')

te=pd.read_csv('../input/titanic-me/test.csv')
tr.loc[tr['Sex']=='male']=0

tr.loc[tr['Sex']=='female']=1

te.loc[te['Sex']=='male']=0

te.loc[te['Sex']=='female']=1
y=tr[['Survived']]

X=tr[['Age','Sex']]
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.33,random_state=5)
lr=LinearRegression()

lr.fit(Xtrain,ytrain) 
f=te[['Age','Sex']]

r=lr.predict(f[:500])

print (r)
file=open('ot.txt','w')

file.write(str(r))

file.close()

df=pd.read_fwf('ot.txt')

df.to_csv('out.csv')