import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train.head()
train.describe()
train.info()

#increase figure size
sns.set_style('whitegrid')
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)


sns.heatmap(train.isnull(),yticklabels=False,cbar=False)

fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
sns.heatmap(test.isnull(),yticklabels=False,cbar=False)
train.drop('Cabin',axis=1,inplace=True)#axis=1 drops a col. default drops rows.
test.drop('Cabin',axis=1,inplace=True)
#also drop text columns name and ticket
train.drop(['Name','Ticket'],axis=1,inplace=True)#axis=1 drops a col. default drops rows.
test.drop(['Name','Ticket'],axis=1,inplace=True)
#drop first to prevent perfectly correlated columns 

Embarked=pd.get_dummies(train['Embarked'],drop_first=True)
Sex=pd.get_dummies(train['Sex'],drop_first=True)
train=pd.concat([train,Embarked,Sex],axis=1)

Embarked=pd.get_dummies(test['Embarked'],drop_first=True)
Sex=pd.get_dummies(test['Sex'],drop_first=True)
test=pd.concat([test,Embarked,Sex],axis=1)
train.head()
train.drop(['Sex','Embarked'],axis=1,inplace=True)
test.drop(['Sex','Embarked'],axis=1,inplace=True)
cols=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Q', 'S', 'male']
trainAGE=pd.concat([test,train[cols]])
def fixnan(x):
    if np.isnan(x):
        return 14.45
    return x

trainAGE['Fare']=trainAGE['Fare'].apply(fixnan)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
X=trainAGE[['Pclass',  'SibSp', 'Parch', 'Fare', 'Q', 'S', 'male']]
y=trainAGE['Age']
index=[np.invert(trainAGE['Age'].isnull())][0]
lm.fit(X[index],y[index])
pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
import math
def applypredictions(age,cols):
    if math.isnan(age):
        z=lm.predict(cols.values.reshape(1, -1))
        if z > 0:
            return z
        return 1.0
    return age
pred_cols=['Pclass',  'SibSp', 'Parch', 'Fare', 'Q', 'S', 'male']
train['Age']=train.apply(lambda row: applypredictions(row['Age'], row[pred_cols]), axis=1)
test['Age']=test.apply(lambda row: applypredictions(row['Age'], row[pred_cols]), axis=1)
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
#remember to fix the nan fare in the actual test data.....zzzz
def fixnan(x):
    if np.isnan(x):
        return 14.45
    return x

test['Fare']=test['Fare'].apply(fixnan)

fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
sns.heatmap(test.isnull(),yticklabels=False,cbar=False)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
train.head()
test.head()
