import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/train.csv',encoding = "ISO-8859-1",low_memory=False)
df.head()
df.columns.values
# Did the gender affected survival chances? 

sns.countplot('Sex',hue='Survived',data=df)

plt.show()
# How about the Pclass

pd.crosstab(df.Pclass,df.Survived,margins=True).style.background_gradient(cmap='summer_r')
sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
# Cleaning data

df.Age.isnull().sum()
mean_age = df.Age.mean()

mean_age
df.loc[df.Age.isnull(),'Age'] = mean_age

df.Age.isnull().sum()
# Age as a Categorical feature

df['Age_band']=0

df.loc[df['Age']<=16,'Age_band']=0

df.loc[(df['Age']>16)&(df['Age']<=32),'Age_band']=1

df.loc[(df['Age']>32)&(df['Age']<=48),'Age_band']=2

df.loc[(df['Age']>48)&(df['Age']<=64),'Age_band']=3

df.loc[df['Age']>64,'Age_band']=4

df.head(2)
#checking the number of passenegers in each band

df['Age_band'].value_counts().to_frame().style.background_gradient(cmap='summer')
# Converting String Values into Numeric

df['Sex'].replace(['male','female'],[0,1],inplace=True)



# dropna() can used to remove values
df_train = df[['Survived','Pclass', 'Age_band', 'Sex']]
from sklearn.model_selection import train_test_split #training and testing data split



train,test=train_test_split(df_train,test_size=0.3,random_state=0,stratify=df_train['Survived'])

train_X=train[train.columns[1:]]

train_Y=train[train.columns[:1]]

test_X=test[test.columns[1:]]

test_Y=test[test.columns[:1]]

X=df_train[df_train.columns[1:]]

Y=df_train['Survived']

train_Y = np.ravel(train_Y)

test_Y=np.ravel(test_Y)

# LogisticRegression

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(train_X,train_Y)


model.score(test_X,test_Y)
from sklearn.model_selection import cross_val_score

cross_val_score(model, X, Y, cv=5)
# support vector Machine

from sklearn import svm

model=svm.SVC(kernel='rbf',C=1,gamma=0.1)

model.fit(train_X,train_Y)

model.score(train_X,train_Y)
# Random Forest

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)

model.fit(train_X,train_Y)

model.score(train_X,train_Y)
#k nearest neighbor

from sklearn.neighbors import KNeighborsClassifier

model=KNeighborsClassifier() 

model.fit(train_X,train_Y)

model.score(train_X,train_Y)