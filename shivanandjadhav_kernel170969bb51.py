import pandas as pd
df=pd.read_csv('../input/titanic/train.csv')

dft=pd.read_csv('../input/titanic/test.csv')
dft.head()
df.head()
df.shape
dft.shape
df.info()
df.describe()
df.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt

sns.heatmap(df.isnull())
df.drop(columns="Cabin",axis=1,inplace=True)#inplace=True
df.head()
dft.drop(columns="Cabin",axis=1,inplace=True)#inplace=True

dft.head()
df.corr()
sns.heatmap(df.corr(),annot=True)
df.Age.fillna(df.Age.mean(),inplace=True)

dft.Age.fillna(dft.Age.mean(),inplace=True)
df.Embarked.fillna(df.Embarked.mode()[0],inplace=True)

dft.Fare.fillna(dft.Fare.mean(),inplace=True)
df.isna().sum()
dft.isna().sum()
sns.countplot(df.Survived,hue=df.Sex)#hue
sns.countplot(df.Survived,hue=df.Sex)
sns.distplot(df.Age)
sns.violinplot(df.Survived,df.Sex,hue=df.Pclass)
'''count plot

distplot

pairplot

violin plot 

box plot 

hist 

#hue --- categ'''
x1=df.copy()

y1=dft.copy()
df.Sex=pd.get_dummies(df.Sex,drop_first=True)

df.head()
dft.Sex=pd.get_dummies(dft.Sex,drop_first=True)

dft.head()
df.Embarked=pd.get_dummies(df.Embarked,drop_first=True)

dft.Embarked=pd.get_dummies(dft.Embarked,drop_first=True)
df.head()
dft.head()
x1.Embarked.unique()
df.Embarked.unique()
df.columns
remove=['PassengerId','Name','Ticket', 'Embarked']
df.drop(columns=remove,inplace=True)

dft.drop(columns=remove,inplace=True)

df.head()
dft.head()
from sklearn.model_selection import train_test_split
target=df.Survived#.values
df.drop(columns="Survived",inplace=True)
df.head()
df_train,df_test,target_train,target_test=train_test_split(df,target,test_size=.30,random_state=0)
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV

lr=LogisticRegression()

lrcv=LogisticRegressionCV(cv=5,random_state=0)
lr.fit(df_train,target_train)
target.shape
result=lr.predict(df_test)
from sklearn.metrics import accuracy_score
accuracy_score(target_test,result)
plt.figure(figsize=(15,8))

sns.scatterplot(df.Age,df.Fare,hue=x1.Survived,alpha=.6)
lrcv.fit(df_train,target_train)
pred_df=lrcv.predict(df_test)
accuracy_score(target_test,pred_df)
from sklearn.model_selection import GridSearchCV
gscv=GridSearchCV(lr,

    {"C":[1,2,3]},

    cv=5,

    return_train_score=False,

)
gscv.fit(df_train,target_train)
pred_gs=gscv.predict(df_test)
accuracy_score(target_test,pred_gs)