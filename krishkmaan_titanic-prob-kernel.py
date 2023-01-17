import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

%matplotlib inline
sns.set()
df_train=pd.read_csv('../input/train.csv')
df_test=pd.read_csv('../input/test.csv')
df_train.head()
df_train.info()
sns.factorplot(x='Survived',col='Sex',kind='count',data=df_train)
sns.countplot(x='Survived',data=df_train)
sns.countplot(x='Sex',data=df_train)
df_train.groupby(['Sex']).Survived.sum()
print(df_train[df_train.Sex=='female'].Survived.sum()/df_train[df_train.Sex=='female'].Survived.count()*100)
print(df_train[df_train.Sex=='male'].Survived.sum()/df_train[df_train.Sex=='male'].Survived.count()*100)
sns.factorplot(x='Survived',col='Pclass',kind='count',data=df_train)
print(df_train[df_train.Pclass==1].Survived.sum()/df_train[df_train.Pclass==1].Survived.count()*100)
print(df_train[df_train.Pclass==2].Survived.sum()/df_train[df_train.Pclass==2].Survived.count()*100)
print(df_train[df_train.Pclass==3].Survived.sum()/df_train[df_train.Pclass==3].Survived.count()*100)
sns.factorplot(x='Survived',col='Embarked',kind='count',data=df_train)
sns.distplot(df_train.Fare,kde=False)
df_train.groupby(['Survived']).Fare.hist(alpha=0.5)
df_train_drop=df_train.dropna()
sns.distplot(df_train_drop.Age,kde=False)
sns.stripplot(x='Survived',y='Fare',data=df_train,alpha=0.3,jitter=True)
sns.swarmplot(x='Survived',y='Fare',data=df_train)
df_train.groupby('Survived').Fare.describe()
sns.lmplot(x='Age',y='Fare',hue='Survived',fit_reg=True,data=df_train,scatter_kws={'alpha':0.5})
sns.pairplot(df_train_drop,hue='Survived');
survived_train=df_train.Survived
data=pd.concat([df_train.drop(['Survived'],axis=1),df_test])
data.info()
data['Age']=data.Age.fillna(data.Age.median())
data['Fare']=data.Fare.fillna(data.Age.median())
data.info()


data=pd.get_dummies(data,columns=['Sex'],drop_first=True);





data.head()
data=data[['Sex_male','Fare','Age','Pclass','SibSp']]
data.head()
data.info()
data.describe()
data_train=data.iloc[:891]
data_test=data.iloc[891:]
x=data_train.values
test=data_test.values
surived_train=df_train.Survived
y=survived_train.values
clf=tree.DecisionTreeClassifier(max_depth=3)
clf.fit(x,y)
y_prediction=clf.predict(test)
df_test['Survived']=y_prediction
df_test[['PassengerId','Survived']].to_csv('Titanic_decisiontree',index=False)
df_test
