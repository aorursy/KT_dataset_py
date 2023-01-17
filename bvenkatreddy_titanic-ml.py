# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.metrics import accuracy_score,recall_score,precision_score,confusion_matrix,f1_score

from IPython.display import Image  

from sklearn import tree

import os







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
train= pd.read_csv('../input/train.csv')

test =pd.read_csv('../input/test.csv')
train.Sex[train.Sex=='male']=0

train.Sex[train.Sex=='female']=1

test.Sex[test.Sex=='male']=0

test.Sex[test.Sex=='female']=1
train.head(4)
PassangerID= test.PassengerId

train['Embarked'].value_counts()
PassangerID.head()
plt.figure(figsize=(25,10))

sns.barplot(train['Age'],train['Survived'], ci=None)

plt.xticks(rotation=90);

train.Age[train.Pclass == 1].plot(kind='kde')    

train.Age[train.Pclass == 2].plot(kind='kde')

train.Age[train.Pclass == 3].plot(kind='kde')

 # plots an axis lable

plt.xlabel("Age")    

plt.title("Age Distribution within classes")

# sets our legend for our graph.

plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') ;
limit_1 = 12

limit_2 = 50



x_limit_1 = np.size(train[train['Age'] < limit_1]['Age'].unique())

x_limit_2 = np.size(train[train['Age'] < limit_2]['Age'].unique())

df=train



plt.figure(figsize=(25,10))

sns.barplot(df['Age'],df['Survived'], ci=None)



plt.axvspan(-1, x_limit_1, alpha=0.25, color='green')

plt.axvspan(x_limit_1, x_limit_2, alpha=0.25, color='red')

plt.axvspan(x_limit_2, 100, alpha=0.25, color='yellow')



plt.xticks(rotation=90);
plt.figure(figsize=(25,10))

sns.barplot(df['Sex'],df['Survived'], ci=None)
plt.figure(figsize=(25,10))

sns.barplot(df['Pclass'],df['Survived'], ci=None)
full_data = [train, test]



# Some features of my own that I have added in

# Gives the length of the name

train['Name_length'] = train['Name'].apply(len)

test['Name_length'] = test['Name'].apply(len)

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S') 

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())    

    dataset['CategoricalFare'] = pd.qcut(train['Fare'], 4,labels= [1,2,3,4])

    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())

train['Age'] = pd.cut(train['Age'], bins=[0, 12, 50, 200], labels=[1,2,3])

train["Embarked"][train["Embarked"] == "S"] = 0

train["Embarked"][train["Embarked"] == "C"] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2

test['Age'] = pd.cut(test['Age'], bins=[0, 12, 50, 200], labels=[1,2,3])

test["Embarked"][test["Embarked"] == "S"] = 0

test["Embarked"][test["Embarked"] == "C"] = 1

test["Embarked"][test["Embarked"] == "Q"] = 2

train.head()
plt.figure(figsize=(25,10))

sns.barplot(df['Name_length'],df['Survived'], ci=None)
plt.figure(figsize=(25,10))

sns.barplot(df['Has_Cabin'],df['Survived'], ci=None)
plt.figure(figsize=(25,10))

sns.barplot(df['Embarked'],df['Survived'], ci=None)
plt.figure(figsize=(25,10))

sns.barplot(df['CategoricalFare'],df['Survived'], ci=None)

plt.figure(figsize=(25,10))

sns.barplot(df['IsAlone'],df['Survived'], ci=None)
train = train.drop(['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin'], axis=1)

test = test.drop(['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin'], axis=1)

train.head()
x= train.iloc[:, 1:10].values

y=train.iloc[:, 0].values

x_pred= test.iloc[:, 0:9].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=1/5,random_state = 0)

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)

model=regressor.fit(x_train,y_train)
test.head()

y_pred=regressor.predict(x_test)

y_pred[y_pred<0.5]=0

y_pred[y_pred>=0.5]=1

y_pred

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#print("accuracy score : "+str(accuracy_score(y_test,y_pred))) #can be used too :)

print("precision score : "+str(precision_score(y_test,y_pred)))
x_test[1]
feat_names = x_test[1]

targ_names = [1,0]

from sklearn.tree import DecisionTreeClassifier,export_graphviz

import graphviz



data = export_graphviz(model,out_file=None,feature_names=feat_names,class_names=targ_names,   

                         filled=True, rounded=True,  

                         special_characters=True)

graph = graphviz.Source(data)

graph 
