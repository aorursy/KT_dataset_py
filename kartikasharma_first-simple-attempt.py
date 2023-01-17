# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#reading files
train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
test.head()
#combining parch , sib SP to one family size column
train['family_size'] = train['SibSp'] + train['Parch'] +1 #+1 for passenger itself
test['family_size'] = test['SibSp'] + test['Parch'] +1

train.head()
sns.barplot(x='Sex',y='Survived',data=train)
sns.barplot(x='Embarked',y='Survived',data=train)
sns.barplot(x='Pclass',y='Survived',data=train)
#null values
train.isna().sum()
test.isna().sum()
sns.heatmap(train.isna())
#replacing Embarked column missing values with most frequent ones
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy='most_frequent')
imputer2 = SimpleImputer(missing_values= np.nan, strategy='median')
train["Embarked"]=imputer.fit_transform(train[["Embarked"]]).ravel()
test["Embarked"]=imputer.fit_transform(test[["Embarked"]]).ravel()
test["Fare"]=imputer2.fit_transform(test[["Fare"]]).ravel()
train.corr()
plt.figure(figsize=(6,6))
sns.boxplot(x='Pclass' , y='Age', data = train)
def age1(cols):
    Age=cols[0]
    Pclass=cols[1]
    if Pclass==1:
        return 37
    elif Pclass==2:
        return 29
    elif Pclass==3:
        return 24
    else:
        return Age
train["Age"]=train[["Age","Pclass"]].apply(age1,axis=1)
test["Age"]=test[["Age","Pclass"]].apply(age1,axis=1)
sns.heatmap(train.isna())
train.isna().sum()
test.isna().sum()
#extracting name titles
train['name_title'] = train['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)
test['name_title'] = test['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)
train["name_title"].value_counts()
test["name_title"].value_counts()
train['name_title'] = train['name_title'].replace('Mlle' , 'Miss')
train['name_title'] = train['name_title'].replace('Mlle' , 'Miss')
train['name_title'] = train['name_title'].replace('Ms' , 'Mrs')
train['name_title'] = train['name_title'].replace(['Dr','Rev','Major','Col','Capt','Jonkheer','Mme','Don'] , 'others')
train['name_title'] = train['name_title'].replace(['Master','Lady','Sir','Countess'] , 'royals')

test['name_title'] = test['name_title'].replace('Mlle' , 'Miss')
test['name_title'] = test['name_title'].replace('Mlle' , 'Miss')
test['name_title'] = test['name_title'].replace('Ms' , 'Mrs')
test['name_title'] = test['name_title'].replace(['Dr','Rev','Major','Col','Capt','Jonkheer','Mme','Dona'] , 'others')
test['name_title'] = test['name_title'].replace(['Master','Lady','Sir','Countess'] , 'royals')

test["name_title"].value_counts()
#turning categorical features to numerical ones
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

train.iloc[:,4] = label.fit_transform(train.iloc[:,4].values)
train.iloc[:,11] = label.fit_transform(train.iloc[:,11].values)

test.iloc[:,3] = label.fit_transform(test.iloc[:,3].values)
test.iloc[:,10] = label.fit_transform(test.iloc[:,10].values)


title_map = {'Mr':1,'Miss':2,'Mrs':3,'royals':4,'others':5}
train['name_title'] = train['name_title'].map(title_map)
test['name_title'] = test['name_title'].map(title_map)
#dropping unwanted columns

train.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)
pid = test['PassengerId']
test.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)
test.head()
x=train.iloc[:,1:8]
y=train.iloc[:,0]
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state = 0)


train
#predicting model results
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
accuracy_lr=round(accuracy_score(y_pred,y_test)*100, 2)
print('accuracy_lr')
print(accuracy_lr)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski', p=2)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
accuracy_knn=round(accuracy_score(y_pred,y_test)*100, 2)
print('accuracy_knn')
print(accuracy_knn)
    
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)
tree.fit(x_train,y_train)
y_pred=tree.predict(x_test)
accuracy_tree=round(accuracy_score(y_pred,y_test)*100, 2)
print('accuracy_tree')
print(accuracy_tree)
    
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10,criterion = 'entropy',random_state=0)
forest.fit(x_train,y_train)
y_pred=forest.predict(x_test)
accuracy_forest=round(accuracy_score(y_pred,y_test)*100, 2)
print('accuracy_forest')
print(accuracy_forest)
prediction = forest.predict(test)
final = pd.DataFrame({'PassengerId':pid , 'Survived':prediction})

final.to_csv('submission.csv',index=False)