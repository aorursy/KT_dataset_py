import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
test.head()
train.tail()
train.Sex.value_counts()
train.isna().sum()
train.Age.value_counts()
train.shape
train.Pclass.value_counts()
missing_val = (train.isna().sum()*100)/len(train)
missing_df = pd.DataFrame({'columns' : train.columns,'missing_%': missing_val})
missing_df
missing_v = (test.isna().sum()*100)/len(train)
missing_test = pd.DataFrame({'columns' : test.columns,'missing_%': missing_v})
missing_test
train.info()
train['Age'] = train['Age'].fillna(train.Age.median(),axis=0)
train['Embarked'] = train['Embarked'].fillna(train.Embarked.mode()[0],axis=0)
test['Age'] = test['Age'].fillna(test.Age.median(),axis=0)
test['Fare'] = test['Fare'].fillna(test.Fare.median(),axis=0)
train.isna().sum()
train.drop(['Cabin'],axis =1,inplace=True)
test.drop(['Cabin'],axis=1,inplace=True)
train.info()
test.isna().sum()
train.isna().sum()
num_col = train.select_dtypes(include = ['int64','float64']).columns
cat_col = train.select_dtypes(include = ['object']).columns
cat_test = test.select_dtypes(include = ['object']).columns
train.head()
num_col,cat_col
import matplotlib.pyplot as plt
import seaborn as sns

train.describe()
sns.boxplot(train['Age']);
plt.hist(train['Age']);
sns.boxplot(train['Fare']);
plt.hist(train['Fare']);
train.columns
train=train[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Embarked', 'Survived']]
train.head()
train.corr()
#Encoding Categorical Variables
#def encoding_cat(col):
lbl=LabelEncoder()
for col in cat_col:
    train[col]= lbl.fit_transform(train[col].astype(str))
for col in cat_test:
    test[col] = lbl.fit_transform(test[col].astype(str))
test.head()
train.head()
def split_data(train_data,test_data):
    X_train = train_data.values[:,:-1]
    y_train = train_data.values[:,-1]
    X_test = test_data
   
    #X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.2,random_state = 100)
    return X_train,X_test,y_train #,y_test 
def train_gini_index(X_train,y_train):
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth =3 ,min_samples_leaf = 10)
    clf_gini.fit(X_train,y_train)
    return clf_gini
def train_entropy(X_train,y_train):
    clf_entropy = DecisionTreeClassifier(criterion= 'entropy',random_state= 100,max_depth =3 ,min_samples_leaf = 10)
    clf_entropy.fit(X_train,y_train)
    return clf_entropy
def prediction(X_test,clf):
    y_pred = clf.predict(X_test)
    return y_pred
def model_score(X_train,y_train,clf):
    print(round(clf.score(X_train,y_train)*100,2))
    return round(clf.score(X_train,y_train)*100,2)
def main():
    X_train,X_test,y_train = split_data(train,test)
    clf_gini=train_gini_index(X_train,y_train)
    clf_entropy=train_entropy(X_train,y_train)
   
    
    #Predicition using Gini Index
    print("Predicition using Gini Index")
    y_pred_gini = prediction(X_test,clf_gini)
    gini_score = model_score(X_train,y_train,clf_gini)
    #col_survived('Survived_Gini',y_pred_gini)
    #y_test=test['Survived_Gini']
    #calc_accuracy(y_test,y_pred_gini)
    
    #Predicition using Entropy
    print("Predicition using Entropy")
    y_pred_entropy = prediction(X_test,clf_entropy)
    ent_score = model_score(X_train,y_train,clf_entropy)
    #calc_accuracy(y_test,y_pred_entropy)
    
    #calc_accuracy(y_test,y_pred)
    if gini_score>ent_score:
        final_pred = y_pred_gini
    else:
        final_pred = y_pred_entropy
    submission  = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':final_pred})
    submission.to_csv('submission.csv',index=False)


if __name__=="__main__":
     main()

