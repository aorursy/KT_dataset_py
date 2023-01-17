import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.ensemble import RandomForestClassifier
dtrain=pd.read_csv(r'../input/titanic/train.csv')

dtest = pd.read_csv(r'../input/titanic/test.csv')
dtrain.head()
X=dtrain[['Name','Pclass','Age','Embarked','SibSp','Sex','Parch','Fare','Cabin']]

y=dtrain.Survived

X_test=dtest[['Name','Pclass','Age','Embarked','SibSp','Sex','Parch','Fare','Cabin']]
#Add Family coulmn by adding SibSp and Parch

#X['name_len'] = X.Name.apply(lambda x:len(x)%25)

X['Family']=X['SibSp']+X['Parch']+1

X = X.drop(columns=['SibSp','Parch'])
#Add Title as Type of Person 

X['Title']= X['Name'].str.extract('([A-Za-z]+\.)',expand=False)

X['Title'].unique()

X['Title'].value_counts(normalize=True)*100
X['Title'] = X['Title'].replace(['Don.', 'Rev.', 'Dr.','Major.', 'Lady.', 'Sir.', 'Col.', 'Capt.','Countess.', 'Jonkheer.'],'Rare.')

X['Title'] = X['Title'].replace('Mlle.', 'Miss.')

X['Title'] = X['Title'].replace('Ms.', 'Miss.')

X['Title'] = X['Title'].replace('Mme.', 'Mrs.')



mappings_title={"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4, "Rare.": 5}

X['Title'] = X['Title'].map(mappings_title)

X['Title'].fillna(0)

X['Title'].value_counts()
X=X.drop(columns=['Name'])

X
X.loc[X.Cabin.isnull(),'HasCabin']  = 0

X.loc[X.Cabin.notnull(),'HasCabin'] = 1

X = X.drop(columns=['Cabin'],axis=1)
X['Embarked'] = X.Embarked.fillna('S')

X['Fare'] = X.Fare.fillna(X.Fare.median())


# mappings_title={"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4, "Rare.": 5}

for i in range(0,6):

    X.loc[(X.Age.isnull())&(X.Title==i),'Age'] = X[X["Title"]==i].Age.dropna().median()
X.loc[:,'Embarked'] = X.Embarked.map({'S':1,'C':2,'Q':3})

X.loc[:,'Sex'] = X.Sex.map({'male':1,'female':2})

X.isnull().sum()
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=17,train_size=0.8)
categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
categorical_cols,numerical_cols
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier(n_estimators = 200,criterion='gini', max_depth=12, max_features='auto')

# from xgboost import XGBClassifier 

#model = XGBClassifier()

model.fit(X_train,y_train)
pred_valid = model.predict(X_valid)

cm = confusion_matrix(pred_valid,y_valid)

acc = accuracy_score(pred_valid,y_valid)

print(cm,acc)
X_test=dtest[['Name','Pclass','Age','Embarked','SibSp','Sex','Parch','Fare','Cabin']]

#Add Family coulmn by adding SibSp and Parch

#X_test['name_len'] = X_test.Name.apply(lambda x:len(x)%25)

X_test['Family']=X_test['SibSp']+X_test['Parch']+1

X_test = X_test.drop(columns=['SibSp','Parch'])
X_test['Title']= X_test['Name'].str.extract('([A-Za-z]+\.)',expand=False)

X_test['Title'].value_counts()


X_test['Title'] = X_test['Title'].replace(['Dona.','Don.', 'Rev.', 'Dr.','Major.', 'Lady.', 'Sir.', 'Col.', 'Capt.','Countess.', 'Jonkheer.'],'Rare.')

X_test['Title'] = X_test['Title'].replace('Mlle.', 'Miss.')

X_test['Title'] = X_test['Title'].replace('Ms.', 'Miss.')

X_test['Title'] = X_test['Title'].replace('Mme.', 'Mrs.')



mappings_title={"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4, "Rare.": 5}

X_test['Title'] = X_test['Title'].map(mappings_title)

X_test['Title'].fillna(0)

X_test['Title'].isnull()

X_test.loc[X_test.Cabin.isnull(),'HasCabin']  = 0

X_test.loc[X_test.Cabin.notnull(),'HasCabin'] = 1

X_test = X_test.drop(columns=['Cabin','Name'],axis=1)
X['Fare'] = X.Fare.fillna(X.Fare.median())

X_test.loc[:,'Embarked'] = X_test.loc[:,'Embarked'].fillna('S')

for i in range(0,6):

    X_test.loc[(X_test.Age.isnull())&(X_test.Title==i),'Age'] = X_test[X_test['Title']==i].Age.dropna().median()
X_test.loc[:,'Embarked'] = X_test.Embarked.map({'S':1,'C':2,'Q':3})

X_test.loc[:,'Sex'] = X_test.Sex.map({'male':1,'female':2})

X_test.isnull().sum()
X_test['Fare'] = X_test.Fare.fillna(X_test.Fare.mean())

X_test.isnull().sum()
results = model.predict(X_test)

output = pd.DataFrame({'PassengerId': dtest.PassengerId,

                       'Survived': results})

output.to_csv('submission.csv', index=False)