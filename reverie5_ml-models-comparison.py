import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#read data

train= pd.read_csv('/kaggle/input/titanic/train.csv')

test= pd.read_csv('/kaggle/input/titanic/test.csv')

sample= pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
#looking at data

print(train.shape)

train.head()
print(test.shape)

test.head()
print(sample.shape)

sample.head()
#descriptive analysis

print(train.describe(include='all'))

print('-------------------')

print(test.describe())
#data prep: appending both train and test to preprocess together

df= train.append(test)

df.shape
#null check and impute if any

df.isnull().sum()
print('df shape= ',df.shape)

print('df passenger id unique= ',df.PassengerId.nunique())

print('df cabin total null val= ',df.Cabin.isnull().sum())

print(train.Cabin.isnull().sum())

print(test.Cabin.isnull().sum())
print(df.Ticket.value_counts())

print(df.Ticket.nunique())
print(df[df.Ticket=='1601'])
#dropping cols

df.drop(['Cabin'], axis=1, inplace = True)

print(df.shape)
#check dtypes

df.dtypes
# Imputing missing val for age, embarked and fare

df['Age'].fillna(df['Age'].median(), inplace = True)

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)

df['Fare'].fillna(df['Fare'].median(), inplace = True)
df.isnull().sum()
df.head()
df.Embarked.value_counts()
# encoding

cal_cols= ['Embarked','Sex']

df= pd.get_dummies(df, columns= cal_cols)
df.head()
#splitting train and test from df

train= df[df['Survived'].isnull()!= True]

test= df[df['Survived'].isnull()== True].drop(['Survived'], axis=1)

print(train.shape)

print(test.shape)
test.isnull().sum()
train.columns
# local validation split

features=['Age', 'Fare', 'Parch', 'Pclass', 'SibSp', 'Embarked_C', 'Embarked_Q', 'Embarked_S',

       'Sex_female', 'Sex_male']

print(len(features))



from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y= train_test_split(train[features], train.Survived, test_size=0.2, random_state=123)

print(train_x.shape)

print(train_y.shape)

print('---------------')

print(val_x.shape)

print(val_y.shape)
from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier



from sklearn.linear_model import LogisticRegression



from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier



from sklearn.metrics import confusion_matrix, accuracy_score



models = [

    #ensemble

    AdaBoostClassifier(),

    ExtraTreesClassifier(),

    GradientBoostingClassifier(),

    RandomForestClassifier(),

    DecisionTreeClassifier(),

    

    #linear models

    LogisticRegression(),

          

    XGBClassifier(),

    LGBMClassifier(),

    CatBoostClassifier()

         ]
#modelling

df_models = pd.DataFrame(columns=['Model_name','Accuracy'])



i=0

for model in models:

    model.fit(train_x,train_y)

    pred_y = model.predict(val_x)

    acc_score = accuracy_score(val_y, pred_y)

    name = str(model)

    print(name[0:name.find("(")])

    df_models.loc[i,'Model_name']= name[0:name.find("(")]

 

    df_models.loc[i,'Accuracy']= acc_score

    print(confusion_matrix(val_y,pred_y))

    print("------------------------------------------------------------")

    i=i+1
#models comparison

df_models.sort_values('Accuracy', ascending=False)
#for submission- training on whole trainset and prediction on test set using our best model

model = LGBMClassifier()

model.fit(train[features], train.Survived)

pred_y = model.predict(test[features])
submission = pd.DataFrame()

submission['PassengerId'] = test['PassengerId']

submission['Survived'] = pred_y.astype(int)

submission.to_csv('lgbm.csv', index=False)

submission.shape
submission