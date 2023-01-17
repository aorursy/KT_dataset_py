# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df    = pd.read_csv("/kaggle/input/titanic/test.csv")

train_df.head(10)
print(f"The train dataset contains %s rows and %s columns" %(train_df.shape[0], train_df.shape[1]))

print(f"The test dataset contains %s rows and %s columns" %(test_df.shape[0],test_df.shape[1]))
train_saved=train_df.copy(deep=True)



train_saved=train_saved.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

test_df=test_df.drop(['Name','Ticket','Cabin'],axis=1)

df_cleaning=[train_saved,test_df]

# df_cleaning[0]

# df_cleaning[1]

# Fill all NAN values before cleaning the Data by replacing NAN<->mean

df_cleaning[0]['Age'].fillna(df_cleaning[0].groupby("Pclass")['Age'].transform('mean'),inplace=True) #Trainig set NAN values replaced

df_cleaning[1]['Age'].fillna(df_cleaning[1].groupby("Pclass")['Age'].transform('mean'),inplace=True)

df_cleaning[0]['Fare'].fillna(df_cleaning[0].groupby("Pclass")["Fare"].transform('mean'),inplace=True)

df_cleaning[1]['Fare'].fillna(df_cleaning[1].groupby("Pclass")["Fare"].transform('mean'),inplace=True)



df_cleaning[0]["Embarked"].fillna(df_cleaning[0]["Embarked"].mode()[0],inplace=True)

df_cleaning[1]["Embarked"].fillna(df_cleaning[1]["Embarked"].mode()[0],inplace=True)

df_cleaning[0]['Age']
# train_df Create A FareBin numerical categorical feature for Fare and AgeBin numerical categorical feature for Age

df_cleaning[0]['FareBin']=pd.qcut(df_cleaning[0]['Fare'],4)

df_cleaning[0]['AgeBin'] = pd.cut(df_cleaning[0]['Age'].astype(int), 5)

# Test_df Create A FareBin numerical categorical feature for Fare and AgeBin numerical categorical feature for Age

df_cleaning[1]['FareBin']=pd.qcut(df_cleaning[1]['Fare'],4)

df_cleaning[1]['AgeBin'] = pd.cut(df_cleaning[1]['Age'].astype(int), 5)





train_sex_column=pd.get_dummies(train_df['Sex'],drop_first=True)

train_embark_column= pd.get_dummies(train_df['Embarked'],drop_first=True)

test_embark_column= pd.get_dummies(test_df['Sex'],drop_first=True)

test_embark_column= pd.get_dummies(test_df['Embarked'],drop_first=True)



label=LabelEncoder()



train_saved['Sex_Code']=label.fit_transform(train_saved['Sex'])

train_saved['Embarked_Code'] = label.fit_transform(train_saved['Embarked'])

train_saved['AgeBin_Code'] = label.fit_transform(train_saved['AgeBin'])

train_saved['FareBin_Code'] = label.fit_transform(train_saved['FareBin'])



#Same for the test set

test_df['Sex']

test_df['Sex_Code']=label.fit_transform(test_df['Sex'])

test_df['Embarked_Code'] = label.fit_transform(test_df['Embarked'])

test_df['AgeBin_Code'] = label.fit_transform(test_df['AgeBin'])

test_df['FareBin_Code'] = label.fit_transform(test_df['FareBin'])

print(train_saved['FareBin'])

print("--"*14)

test_df['FareBin']
df_cleaning[0].drop(['Sex','Embarked'],axis=1,inplace=True)

df_cleaning[1].drop(['Sex','Embarked'],axis=1,inplace=True)
print(test_df.head(10))

print(train_saved.head(10))
train_saved["Embarked_S"] = 0

train_saved['Embarked_S'].loc[train_saved['Embarked_Code'] == 2] = 1

train_saved["Embarked_Code"].loc[train_saved['Embarked_Code'] == 2] = 0

# Test set 

test_df["Embarked_S"] = 0

test_df['Embarked_S'].loc[test_df['Embarked_Code'] == 2] = 1

test_df["Embarked_Code"].loc[test_df['Embarked_Code'] == 2] = 0

train_saved.drop(['Age','Fare','FareBin','AgeBin'],axis=1,inplace=True)

test_df.drop(['Age','Fare','FareBin','AgeBin'],axis=1,inplace=True)



print("The train dataset contains %s rows and %s columns" %(train_saved.shape[0], train_saved.shape[1]))

print("The test dataset contains %s rows and %s columns" %(test_df.shape[0],test_df.shape[1]))

train_saved.head()
RandomForest=RandomForestClassifier()

RandomForest.fit(train_saved.drop(['Survived'],axis=1),train_saved['Survived'])

predicted_y= RandomForest.predict(test_df.drop(['PassengerId'],axis=1))
submission=test_df['PassengerId']

submission=pd.DataFrame(submission)

submission["Survived"]=predicted_y
submission.to_csv("submit.csv", index=False)
submit_csv