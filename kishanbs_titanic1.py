## SAMPLE NOTEBOOK FOR TITANIC PREDICTION
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

import os

print(os.listdir("../input"))



# from IPython import InteractiveShell

# InteractiveShell.ast_node_interactivity="all"



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/gender_submission.csv')



df.head()



train_df=pd.read_csv('../input/train.csv')



train_df.head()



test_df=pd.read_csv('../input/test.csv')



test_df.head()

train_df.describe(include='all')
y=train_df.Survived



train_df.Sex=train_df["Sex"].map({'male': 1, 'female': 0})



train_df['Age'].fillna(train_df['Age'].median(), inplace=True)



train_df['Age']=pd.cut(train_df['Age'],bins=4)



label=LabelEncoder()



train_df['Age']=label.fit_transform(train_df['Age'])



train_df['SibSp'].fillna(train_df['SibSp'].median(),inplace=True)



train_df['Parch'].fillna(train_df['Parch'].median(),inplace=True)



train_df['Fare'].fillna(train_df['Fare'].median(),inplace=True)



train_df['Fare']=label.fit_transform(train_df['Fare'])

train_df['Cabin'].fillna('Missing',inplace=True)



train_df['Embarked'].fillna('Missing',inplace=True)



train_df['Cabin']=label.fit_transform(train_df['Cabin'])

train_df['Embarked']=label.fit_transform(train_df['Embarked'])

train_cols=['Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']



X=train_df.loc[:,train_cols]

y=np.array(y).reshape(-1,1)

model=LogisticRegression(max_iter=500)



X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=0.2,random_state=42)



model.fit(X_train,y_train)



model.predict(X_valid)
test_df.Sex=test_df["Sex"].map({'male': 1, 'female': 0})



test_df['Age'].fillna(test_df['Age'].median(), inplace=True)



test_df['Age']=pd.cut(test_df['Age'],bins=4)



label=LabelEncoder()



test_df['Age']=label.fit_transform(test_df['Age'])



test_df['SibSp'].fillna(test_df['SibSp'].median(),inplace=True)



test_df['Parch'].fillna(test_df['Parch'].median(),inplace=True)



test_df['Fare'].fillna(test_df['Fare'].median(),inplace=True)



test_df['Fare']=label.fit_transform(test_df['Fare'])

test_df['Cabin'].fillna('Missing',inplace=True)



test_df['Embarked'].fillna('Missing',inplace=True)



test_df['Cabin']=label.fit_transform(test_df['Cabin'])

test_df['Embarked']=label.fit_transform(test_df['Embarked'])
X_predict=test_df.loc[:,train_cols]



Survived=pd.DataFrame()



Survived['PassengerId']=test_df['PassengerId']



Survivedlist=list(model.predict(X_predict))



Survived['Survived']=Survivedlist



Survived.to_csv('Survived.csv',index=False)






