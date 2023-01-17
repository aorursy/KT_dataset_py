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



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

print("*"*10, "Dataset Information", "*"*10)

print(train_df.info())
print("*"*10, "First five train file rows", "*"*10)

train_df.head(5)
train_df.info()
train_df.drop(['Name','Age','Ticket','Fare','Cabin'], inplace = True, axis =1)

test_df.drop(['Name','Age','Ticket','Fare','Cabin'], inplace = True, axis =1)
train_df.Embarked.value_counts()
#Embarked null fix

data = [train_df, test_df]



for dataset in data:

    dataset.Embarked = dataset.Embarked.fillna("S")
train_df.info()
train_df["Sex"].value_counts()
genderMap = {'male': 0, 'female':1}

data = [train_df, test_df]



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genderMap)

    

test_df["Embarked"].value_counts()
embarkedMap = {"S":0, "C":1, "Q":2}

data = [train_df, test_df]



for dataset in data:

    dataset["Embarked"] = dataset["Embarked"].map(embarkedMap)

    
print(train_df.info(), test_df.info())
train_df.drop(['Embaked'], inplace = True, axis =1)

test_df.drop(['Embaked'], inplace = True, axis =1)
X_train = train_df.drop(['Survived','PassengerId'], axis=1)

Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0)

clf.fit(X_train, Y_train)

Y_predict = clf.predict(X_test)
print(Y_predict)
acc_logistic = round (clf.score(X_train, Y_train)*100, 2)

print(acc_logistic)
output = pd.DataFrame ({'PassengerId': test_df.PassengerId, 'Survived': Y_predict})

output.to_csv('my_submission.csv', index= False)

print("Your Submission was successfully saved")