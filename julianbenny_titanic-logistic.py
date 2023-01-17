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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train_data.head()
test_data.head()
train_data.drop(["Ticket","Embarked"],axis=1,inplace=True)

test_data.drop(["Ticket","Embarked"],axis=1,inplace=True)
train_data.isnull().sum()
train_data["Title"] = np.zeros(len(train_data))

test_data["Title"] = np.zeros(len(test_data))





for i in range(len(train_data)):

    if ("Mr." in train_data['Name'][i]):

        train_data["Title"].iloc[i] = 0

    else:

        train_data["Title"].iloc[i] = 1



for i in range(len(test_data)):

    if ("Mr." in test_data['Name'][i]):

        test_data["Title"].iloc[i] = 0

    else:

        test_data["Title"].iloc[i] = 1

        
train_data.drop("Name",axis=1,inplace=True)

test_data.drop("Name",axis=1,inplace=True)
train_data.groupby(['Pclass', 'Title'])['Age'].agg(['mean', 'median']).round(1)
train_data["Age"].value_counts(dropna=False)
check = train_data["Age"].isna()
for i in range(len(train_data)):

    if check[i] == True:

        if train_data["Pclass"][i] == 1:

            if train_data["Title"][i] == 0:

                train_data["Age"][i] = 40.8

            else:

                train_data["Age"][i] = 35.1



        elif train_data["Pclass"][i] == 2:

            if train_data["Title"][i] == 0:

                train_data["Age"][i] = 31.9

            else:

                train_data["Age"][i] = 27.7



        else:

            if train_data["Title"][i] == 0:

                train_data["Age"][i] = 27.3

            else:

                train_data["Age"][i] = 18.3

        

train_data
test_data.groupby(['Pclass', 'Title'])['Age'].agg(['mean', 'median']).round(1)
test_data["Age"].value_counts(dropna=False)
check_test = test_data["Age"].isna()

check_test.value_counts()
for i in range(len(test_data)):

    if check_test[i] == True:

        if test_data["Pclass"][i] == 1:

            if test_data["Title"][i] == 0:

                test_data["Age"][i] = 41

            else:

                test_data["Age"][i] = 43



        elif test_data["Pclass"][i] == 2:

            if test_data["Title"][i] == 0:

                test_data["Age"][i] = 32

            else:

                test_data["Age"][i] = 25.7



        else:

            if test_data["Title"][i] == 0:

                test_data["Age"][i] = 27.4

            else:

                test_data["Age"][i] = 19
test_data.isnull().sum()
# train_data["Cabin"].fillna(value="Cabin_unk",inplace=True)

# test_data["Cabin"].fillna(value="Cabin_unk",inplace=True)

print(train_data.shape)

print(test_data.shape)
gender_map = {

    "male":0,

    "female":1

}



train_data.Sex = train_data.Sex.map(gender_map)

test_data.Sex = test_data.Sex.map(gender_map)
train_data["Fare"] =  (train_data["Fare"] - train_data["Fare"].mean(axis=0)) / (train_data["Fare"].max() - train_data["Fare"].min(axis=0))

test_data["Fare"] =  (test_data["Fare"] - test_data["Fare"].mean(axis=0)) / (test_data["Fare"].max() - test_data["Fare"].min(axis=0))
# dummies=pd.get_dummies(train_data['Cabin'])

# dummies.columns=str(i)+'_'+dummies.columns

# train_data=pd.concat([train_data,dummies],axis=1)

# train_data.drop("Cabin",axis=1,inplace=True)
# dummies=pd.get_dummies(test_data['Cabin'])

# dummies.columns=str(i)+'_'+dummies.columns

# test_data=pd.concat([test_data,dummies],axis=1)

# test_data.drop("Cabin",axis=1,inplace=True)
print(train_data.shape)

print(test_data.shape)
for i in train_data.drop("Survived",axis=1).columns:

    if i not in test_data.columns:

        test_data[i] = np.zeros(len(test_data))



for i in test_data.columns:

    if i not in train_data.columns:

        train_data[i] = np.zeros(len(train_data))
print(train_data.shape)

print(test_data.shape)
test_data = test_data[train_data.drop("Survived",axis=1).columns]
train_data.keys()
test_data.keys()
train_data.drop(["PassengerId","Cabin"],axis=1,inplace=True)

test_data.drop("Cabin",axis=1,inplace=True)
print(train_data.shape)

print(test_data.shape)
# train_data = train_data.dropna(axis=0)

train_data.isnull().sum()
test_data.isnull().sum()
test_data["Fare"].fillna(value=6.375629e-17,inplace=True)
X = train_data.drop("Survived",axis=1)

y = train_data['Survived']



from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)



print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver="lbfgs")

lr.fit(X_train,y_train)

lr.score(X_test,y_test)
print(lr.coef_)
train_data.keys()
test_data.keys()
pred = lr.predict(test_data.drop("PassengerId",axis=1))
submission = pd.DataFrame(data=pred,columns=["Survived"])
submission["PassengerId"] = test_data['PassengerId']
submission.set_index("PassengerId",inplace=True)
submission.to_csv("submission.csv")