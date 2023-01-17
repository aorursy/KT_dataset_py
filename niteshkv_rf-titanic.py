# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv'

                        , header=0, encoding="ISO-8859-1")
train.head(6)
train[train.Fare == 0]
train.groupby('Fare').Cabin.nunique()
train.Fare
train.sort(['Fare'], ascending=[True])
train.sort(['Survived'], ascending=[True])
#train.drop(train[train.PassengerId == 259)

#df.drop(df[df.score < 50].index, inplace=True)

train.drop(train[train.PassengerId == 259].index, inplace=True)

train.drop(train[train.PassengerId == 528].index, inplace=True)

train.drop(train[train.PassengerId == 680].index, inplace=True)

train.drop(train[train.PassengerId == 738].index, inplace=True)
len(train[train.Cabin.isnull()])
test = pd.read_csv("../input/test.csv" , header=0, encoding="ISO-8859-1")
cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked'] 
colsRes = ['Survived']
di = {'S': 1, 'C': 2, 'Q': 3}

train["Embarked"].replace(di, inplace=True)
train.head(5)
len(train[train.Cabin.isnull()])
cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'] 
train.replace(['male', 'female'], 

                     [0, 1], inplace=True) 
train.head(5)
train.Age.unique()
mean_age_sur = int(train[train.Survived == 1][np.isfinite(train[train.Survived == 1].Age)].Age.sum()/

                   len(train[train.Survived == 1][np.isfinite(train[train.Survived == 1].Age)]))

mean_age_sur
mean_age_died = int(train[train.Survived == 0][np.isfinite(train[train.Survived == 0].Age)].Age.sum()/

                   len(train[train.Survived == 0][np.isfinite(train[train.Survived == 0].Age)]))

mean_age_died
NaN_Sur = {np.nan: 28}

NaN_Died = {np.nan: 30}

NaN_Age_Default = {np.nan : 29}

#train[train.Age.isnull() & train.Survived == 1]["Age"].replace(NaN_Sur, inplace=True)

train['Age'].replace(NaN_Age_Default, inplace=True)
train.Age.unique()
train.Embarked.unique()

train["Embarked"].replace({np.nan: 1}, inplace=True)

train.Embarked.unique()

trainArr = train.as_matrix(cols)

trainRes = train.as_matrix(colsRes)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=128)

rf.fit(trainArr, trainRes)

test.head(5)

test.Embarked.unique()
di = {'S': 1, 'C': 2, 'Q': 3}

test["Embarked"].replace(di, inplace=True)
test.Embarked.unique()
test.head(5)
test.replace(['male', 'female'], 

                     [0, 1], inplace=True) 
test.Age.unique()
test['Age'].replace(NaN_Age_Default, inplace=True)
test.Age.unique()
test.Embarked.unique()
test["Embarked"].replace({np.nan: 1}, inplace=True)
test.Embarked.unique()
test.Fare.unique()
test[np.isnan(test['Fare'])]
test["Fare"].replace({np.nan: test.Fare.sum()/len(test)}, inplace=True)
test.Fare.unique()
testArr = test.as_matrix(cols)
results = rf.predict(testArr)
test['Survived'] = results
test.head(5)
#test[['PassengerId', 'Survived']].to_csv('../input/submission_v3.csv', index=False)