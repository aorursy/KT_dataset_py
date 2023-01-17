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



from sklearn.ensemble import RandomForestClassifier
# load data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



both = [train, test]



for s in both:

    s["Age"] = s["Age"].fillna( train["Age"].median() )

    s["Fare"] = s["Fare"].fillna( train["Fare"].median() )

    

    s["Sex"] = s["Sex"].map( {'female': 0, 'male': 0} ).astype(int)



    s["Embarked"] = s["Embarked"].fillna('S')

    s["Embarked"] = s["Embarked"].map( {'S': 0, 'C': 1, 'Q' : 2} ).astype(int)



print(train.head())
features = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

targets = train["Survived"].values



forest = RandomForestClassifier()

forest = forest.fit(features, targets)

print( forest.score(features, targets) )
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

out = forest.predict(test_features)
result = pd.DataFrame({ 'PassengerId': test['PassengerId'].values, 'Survived': out})

result.head()
result.to_csv('predictions_000.csv',index=False)