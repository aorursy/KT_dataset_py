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
titanic = pd.read_csv('../input/train.csv')
titanic.head(5)#Overview of the data
print(titanic.describe())#Some information of Characters
titanic["Age"] = titanic["Age"].fillna(titanic['Age'].median())#fill value 
print(titanic.describe())#Some information of Characters
titanic["Age"] = titanic["Age"].fillna(titanic['Age'].median())

titanic["Fare"] = titanic["Fare"].fillna(titanic['Fare'].median())

print(titanic.describe())#Some information of Characters
print(titanic["Sex"].unique())
titanic.loc[titanic["Sex"]=="male","Sex"]=0

titanic.loc[titanic["Sex"]=="female","Sex"]=1
print(titanic.describe())
from sklearn.linear_model import LinearRegression

from sklearn.cross_validation import KFold

predictors=["Pclass","Sex"]#,"Embarked"]
alg=LinearRegression()
kf=KFold(titanic.shape[0],n_folds=3,random_state=1)
predictions=[]
for train , test in kf:

    train_predictors=(titanic[predictors].iloc[train,:])

    train_target =titanic["Survived"].iloc[train]

    alg.fit(train_predictors,train_target)

    test_predictions = alg.predict(titanic[predictors].iloc[test,:])

    predictions.append(test_predictions)
predictions = np.concatenate(predictions, axis=0)

predictions
predictions[predictions > 0.5]=1

predictions[predictions <= 0.5]=0
predictions