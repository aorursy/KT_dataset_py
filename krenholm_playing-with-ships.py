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
titanic.head()
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler



stdScaler = StandardScaler()

labelEncoder = LabelEncoder()

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)



# Fix age

titanic[['Age']] = imputer.fit_transform(titanic[['Age']]) # Make better



# Numerical sex

titanic['Sex'] = labelEncoder.fit_transform(titanic['Sex'])



# Scale numerical values

titanic[['Age', 'Fare']] = stdScaler.fit_transform(titanic[['Age', 'Fare']])
titanic
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold



kf = KFold(n_splits=4, random_state=1)



model = RandomForestClassifier()



predictors = titanic[["Age", "Sex", "Fare"]]

targets = titanic["Survived"]



score = [model.fit(predictors.iloc[train,:], targets.iloc[train]).score(predictors.iloc[test,:], targets.iloc[test]) 

   for train, test in kf.split(predictors)]

score