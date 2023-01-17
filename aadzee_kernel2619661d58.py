# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/titanic-solution-for-beginners-guide/train.csv')
df.head()
y=df.pop("Survived")
y.head()
df.info()

numeric_variables=list(df.dtypes[df.dtypes!="object"].index)
df[numeric_variables].head()
df["Age"].fillna(df.Age.mean(),inplace=True)
model= RandomForestClassifier(n_estimators=2500)

model.fit(df[numeric_variables],y)
print("training accuracy:" , accuracy_score(y, model.predict(df[numeric_variables])))
test=pd.read_csv("../input/titanic-solution-for-beginners-guide/test.csv")
test["Age"].fillna(df.Age.mean(),inplace=True)
test=test[numeric_variables].fillna(test.mean()).copy()
test

y_pred=model.predict(test[numeric_variables])
submission=pd.DataFrame({"PassengerId":test["PassengerId"],"Survived":y_pred})

submission.to_csv("titanic.csv",index=False)