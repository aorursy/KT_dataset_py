# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import DataFrame,Series

from sklearn.ensemble import RandomForestClassifier

import sklearn as sk

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

test_df=pd.read_csv('../input/test.csv')
train_df.head()
train_df.drop(["Name","Sex","Age","Ticket","Fare","Cabin","Embarked"],axis=1,inplace=True

            )

train_df.info()
train_df.drop(["Name","Sex","Age","Ticket","Fare","Cabin","Embarked"],axis=1,inplace=True

            )
train_df.info()
model=RandomForestClassifier(n_estimators=100)

model.fit(train_df.drop(["Survived"],axis=1),train_df["Survived"])
test_df.drop(["Name","Sex","Age","Ticket","Fare","Cabin","Embarked"],axis=1,inplace=True

            )

test_df.info()
ypredict=model.predict(test_df)

ypredict
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": ypredict

    })

submission.to_csv('titanic.csv', index=False)