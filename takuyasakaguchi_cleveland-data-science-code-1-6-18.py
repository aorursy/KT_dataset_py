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
df_train = pd.read_csv("../input/train.csv")
df_train
df_test = pd.read_csv('../input/test.csv')
df_test
df_train.shape
df_test.shape
df_combined = pd.concat([df_train, df_test])
df_combined.head()
df_combined.index = range(1, len(df_combined)+1)
df_combined.tail()
def gender_conversion(gender):

    if gender =="female":

        return 0

    else:

        return 1
df_combined.Sex = df_combined['Sex'].apply(gender_conversion)
df_combined.head()
df_combined[["Age", "Fare", "Pclass", "Sex","Survived"]]
df = df_combined[["Age", "Fare", "Pclass", "Sex","Survived"]]
df = df.fillna(-1)
df
train = df[df.Survived>=0]
train
test = df[df.Survived == -1]
test
y_train = train.pop("Survived")
x_train = train
x_test = test.pop("Survived")
y_test = x_test
x_test = test
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
Xtrain, Xtest, ytrain, ytest = train_test_split(x_train, y_train)
clf= RandomForestClassifier(n_estimators = 200)
clf.fit(Xtrain,ytrain)
ypred = clf.predict(Xtest)
metrics.accuracy_score(ypred,ytest)