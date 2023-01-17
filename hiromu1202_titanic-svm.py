# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC,LinearSVC
import matplotlib.pyplot as plt
def make_input(data):
    input = pd.concat(
        [
            data["Pclass"],
            pd.get_dummies(data["Sex"]),
            data["Age"],
            data["SibSp"],
            data["Parch"],
            data["Fare"],
            pd.get_dummies(data["Embarked"])
        ], axis=1
    )
    return input.fillna(input.mean())

def make_output(data):
    return data["Survived"].values
train = pd.read_csv('../input/train.csv').sample(frac=1).reset_index(drop=True)
trainX = make_input(train)
trainY = make_output(train)
C = 5
lin_clf = LinearSVC(loss="hinge", C=C, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(trainX)

lin_clf.fit(X_scaled, trainY)
test = pd.read_csv('../input/test.csv').sample(frac=1).reset_index(drop=True)
testX = make_input(test)
testX_scaled = scaler.fit_transform(testX)
predict = lin_clf.predict(testX)
output = pd.concat(
        [
            test['PassengerId']
        ], axis=1
    )
output['Survived'] = predict
output.to_csv('submission.csv',index=False)
