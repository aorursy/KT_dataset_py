# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import svm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "."]).decode("utf8"))

#print(check_output(["rm", "submission_titanic.csv"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

#df = pd.read_csv("../input/train.csv")

df_train = pd.read_csv("../input/train.csv")

print(df_train.shape)

df_test = pd.read_csv("../input/test.csv")

print(df_test.shape)

df = pd.concat([df_train, df_test])

df = pd.get_dummies(df)._get_numeric_data()

df = df.fillna(0)

df_train = df[0:df_train.shape[0]]

df_test = df[df_train.shape[0]:]



y_train = np.array(df_train.Survived).astype(int)

X_train = np.array(df_train.drop('Survived', 1))

X_test = np.array(df_test.drop('Survived', 1))

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)

#clf = svm.SVC().fit(X_train, y_train)



y_pred = clf.predict(X_test)



submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": y_pred

    })

submission.to_csv('submission_titanic.csv', index=False)