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
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

from sklearn.tree import DecisionTreeClassifier

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



X = pd.read_csv("../input/train.csv")

X_test = pd.read_csv("../input/test.csv")

Y = X.pop("Survived")



n = list(X.dtypes[X.dtypes != 'object'].index)

X=X[n]

X_test=X_test[n]



X["Age"].fillna(X.Age.mean(), inplace = True);

X_test["Age"].fillna(X_test.Age.mean(), inplace = True);

X_test["Fare"].fillna(X_test.Fare.mean(), inplace = True);

dtc = DecisionTreeClassifier()

dtc.fit(X,Y)

Y_test = dtc.predict(X_test)

submission = pd.DataFrame({

        "PassengerId": X_test["PassengerId"],

        "Survived": Y_test

    })
submission
submission.to_csv('submission.csv', index=False)