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
import pandas as pd

import numpy

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.metrics import accuracy_score





neigh = KNeighborsClassifier(n_neighbors=3)
df = pd.read_csv('../input/train.csv')
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

y_train = df[['Survived']]

X = X.replace(['male', 'female', 'S', 'C', 'Q'],[0,1,1,2,3])



X = X.as_matrix()

X_train = numpy.nan_to_num(X)
neigh.fit(X_train, y_train)
df_test  = pd.read_csv('../input/test.csv')
X_test = df_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

X_test = X_test.replace(['male', 'female', 'S', 'C', 'Q'],[0,1,1,2,3])



X_test = X_test.as_matrix()

X_test = numpy.nan_to_num(X_test)
neigh.predict(X_test)