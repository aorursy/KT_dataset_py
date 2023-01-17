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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style='white', color_codes=True)

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



iris = pd.read_csv('../input/Iris.csv')

iris.head()
iris.describe()
sns.pairplot(iris.drop('Id', axis=1), hue='Species',diag_kind='kde')
print(iris.Species.value_counts())
# Just use the first two Species--the first 100 rows

# And the slice of DataFrame（inclusive） is different with List（exclusive）

iris = iris.loc[:99 ,:]
from sklearn import linear_model

from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder



X = iris.loc[:,'SepalLengthCm':'PetalWidthCm']



le = LabelEncoder()

le.fit(iris.Species)

y = le.transform(iris.Species)





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

clf = linear_model.LinearRegression()

clf.fit(X_train, y_train)

pred = clf.predict(X_test)
for p, y in zip(pred, y_test):

    print(p,y)
acc = clf.score(X_train, y_train)



print ('The accuracy: {}'.format(acc))

acc = clf.score(X_test, y_test)



print ('The accuracy: {}'.format(acc))