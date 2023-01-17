# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.externals import joblib

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score,classification_report



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')
df.head()
dt = DecisionTreeClassifier(random_state=1)

rf = RandomForestClassifier(random_state=1)

lg = LogisticRegression()

et = ExtraTreesClassifier(random_state=1)

knn = KNeighborsClassifier(n_neighbors=13)
df = df.drop(['int_memory','touch_screen','wifi'],axis=1)
x = df.drop('price_range',axis=1)

y = df['price_range']

x = x.astype('int')

y = y.astype('int')
score = cross_val_score(knn,x,y,cv=4)

score.mean()
knn.fit(x,y)
joblib.dump(knn,'knn_joblib')
pred = knn.predict(x)
accuracy_score(y,pred)
print(classification_report(pred,y))