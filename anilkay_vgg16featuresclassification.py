# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

datalar = pd.read_csv("../input/datalar.csv")
datalar.head()
import seaborn as sns

sns.relplot(data=datalar,x="f1",y="f2",hue="labels")
x=datalar.iloc[:,0:2]

y=datalar.iloc[:,2:]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

from tpot import TPOTClassifier

tpot = TPOTClassifier(verbosity=2,max_time_mins=120)

tpot.fit(x_train, y_train)

print(tpot.score(x_test, y_test))
import sklearn.metrics as metrik

ypred=tpot.predict(x_test)

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=9)

knn.fit(x_train,y_train)

ypred=knn.predict(x_test)

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))