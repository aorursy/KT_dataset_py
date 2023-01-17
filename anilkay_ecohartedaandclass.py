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
data=pd.read_csv("/kaggle/input/echocardiogram-uci/echocardiogram.csv")

data.head()
del data["name"]

del data["group"]
data.dtypes
from fancyimpute import KNN

deyta=KNN(k=2).fit_transform(data)

deyta.shape
deyta=pd.DataFrame(deyta)

deyta.columns=data.columns
deyta.head()
deyta.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt

correlation=deyta.corr()

plt.figure(figsize=(16, 16))

sns.heatmap(correlation,annot=True)
y=deyta["alive"]

x=deyta.loc[:,(deyta.columns!="alive") & (data.columns!="aliveat1")]
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()    

from sklearn.model_selection import cross_val_score

print(cross_val_score(rfc,x, y, cv=4))
from sklearn.tree import DecisionTreeClassifier

print(cross_val_score(DecisionTreeClassifier(),x, y, cv=4))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=1923)



from tpot import TPOTClassifier

tpot = TPOTClassifier(verbosity=2,max_time_mins=40)

tpot.fit(X_train, y_train)

print(tpot.score(X_test, y_test))
tpot.export("classifierpipeline.py")
ypred=tpot.predict(X_test)
import sklearn.metrics as metrik

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))