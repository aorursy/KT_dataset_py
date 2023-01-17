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
data=pd.read_csv("../input/train.csv")

data.head()
data.describe()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data.corr()
cormatrix=data.corr()
cormatrix.max()
target=data["TARGET"]

for col in data.columns:

    column=data[col]

    temp=pd.concat([target,column], axis=1)

    print(temp.corr())
type(data["TARGET"])
data.shape
from sklearn.decomposition import PCA

pca=PCA(n_components=50)

x=data.iloc[:,1:370]

y=data.iloc[:,370]

X=pca.fit_transform(x)

xfrm=pd.DataFrame(X)

type(xfrm)
y.describe()
xfrm.head()
from xgboost import XGBClassifier

cf=XGBClassifier()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(xfrm, y, test_size=0.25, random_state=4200)

from sklearn.model_selection import cross_val_score

accuries = cross_val_score(estimator=cf,X=X_train, y = y_train,cv=10)

print(accuries)
test=pd.read_csv("../input/test.csv")

test.head()
test.shape
tx=data.iloc[:,1:370]

TX=pca.transform(tx)

TXFRM=pd.DataFrame(TX)

TXFRM.head()
cf.fit(X_train,y_train)

ypred=cf.predict(TXFRM)

print(ypred)
len(ypred)
zerocount=0

for k in ypred:

    if k==0:

        zerocount=zerocount+1

print(zerocount)

print(len(ypred)-zerocount)
ypred=cf.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
from imblearn.over_sampling import SMOTE



# Resample the minority class. You can change the strategy to 'auto' if you are not sure.

sm = SMOTE(sampling_strategy='minority', random_state=15)



# Fit the model to generate the data.

oversampled_trainX, oversampled_trainY = sm.fit_sample(xfrm,y)
X_train, X_test, y_train, y_test = train_test_split(oversampled_trainX, oversampled_trainY, test_size=0.25, random_state=4200)

cf=XGBClassifier()

cf.fit(X_train,y_train)

ypred=cf.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
ypred=cf.predict(TXFRM.values)

print(ypred)
zerocount=0

for k in ypred:

    if k==0:

        zerocount=zerocount+1

print(zerocount)

print(len(ypred)-zerocount)