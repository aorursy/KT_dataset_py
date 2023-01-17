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

df=pd.read_csv("../input/advertising/advertising.csv")

df.head()
df.dropna(axis=0)
df.columns
x=df.iloc[:,0:7]

x=x.drop(['Ad Topic Line','City'],axis=1)

x
y=df.iloc[:,9]

y
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=4)

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
from sklearn.linear_model import LogisticRegression

Lr=LogisticRegression(C=0.01,random_state=0)

Lr.fit(x_train,y_train)

y_pred=Lr.predict(x_test)

y_pred
y_pred_proba=Lr.predict_proba(x_test)

y_pred_proba
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred))

from sklearn.metrics import jaccard_similarity_score

print(jaccard_similarity_score(y_test,y_pred))

from sklearn.metrics import f1_score

print(f1_score(y_test,y_pred))
