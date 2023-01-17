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
df=pd.read_csv("../input/ChurnData.csv")
df.dropna().head()
df["churn"]=df["churn"].astype(int)
df.head()
X=np.asarray(df[['age','income','tenure']])
y=np.asarray(df["churn"])
from sklearn import preprocessing
X=preprocessing.StandardScaler().fit(X).transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression(C=0.1,solver='liblinear').fit(X_train,y_train)
yhat=LR.predict(X_test)
yhat
yhat_prob=LR.predict_proba(X_test)
yhat_prob
from sklearn.metrics import jaccard_similarity_score as jcs
score=jcs(yhat,y_test)
print(score)