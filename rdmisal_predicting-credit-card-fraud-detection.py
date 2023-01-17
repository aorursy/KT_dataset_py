import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

data=pd.read_csv('../input/creditcard.csv')

data.head()
corr=np.array(data.corr())

corr=np.around(corr[-1],decimals=2)

corr=pd.DataFrame(corr,index=data.columns)

corr
x=data[['V4','V3','V10','V12','V14','V16','V17']]

y=data[['Class']]

from sklearn.model_selection import train_test_split

xtr,xte,ytr,yte=train_test_split(x,y,test_size =0.33)
from sklearn.naive_bayes import BernoulliNB

model=BernoulliNB()

model.fit(xtr,ytr)
predict=model.predict(xte)

from sklearn.metrics import accuracy_score, confusion_matrix

accuracy_score(yte,predict)
confusion_matrix(yte,predict)