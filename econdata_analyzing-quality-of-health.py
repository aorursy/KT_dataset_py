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
data=pd.read_csv('../input/health-quality-dataset/quality.csv')
data.head()
data.info()
#data.isnull().sum()
data.describe()
data.head()
data.head(20)
data_train=data['MemberID']<=100
data_test=data['MemberID']>100
X_train=data.drop(['MemberID'],axis=1)
X_train=X_train.drop(['PoorCare'],axis=1)
X_train
X_train['DaysSinceLastERVisit']=X_train['DaysSinceLastERVisit']/(X_train['DaysSinceLastERVisit'].mean())
X_train['ClaimLines']=X_train['ClaimLines']/(X_train['ClaimLines'].mean())
X_train['StartedOnCombination']=X_train['StartedOnCombination'].astype(int)
Y_train=data['PoorCare']
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(C=1e5)
logreg.fit(X_train,Y_train)
logreg.coef_
Xm=['StartedOnCombination','ProviderCount']

Xm=pd.DataFrame(X_train[Xm])
Xm
model=LogisticRegression(C=1e5)
model.fit(Xm,Y_train)
model.coef_