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
df=pd.read_csv("/kaggle/input/1056lab-fraud-detection-in-credit-card/train.csv",index_col=0)
df
from imblearn.over_sampling import SMOTE
smote=SMOTE(kind='svm')
X=df.drop('Class',axis=1).values

y=df['Class'].values
Xs,ys=smote.fit_sample(X,y)
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=100,random_state=72)
rfc.fit(Xs,ys)
dft=pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/test.csv',index_col=0)
dft
Xt=dft.values
predict = rfc.predict_proba(Xt)[:, 1]



submit = pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/sampleSubmission.csv')

submit['Class'] = predict

submit.to_csv('submission1.csv', index=False)