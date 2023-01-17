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
df=pd.read_csv("/kaggle/input/1056lab-diabetes-diagnosis/train.csv",index_col=0)
df
df['Diabetes'].value_counts()
df['Gender']=df['Gender'].map({'male':0,'female':1})
from imblearn.over_sampling import SMOTE
smote=SMOTE()
X=df.drop('Diabetes',axis=1).values

y=df['Diabetes'].values
Xs,ys=smote.fit_sample(X,y)
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=100,random_state=72)
rfc.fit(Xs,ys)
dft=pd.read_csv('/kaggle/input/1056lab-diabetes-diagnosis/test.csv',index_col=0)
dft['Gender']=dft['Gender'].map({'male':0,'female':1})
Xt=dft.values
predict = rfc.predict_proba(Xt)[:, 1]



submit = pd.read_csv('/kaggle/input/1056lab-diabetes-diagnosis/sampleSubmission.csv')

submit['Diabetes'] = predict

submit.to_csv('submission1.csv', index=False)