# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/1056lab-brain-cancer-classification/train.csv",index_col=0)
df
df.isnull().any()
df['type'].value_counts()
df.type=df.type.map({'ependymoma':1,'glioblastoma':2,'medulloblastoma':3,'pilocytic_astrocytoma':4,'normal':0})
X=df.drop(['type'],axis=1).values

y=df.type.values
from sklearn.feature_selection import RFE

#from sklearn.feature_selection import SelectKBest

#fs1 = SelectKBest(k=50)

#fs1.fit(X, y)

#X_ = fs1.transform(X)

from sklearn.ensemble import RandomForestClassifier

est = RandomForestClassifier(n_estimators=10,random_state=72)

fs1  = RFE(est, n_features_to_select=30)

fs1.fit(X, y)

X_  = fs1.transform(X)
from sklearn.decomposition import TruncatedSVD

fs2 = TruncatedSVD(n_components=10)

fs2.fit(X_)

X_2 = fs2.transform(X_)
rfc=RandomForestClassifier(n_estimators=30,random_state=72)
rfc.fit(X_2,y)
dft=pd.read_csv("/kaggle/input/1056lab-brain-cancer-classification/test.csv",index_col=0)
Xt=dft.values

Xt_=fs1.transform(Xt)

Xt_2=fs2.transform(Xt_)
predict=rfc.predict(Xt_2)
submit = pd.read_csv('/kaggle/input/1056lab-brain-cancer-classification/sampleSubmission.csv')

submit['type'] = predict

submit.to_csv('submission.csv', index=False)