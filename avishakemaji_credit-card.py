# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize']=14,8
RANDOM_SEED=42
LABELS=["Normal","Fraud"]
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.head()
df.info()
columns=df.columns.tolist()
columns=[c for c in columns if c not in ["Class"]]
target="Class"
state=np.random.RandomState(42)
X=df.drop('Class',axis=1)
Y=df['Class']
X_outlier=state.uniform(low=0,high=1,size=(X.shape[0],X.shape[1]))

df.isnull().values.any()
plt.figure(figsize=(16,9))
sns.countplot(df['Class'])
fraud=df[df['Class']==1]
normal=df[df['Class']!=1]
print(fraud.shape[0])
print(normal.shape[0])
from imblearn.under_sampling import NearMiss
nm=NearMiss()
X_res,y_res=nm.fit_sample(X,Y)
X_res.shape,y_res.shape
from collections import Counter
Counter(Y),Counter(y_res)
from imblearn.combine import SMOTETomek
smk=SMOTETomek()
X_res,y_res=smk.fit_sample(X,Y)
X_res.shape,y_res.shape
from collections import Counter
Counter(Y),Counter(y_res)
from imblearn.over_sampling import RandomOverSampler
os=RandomOverSampler(1)
X_train,y_train=os.fit_sample(X,Y)
X_train.shape,y_train.shape
from collections import Counter
Counter(Y),Counter(y_train)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.3,random_state=1)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
from sklearn.metrics import accuracy_score
print(round(accuracy_score(y_test,y_pred)*100,2),"%")