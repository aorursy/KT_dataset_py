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
import pandas as pd
import numpy as np
import seaborn as sns
%matplotlib inline
df = pd.read_csv("../input/creditcard.csv")
df_nonFraud = df.loc[df.Class==0]
df_nonFraud.head()
df_nonFraud.drop('Time', axis=1,inplace=True)
df.drop('Time', axis=1,inplace=True)
df_nonFraud.head()
df.head()
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
df_nonFraud = scaler.fit_transform(df_nonFraud)
df_nonFraud.head()
sns.countplot(df_nonFraud.Class)  ## 0 > Non fraud 1> fraud
sns.countplot(df.Class)
x = df_nonFraud.iloc[:,:-1]
y = df_nonFraud.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8, random_state=42,stratify=y)
## contamination 
len(y_train[y_train==1])/len(y_train)
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.svm import OneClassSVM
classifier_ocsvm = OneClassSVM()
classifier_ocsvm.fit(x_train)
acc = accuracy_score(y,y_pred_lof)
print('acc is {}'.format(acc))

cm = confusion_matrix(y,y_pred_lof)
print(cm)

rc = recall_score(y,y_pred_lof)
print('rc is {}'.format(rc*100))