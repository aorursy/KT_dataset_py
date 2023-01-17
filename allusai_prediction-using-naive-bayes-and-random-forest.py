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
data=pd.read_csv('/kaggle/input/wine-quality/winequalityN.csv')
data.head()
data['quality'].value_counts()
data.shape
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
x['type'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x['type']=le.fit_transform(x['type'])
x['type'].value_counts()
null_values=data.isnull().sum()
null_values
for z in x:
 if(null_values[z]>0) : 
     x.loc[(x[z].isnull()),z]=x[z].mean()
x.isnull().sum()
x
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
i=1;
for z in x:
    plt.subplot(4,4,i)
    i=i+1
    plt.boxplot(x[z])
    plt.title(z)
x.corr()
y=pd.DataFrame(y,columns=['quality'])
y.head()
y.info()
y.loc[(y['quality'] <= 5),'quality']=0
y.loc[(y['quality'] <= 7) & (y['quality'] > 5),'quality']=1
y.loc[(y['quality'] >7 ) ,'quality']=2

y['quality'].value_counts()

x
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
x_train.shape
y_train.shape

y_train['quality'].value_counts()
x_test.shape
from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

#training the model
nb.fit(x_train,y_train)


y_pred=nb.predict(x_test)
y_pred
from sklearn import metrics

metrics.accuracy_score(y_test,y_pred)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(max_depth=32,n_estimators=120,random_state=1)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
from sklearn import metrics

metrics.accuracy_score(y_test,y_pred)
rf.predict([[1,2,3,4,5,1,2,3,4,1,2,3]])
