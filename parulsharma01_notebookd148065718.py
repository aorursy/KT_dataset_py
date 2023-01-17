# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
data
data.head()
data.isnull().sum()
data.describe()
data.shape
data['Class']
fraud=data.loc[data['Class']==1]

normal=data.loc[data['Class']==0]
len(normal)
len(fraud)
sns.relplot(x='Amount',y="Time",hue="Class",data=data)
from sklearn import linear_model

from sklearn.model_selection import train_test_split
x=data.iloc[:,:-1]

y=data["Class"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35)
clf=linear_model.LogisticRegression(C=1e5)
clf.fit(x_train,y_train)
y_pred=np.array(clf.predict(x_test))
y=np.array(y_test)

from sklearn.metrics import  confusion_matrix, classification_report,accuracy_score
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))