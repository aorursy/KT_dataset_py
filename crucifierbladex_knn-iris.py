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
df=pd.read_csv('/kaggle/input/iris/Iris.csv')

df=df.drop(['Id'],axis=1)

df.head()
df.info()
df['Species'].value_counts()
#ports={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}

#df['Species']=df['Species'].map(ports)

#df.head()
x=df.drop(['Species'],axis=1)

y=df['Species'].values
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier

model=KNeighborsClassifier(n_neighbors=3)

x_train=scaler.fit_transform(x_train)

y_train=scaler.fit_transform(y_train)

model.fit(x_train,x_test)

y_pred=model.predict(y_train)

from sklearn import metrics 

metrics.accuracy_score(y_test,y_pred)
print(metrics.classification_report(y_test,y_pred))
metrics.confusion_matrix(y_test,y_pred)
df.head()
data={'SepalLengthCm':3.4,'SepalWidthCm':3.7,'PetalLengthCm':1.9,'PetalWidthCm':0.9}

_=pd.DataFrame(data,index=[0])
model.predict(_)