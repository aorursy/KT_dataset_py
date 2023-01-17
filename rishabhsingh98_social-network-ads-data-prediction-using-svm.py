# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/social-network-ads/Social_Network_Ads.csv')
df.head()
df.info()
sex=pd.get_dummies(df['Gender'], drop_first=True)
sex.head()
df=pd.concat([sex,df],axis=1)
df.drop('User ID', inplace=True,axis=1)
sns.heatmap(df.corr(), cmap='coolwarm')
df['Purchased'].value_counts()
df.drop(['Male','Gender'],axis=1, inplace=True)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
scaled_array=ss.fit_transform(df.drop('Purchased',axis=1))
scaled_array.shape
x=pd.DataFrame(data=scaled_array, columns=df.columns[:-1])
y = df['Purchased']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
from sklearn.svm import SVC
model = SVC()
model.fit(x_train,y_train)
ypred=model.predict(x_test)
ypred
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
confusion_matrix(y_test,ypred)
print('Accuracy score :', accuracy_score(y_test,ypred))