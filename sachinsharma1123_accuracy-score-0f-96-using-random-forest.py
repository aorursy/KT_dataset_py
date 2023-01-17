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
df=pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
df
df.info
df.isnull().sum()
df=df.drop(['country','agent','company'],axis=1)
df=df.dropna()
df.isnull().sum()
df['hotel'].unique()
df['deposit_type'].unique()
df['arrival_date_month'].unique()
#lets find the categorialfeatures

list_1=list(df.columns)
list_cate=[]

for i in list_1:

    if df[i].dtype=='object':

        list_cate.append(i)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
for i in list_cate:

    df[i]=le.fit_transform(df[i])
df
y=df['hotel']

x=df.drop('hotel',axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_1=accuracy_score(y_test,pred_1)
score_1
from sklearn.neighbors import KNeighborsClassifier

list_1=[]

for i in range(1,11):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    pred_y=knn.predict(x_test)

    scores=accuracy_score(y_test,pred_y)

    list_1.append(scores)
import matplotlib.pyplot as plt

plt.plot(range(1,11),list_1)

plt.xlabel('k values')

plt.ylabel('accuracy scores')

plt.show()
print(max(list_1

         ))
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)
pred_2=rfc.predict(x_test)

score_2=accuracy_score(y_test,pred_2)
score_2
#so from all the classifiers random forest gives the best accuracy score