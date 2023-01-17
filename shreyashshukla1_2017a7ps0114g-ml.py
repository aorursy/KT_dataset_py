# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split 

from sklearn.neighbors import KNeighborsRegressor

#from sklearn import metrics

from sklearn.metrics import mean_squared_error

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
nd = pd.read_csv("/kaggle/input/bits-f464-l1/train.csv", sep=',')

df1=nd.drop("id",axis=1)

#dfx=df1.drop("label",axis=1)

df2=df1.drop(["label","b13","b68","b40","b17","b58","b82","b39"], axis = 1)

df3=df2.drop(["b44","b54","b72","b86","b46","b74","b36","b10"], axis = 1)

df=df3.drop(["b12","b26","b61","b81"], axis = 1)

lab=nd['label']

df.head()
X_train, X_test, y_train, y_test = train_test_split(df,lab,test_size = 0.3, random_state = 1) 
#krange=range(1,7)

#scores={}

#score=[]

#for k in krange:

 #   knn=KNeighborsRegressor(n_neighbors=k)

  #  knn.fit(X_train, y_train)

   # y_pred=knn.predict(X_test)

    #scores[k]=np.sqrt(mean_squared_error(y_test,y_pred))

    #score.append(scores[k])
from sklearn.ensemble import RandomForestRegressor 

regressor = RandomForestRegressor(n_estimators = 40, random_state = 0) 

regressor.fit(X_train, y_train)

y_pred=regressor.predict(X_test)

scorea=np.sqrt(mean_squared_error(y_test,y_pred))

print(scorea)
nd1 = pd.read_csv("/kaggle/input/bits-f464-l1/test.csv", sep=',')

df11=nd1.drop("id",axis=1)

df21=df11.drop(["b13","b68","b40","b17","b58","b82","b39"], axis = 1)

df31=df21.drop(["b44","b54","b72","b86","b46","b74","b36","b10"], axis = 1)

dfa=df31.drop(["b12","b26","b61","b81"], axis = 1)

#lab=nd['label']

dfa.head()
y=regressor.predict(dfa)

dfb = pd.DataFrame(data=y, columns=["label"] )

dfb.tail(20)
dfb.insert(0, "id", 0)

#dfc.append(dfb)

dfb.head()

#dfc.to_csv('mysub.csv',index=False)
for i in range(len(y)):

    dfb['id'][i]=1+i

dfb.head()    
dfb.to_csv('mycsv5.csv',index=False)
#%matplotlib inline

#import matplotlib.pyplot as plt

#plt.plot(krange,score)