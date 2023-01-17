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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



df = pd.read_csv('../input/wwwkagglecomrishikesh247/income.csv')

df.head()

df.isnull()

df.isnull().sum()
dfnew = df.drop(['Name'], axis=1)

dfnew.head()
plt.scatter(dfnew['Age'], dfnew['Income($)'])
from sklearn.cluster import KMeans

km = KMeans(n_clusters = 3)

predict = km.fit_predict(dfnew[['Age','Income($)']])

dfnew['cluster'] = predict

dfnew.head()
df1 = dfnew[dfnew.cluster==0]

df2 = dfnew[dfnew.cluster==1]

df3 = dfnew[dfnew.cluster==2]

plt.scatter(df1.Age,df1['Income($)'],color='green')

plt.scatter(df2.Age,df2['Income($)'],color='red')

plt.scatter(df3.Age,df3['Income($)'],color='black')

plt.xlabel('Age')

plt.ylabel('Income ($)')

plt.legend()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



scaler.fit(dfnew[['Income($)']])

dfnew['Income($)'] = scaler.transform(dfnew[['Income($)']])



scaler.fit(dfnew[['Age']])

dfnew['Age'] = scaler.transform(dfnew[['Age']])

plt.scatter(dfnew.Age,dfnew['Income($)'])



km = KMeans(n_clusters=3)

y_predicted = km.fit_predict(dfnew[['Age','Income($)']])

dfnew['cluster']=y_predicted

dfnew.head()
km.cluster_centers_
df1 = dfnew[dfnew.cluster==0]

df2 = dfnew[dfnew.cluster==1]

df3 = dfnew[dfnew.cluster==2]

plt.scatter(df1.Age,df1['Income($)'],color='green')

plt.scatter(df2.Age,df2['Income($)'],color='red')

plt.scatter(df3.Age,df3['Income($)'],color='black')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

plt.legend()
#ELBow gragh

sse = []

k_rng = range(1,10)

for k in k_rng:

    km = KMeans(n_clusters=k)

    km.fit(dfnew[['Age','Income($)']])

    sse.append(km.inertia_)



plt.xlabel('K')

plt.ylabel('Sum of squared error')

plt.plot(k_rng,sse)