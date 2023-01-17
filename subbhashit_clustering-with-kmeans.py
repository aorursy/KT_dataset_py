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
data=pd.read_csv("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
data.head()
from sklearn.preprocessing import LabelEncoder
la=LabelEncoder()
data.Gender=la.fit_transform(data.Gender)
y=data.iloc[:,[3,4]].values
from sklearn.cluster import KMeans
l=[]
for i in range(1,10):
    m=KMeans(n_clusters=i)
    m.fit(y)
    l.append(m.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1,10),l,color='r')
plt.scatter(range(1,10),l,color='g')
model=KMeans(n_clusters=5)
model.fit(y)
a=model.cluster_centers_
yp=model.predict(y)
a
plt.scatter(y[yp==0,0],y[yp==0,1],label='1')
plt.scatter(y[yp==1,0],y[yp==1,1],label='2')
plt.scatter(y[yp==2,0],y[yp==2,1],label='3')
plt.scatter(y[yp==3,0],y[yp==3,1],label='4')
plt.scatter(y[yp==4,0],y[yp==4,1],label='5')
plt.legend()
plt.scatter(a[:,0],a[:,1],marker='x',color='r')