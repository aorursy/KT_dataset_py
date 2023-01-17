# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pa # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pa.read_csv('../input/Mall_Customers.csv')
df['Gender_code']=df['Gender'].apply(lambda a:1 if a=='Male' else 0) 
df2= df[['Age','Annual Income (k$)','Spending Score (1-100)','Gender_code']]
df2.isnull().values.any()

df2.head()
import seaborn as sns

sns.lineplot(data=df2,x ='Age',y='Annual Income (k$)')

sns.lineplot(data=df2,x ='Age',y='Spending Score (1-100)')

#sns.scatterplot(data=df3)
sns.scatterplot(data=df2,y='Spending Score (1-100)',x='Annual Income (k$)',hue='Gender_code')
from sklearn.cluster import KMeans
df3=df2[['Annual Income (k$)', 'Spending Score (1-100)', 'Gender_code']]
from sklearn.preprocessing import MinMaxScaler

mms= MinMaxScaler()

df3=mms.fit_transform(df3)


inertia=[]

for i in range(2,10):

    model = KMeans(n_clusters=i)

    model.fit(df2)

    inertia.append(model.inertia_)

    
import matplotlib.pyplot as plt 

plt.figure(figsize=(16,6))

plt.plot(np.arange(2,10),inertia)
model = KMeans(n_clusters=5)

model.fit(df2)
print(model.labels_)

print(model.cluster_centers_)