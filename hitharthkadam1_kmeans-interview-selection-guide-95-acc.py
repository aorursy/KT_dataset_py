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
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/kaggle/input/Classified Data')
df.head() #checking the head of our data
df.drop(['Unnamed: 0'],axis = 1,inplace = True) #dropping the column
df.isnull().sum() #checking for null values
#plt.figure(figsize = (10,10)) #getting a visual of our data
#sns.heatmap(df)
sns.scatterplot(x='WTT',y='PTI',data = df)
sns.scatterplot(x='EQW',y='SBI',data = df)
sns.scatterplot(x='LQE',y='QWG',data = df)
sns.scatterplot(x='FDJ',y='PJF',data = df)
sns.scatterplot(x='HQE',y='NXJ',data = df)
from sklearn.model_selection import train_test_split #importing our libraries
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report,confusion_matrix
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis = 1)) #standard scaling our data for better results
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis = 1))
df_feat = pd.DataFrame(scaled_features,columns = df.columns[:-1])
df_feat.head()
x = df_feat
y = df['TARGET CLASS']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 101)

kmeans = KMeans(4)


kmeans.fit(x_train,y_train)

pred = kmeans.fit_predict(x_test)
print(classification_report(y_test,pred))
print('\n')
print(confusion_matrix(y_test,pred))

wcss = []
# 'cl_max' is a that keeps track the highest number of clusters we want to use the WCSS method for.
# Note that 'range' doesn't include the upper boundery
cl_max = 10
for i in range (1,cl_max):
    kmeans= KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)
wcss
plt.figure(figsize = (10,6))
plt.plot(range(1,10),wcss)
plt.title('Cluster values')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
kmeans = KMeans(2)
kmeans.fit(x_train,y_train)
pred1 = kmeans.predict(x_test)
print(classification_report(y_test,pred1))
print('\n')
print(confusion_matrix(y_test,pred1))
