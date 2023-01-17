import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/Classified Data')
df.head() #checking the head of our data
df.info()
df.drop(['Unnamed: 0'],axis = 1,inplace = True) #dropping the column
df.isnull().sum() #checking for null values
plt.figure(figsize = (10,10)) #getting a visual of our data

sns.heatmap(df.corr(),annot = True)
sns.scatterplot(x='WTT',y='PTI',data = df)
sns.scatterplot(x='EQW',y='SBI',data = df)
sns.scatterplot(x='LQE',y='QWG',data = df)
sns.scatterplot(x='FDJ',y='PJF',data = df)
sns.scatterplot(x='HQE',y='NXJ',data = df)
from sklearn.model_selection import train_test_split #importing our libraries

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report,confusion_matrix
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis = 1)) #standard sccaling our data for better results
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis = 1))
df_feat = pd.DataFrame(scaled_features,columns = df.columns[:-1])
df_feat.head()
x = df_feat

y = df['TARGET CLASS']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 101)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
pred = knn.predict(x_test)
print(classification_report(y_test,pred))

print('\n')

print(confusion_matrix(y_test,pred))
error = []

for i in  range(1,100):

    knn = KNeighborsClassifier(n_neighbors= i)

    knn.fit(x_train,y_train)

    pred_i = knn.predict(x_test)

    error.append(np.mean(pred_i != y_test))
plt.figure(figsize = (10,6))

plt.plot(range(1,100),error)

plt.title('K-values')

plt.xlabel('K')

plt.ylabel('Error')
knn = KNeighborsClassifier(n_neighbors=40)

knn.fit(x_train,y_train)

pred1 = knn.predict(x_test)
print(classification_report(y_test,pred1))

print('\n')

print(confusion_matrix(y_test,pred1))