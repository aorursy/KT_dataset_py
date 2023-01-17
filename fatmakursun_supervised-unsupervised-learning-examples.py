import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 



import os



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/USA_Housing.csv')
df.head()   
df.info() 
df.describe() 
sns.distplot(df['Price']) 
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')  
x=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',

       'Avg. Area Number of Bedrooms', 'Area Population']]

y=df['Price']


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=101) 
from sklearn.linear_model import LinearRegression

lm=LinearRegression() 

lm.fit(x_train,y_train) 
print(lm.intercept_) #bu bizim formülümüzdeki b0
cdf=pd.DataFrame(lm.coef_,x.columns,columns=['Coeff']) 

cdf

predictions=lm.predict(x_test) 
plt.figure(figsize=(16,8))

plt.scatter(y_test,predictions)

plt.show()

plt.figure(figsize=(16,8))

sns.distplot((y_test-predictions),bins=50)

plt.show()
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
from sklearn.datasets import make_blobs
data = make_blobs(n_samples=200, n_features=2, 

                           centers=4, cluster_std=1.8,random_state=101)  
plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow') 
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)

kmeans.fit(data[0])
kmeans.cluster_centers_  
kmeans.labels_
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))

ax1.set_title('K Means')

ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')

ax2.set_title("Original")

ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')   