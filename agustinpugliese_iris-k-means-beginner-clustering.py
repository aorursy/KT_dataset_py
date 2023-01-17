import numpy as np    
import pandas as pd 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
import seaborn as sns
sns.set()

%matplotlib inline

from sklearn.cluster import KMeans 
from sklearn import preprocessing 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')

print(data.head())
print('--' * 40)
print(data.describe())
print(pd.isnull(data).sum()) # Missing values?
data.groupby('species').size()
data.groupby('species').mean()
correlacion = sns.pairplot(data, hue = "species", palette = "husl", corner = True)
data2 = data.copy()
x = data2.iloc[:,[0,1]]
x.head()
plt.scatter(x['sepal_length'], x['sepal_width'],s = 20, c='red', alpha = 0.5)
plt.xlabel('Sepal Lenght', fontsize = 15, c = 'black')
plt.ylabel('Sepal Width', fontsize = 15, c = 'black')
plt.title('Sepal Lenght vs Sepal Width', fontsize = 20 , c = 'Black')
plt.show()
x_scaled = preprocessing.scale(x)
kmeans = KMeans(4) 
kmeans.fit(x_scaled)
# Cluster variable creation.
Clusters = kmeans.fit_predict(x_scaled) 
# Checking the result
Clusters


data_con_clusters = data.copy()
data_con_clusters['Cluster'] = Clusters 
data_con_clusters 
plt.scatter(data_con_clusters['sepal_length'], data_con_clusters['sepal_width'],s = 20,
            c = data_con_clusters['Cluster'], cmap = 'rainbow', alpha = 1) #Cada cluster cada color!
plt.title('Clustering Iris Sepal length vs. Sepal Width', fontsize = 20)
plt.xlabel('Sepal length', fontsize = 15) #EJES
plt.ylabel('Sepal Width', fontsize = 15)
plt.show()
kmeans.inertia_

wcss=[]

for i in range (1,10): 
    kmeans=KMeans(i)
    kmeans.fit(x)
    wcss_iter=kmeans.inertia_
    wcss.append(wcss_iter)
    
    number_clusters=range(1,10)
    
plt.plot(number_clusters,wcss) 
plt.show()
kmeans = KMeans(3) 
kmeans.fit(x_scaled)
Clusters = kmeans.fit_predict(x_scaled) 
Clusters
data_con_clusters = data.copy()
data_con_clusters['Cluster'] = Clusters 
data_con_clusters 
plt.scatter(x.values[Clusters == 0, 0], x.values[Clusters == 0, 1], s = 20, c = 'red', label = 'Setosa')
plt.scatter(x.values[Clusters == 1, 0], x.values[Clusters == 1, 1], s = 20, c = 'blue', label = 'Virginica')
plt.scatter(x.values[Clusters == 2, 0], x.values[Clusters == 2, 1], s = 20, c = 'green', label = 'Versicolor')

plt.title('Clustering Iris Sepal length vs. Sepal Width', fontsize = 20)
plt.xlabel('Sepal length', fontsize = 15) 
plt.ylabel('Sepal Width', fontsize = 15)
plt.legend()
plt.show()