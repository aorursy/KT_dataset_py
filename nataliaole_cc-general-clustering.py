import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



import warnings
warnings.filterwarnings('ignore')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
sns.set(font_scale=1.0)


raw_data = pd.read_csv('../input/creditcarddata/CC GENERAL.csv')
raw_data.head()

raw_data.info()
raw_data.describe().T
nulls_summary = pd.DataFrame(raw_data.isnull().any(), columns=['Nulls'])   
nulls_summary['Num_of_nulls [qty]'] = pd.DataFrame(raw_data.isnull().sum())   
nulls_summary['Num_of_nulls [%]'] = round((raw_data.isnull().mean()*100),2)   
print(nulls_summary) 


raw_data.dropna(axis=0, subset=['CREDIT_LIMIT'], inplace=True)
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



X= raw_data.iloc[:, [3,13]].values
#Visualise data points
plt.scatter(X[:,0],X[:,1],c='black')
plt.xlabel('PURCHASES')
plt.ylabel('CREDIT_LIMIT')
plt.show()

kmeans5 = KMeans(n_clusters=5)
y_kmeans5 = kmeans5.fit_predict(X)
print(y_kmeans5)

kmeans5.cluster_centers_
Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(X)
    kmeans.fit(X)
    Error.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()
kmeansmodel = KMeans(n_clusters= 4)
y_kmeans= kmeansmodel.fit_predict(X)
kmeans.cluster_centers_
plt.scatter(X[:, 0], X[:, 1], s = 300, c = y_kmeans, cmap='rainbow')