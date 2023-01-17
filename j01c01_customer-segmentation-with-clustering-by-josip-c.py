import pandas as pd
import numpy as np
import math
import xlrd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
data=pd.read_csv("../input/CC GENE.csv")
data.head()
data.shape
data.describe()
data.isnull().sum()
data.MINIMUM_PAYMENTS=data.MINIMUM_PAYMENTS.fillna(data.MINIMUM_PAYMENTS.mean())
data.CREDIT_LIMIT=data.CREDIT_LIMIT.fillna(data.CREDIT_LIMIT.mean())
data.isnull().sum()
data.drop("CUST_ID", axis=1, inplace=True)
data.shape
plt.figure(figsize=(9,7))
sns.heatmap(data.corr(),cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
#sns.pairplot(data)
#plt.show()
wcss = []
K = range(1,30)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(data)
    wcss.append(kmeanModel.inertia_)
plt.figure(figsize=(16,8))
plt.plot(K, wcss, 'bx-')
plt.xlabel('Number of k')
plt.ylabel('WCSS')
plt.title('The Elbow Method showing the optimal number of k')
plt.show()
Kmeans=KMeans(n_clusters=8)
Kmeans.fit(data)
y_Kmeans=Kmeans.predict(data)
data["Cluster"] = y_Kmeans
data_model=["BALANCE", "PURCHASES", "CASH_ADVANCE","CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "TENURE"]
data["Cluster"] = y_Kmeans
data_model.append("Cluster")
data[data_model].head()
plt.figure(figsize=(25,25))
sns.pairplot( data[data_model], hue="Cluster")