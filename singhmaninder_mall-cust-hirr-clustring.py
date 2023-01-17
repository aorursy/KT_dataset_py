import numpy as np
import pandas as pd
data=pd.read_csv("../input/mall-customer-dataset/Mall_Customers.csv")
data.head()
data.shape
data.info()
data.describe()
data.isna().sum()
import seaborn as sb
sb.countplot(x="Genre",data=data)
data["Annual Income (k$)"].plot(kind="hist")
data["Spending Score (1-100)"].plot(kind="hist")
X=data.iloc[:,[3,4]].values
y=data.iloc[:,[3]].values
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
X=scale.fit_transform(X)
y=scale.fit_transform(y)
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram , linkage
linked=linkage(X,"ward")
plt.figure(figsize=(10,7))
plt.xlabel("Customers")
dendrogram(linked,
           orientation='top',
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()

