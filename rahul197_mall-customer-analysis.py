# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

%config InlineBackend.print_figure_kwargs = {'bbox_inches':None}



data=pd.read_csv("../input/Mall_Customers.csv")
data.head()
data.shape
data.info()
data.describe()
data.isnull().any()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data['Gender']=le.fit_transform(data['Gender'])
import matplotlib.pyplot as plt

import seaborn as sns

import shap
sns.set(style="white", palette="PuBuGn_d", color_codes=True)

sns.countplot('Gender',data=data,palette='winter')

size=data['Gender'].value_counts()

print('Female :',size[0]/(size[0]+size[1])*100)

print('Male :',size[1]/(size[0]+size[1])*100)

plt.title("Gender distirbution")
plt.figure(1 , figsize = (15 ,6))

n = 0 

color=['red','green','blue']

count=0

for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    n += 1

    plt.subplot(1 , 3 , n)

    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)

    sns.distplot(data[x] , color=color[count])

    plt.title('Distplot of {}'.format(x))

    count+=1

plt.show()
sns.pairplot(data)

plt.plot()
plt.rcParams['figure.figsize'] = (18, 8)

corr=data.corr()

sns.heatmap(corr)

plt.title("Data correleation", fontsize=14)

plt.plot()
plt.rcParams['figure.figsize'] = (18, 6)

sns.violinplot(data['Gender'], data['Spending Score (1-100)'], palette = 'pastel')

plt.title('Gender vs Spending Score', fontsize = 14)

plt.show()
plt.rcParams['figure.figsize'] = (18, 6)

sns.violinplot(data['Age'], data['Spending Score (1-100)'], palette = 'pastel')

plt.title('Age vs Spending Score', fontsize = 14)

plt.show()
plt.rcParams['figure.figsize'] = (18, 6)

sns.violinplot(data['Annual Income (k$)'], data['Spending Score (1-100)'], palette = 'pastel')

plt.title('Gender vs Spending Score', fontsize = 14)

plt.show()
plt.rcParams['figure.figsize'] = (18, 6)

sns.violinplot(data['Gender'], data['Annual Income (k$)'], palette = 'pastel')

plt.title('Gender vs Annual Income (k$)', fontsize = 14)

plt.show()
plt.rcParams['figure.figsize'] = (18, 6)

sns.violinplot(data['Age'], data['Annual Income (k$)'], palette = 'pastel')

plt.title('Gender vs Annual Income (k$)', fontsize = 14)

plt.show()
X=data.iloc[:,:-1]

y=data.iloc[:,-1]
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=10, n_estimators=300)

clf.fit(X,y)
shap_values = shap.TreeExplainer(clf).shap_values(X)
shap.summary_plot(shap_values[0], X)
shap.dependence_plot("Age", shap_values[0], X)
shap.dependence_plot("Gender", shap_values[0], X)
shap.dependence_plot('Annual Income (k$)', shap_values[0], X)

plt.show()
from mpl_toolkits.mplot3d import Axes3D



sns.set_style("white")

fig = plt.figure(figsize=(18,10))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(data['Age'], data["Annual Income (k$)"], data["Spending Score (1-100)"], c='red', s=60)

ax.view_init(30, 185)

plt.xlabel("Age")

plt.ylabel("Annual Income (k$)")

ax.set_zlabel('Spending Score (1-100)')

plt.show()
from sklearn.cluster import KMeans



wcss = []

for k in range(1,11):

    kmeans = KMeans(n_clusters=k, init="k-means++")

    kmeans.fit(data.iloc[:,1:])

    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12,6))    

plt.grid()

plt.plot(range(1,11),wcss, linewidth=2, color="blue", marker ="8")

plt.xlabel("K Value")

plt.xticks(np.arange(1,11,1))

plt.ylabel("WCSS")

plt.show()
km = KMeans(n_clusters=5)

clusters = km.fit_predict(data.iloc[:,1:])

data["label"] = clusters

fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(data.Age[data.label == 0], data["Annual Income (k$)"][data.label == 0], data["Spending Score (1-100)"][data.label == 0], c='blue', s=60)

ax.scatter(data.Age[data.label == 1], data["Annual Income (k$)"][data.label == 1], data["Spending Score (1-100)"][data.label == 1], c='red', s=60)

ax.scatter(data.Age[data.label == 2], data["Annual Income (k$)"][data.label == 2], data["Spending Score (1-100)"][data.label == 2], c='green', s=60)

ax.scatter(data.Age[data.label == 3], data["Annual Income (k$)"][data.label == 3], data["Spending Score (1-100)"][data.label == 3], c='orange', s=60)

ax.scatter(data.Age[data.label == 4], data["Annual Income (k$)"][data.label == 4], data["Spending Score (1-100)"][data.label == 4], c='purple', s=60)

ax.view_init(30, 185)

plt.xlabel("Age")

plt.ylabel("Annual Income (k$)")

ax.set_zlabel('Spending Score (1-100)')

plt.show()
data['Spending Score (1-100)']=data['Spending Score (1-100)'].astype(float)
import scipy.cluster.hierarchy as sch

dendogram=sch.dendrogram(sch.linkage(data,method='ward'))

plt.title('Dendogram', fontsize=20)

plt.xlabel("Customers")

plt.ylabel("Euclidean Distance")

plt.show()
import scipy.cluster.hierarchy as sch

dendogram=sch.dendrogram(sch.linkage(data.iloc[:,3:5],method='ward'))

plt.title('Dendogram', fontsize=20)

plt.xlabel("Customers")

plt.ylabel("Euclidean Distance")

plt.show()
data.head(5)
import scipy.cluster.hierarchy as sch

dendogram=sch.dendrogram(sch.linkage(data.iloc[:,[1,4]],method='ward'))

plt.title('Dendogram', fontsize=20)

plt.xlabel("Customers")

plt.ylabel("Euclidean Distance")

plt.show()
import scipy.cluster.hierarchy as sch

dendogram=sch.dendrogram(sch.linkage(data.iloc[:,[2,4]],method='ward'))

plt.title('Dendogram', fontsize=20)

plt.xlabel("Customers")

plt.ylabel("Euclidean Distance")

plt.show()