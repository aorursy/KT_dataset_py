# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing as prep

from sklearn.cluster import KMeans

import plotly.express as px

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")

df.head()
df.describe()
df.info()
df.corr()
import seaborn as sns

import matplotlib.pyplot as plt

heatmap=sns.heatmap(df.corr(),vmax=1, vmin=-1,annot=True,cmap="BrBG")

plt.figure(figsize=(16, 6))
df.tail()
df.columns
df=df.rename(columns={'Spending Score (1-100)':"score",'Annual Income (k$)':"annualinc"})
df=df.drop(["CustomerID"],axis=1)

df.head()
px.histogram(df,x="score")

df.Gender.value_counts().plot(kind="bar")
px.histogram(df,x="annualinc",color="score")
sns.scatterplot(x="annualinc",y="score",hue="Gender",data=df)
sns.regplot(x="annualinc",y="score",data=df)
sns.scatterplot(x="Age",y="score",hue="Gender",data=df)
sns.regplot(x="Age",y="score",data=df)
df.head()
new_df= prep.StandardScaler().fit_transform(df.iloc[:,1:])

new_df = pd.DataFrame(new_df, columns=[ 'Age', 'annualinc', 'score'])
new_df.head()
X1=new_df.iloc[:,[1,2]].values

X1
Error =[]

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i).fit(X1)

    kmeans.fit(X1)

    Error.append(kmeans.inertia_)



plt.plot(range(1, 11), Error)

plt.title('Elbow method')

plt.xlabel('No of clusters')

plt.ylabel('Error')

plt.show()



kmeans1=KMeans(n_clusters = 5).fit(X1)

y_pred1=kmeans1.fit_predict(X1)

print(y_pred1)
plt.scatter(X1[:,0],X1[:,1],c=y_pred1,cmap="rainbow")

f1=kmeans1.cluster_centers_

plt.scatter(X1[:,0],X1[:,1],c=y_pred1,cmap="rainbow")

plt.scatter(x = f1[: , 0] , y =  f1[: , 1] , s = 300 , c = 'red' , alpha = 0.5)

plt.xlabel("annual income")

plt.ylabel("Score")

plt.figure(figsize=(16,6))

plt.show()
X2=new_df.iloc[:,[0,2]].values

X2
Error =[]

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i).fit(X2)

    kmeans.fit(X2)

    Error.append(kmeans.inertia_)



plt.plot(range(1 , 11) , Error , 'x')

plt.plot(range(1 , 11) , Error , '-' , alpha = 0.5)



plt.title('Elbow method')

plt.xlabel('No of clusters')

plt.ylabel('Error')

plt.show()

kmeans2=KMeans(n_clusters = 5).fit(X2)

y_pred2=kmeans2.fit_predict(X2)

print(y_pred2)
f2=kmeans2.cluster_centers_

plt.scatter(X2[:,0],X2[:,1],c=y_pred2,cmap="rainbow")

plt.scatter(x = f2[: , 0] , y =  f2[: , 1] , s = 300 , c = 'red' , alpha = 0.5)

plt.xlabel("Age")

plt.ylabel("Score")

plt.figure(figsize=(16,6))

plt.show()
X3=new_df.iloc[:,:].values

X3
Error =[]

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i).fit(X3)

    kmeans.fit(X3)

    Error.append(kmeans.inertia_)



plt.plot(range(1 , 11) , Error , 'x')

plt.plot(range(1 , 11) , Error , '-' , alpha = 0.5)



plt.title('Elbow method')

plt.xlabel('No of clusters')

plt.ylabel('Error')

plt.show()
kmeans3=KMeans(n_clusters = 6).fit(X3)

y_pred3=kmeans3.fit_predict(X3)

print(y_pred3)
f3=kmeans3.cluster_centers_

f3


from mpl_toolkits import mplot3d

ax = plt.axes(projection='3d')





# Data for three-dimensional scattered points

zdata = X3[:,2]

xdata = X3[:,1]

ydata = X3[:,0]

ax.scatter3D(xdata, ydata, zdata, c=y_pred3, cmap='rainbow');





ax.set_xlabel("Annual income")

ax.set_ylabel("Age")

ax.set_zlabel("Score")
