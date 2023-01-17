# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

from sklearn.cluster import KMeans

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/wine-pca/Wine.csv")

df.head()
df.info()
import seaborn as sns

import matplotlib.pyplot as plt

heatmap=sns.heatmap(df.corr(),vmax=1, vmin=-1,annot=True,cmap="BrBG")

plt.figure(figsize=(160, 160))
df.corr()
df.Customer_Segment.unique()
sns.regplot(x="Flavanoids",y="Total_Phenols",data=df)

plt.figure(figsize=(16,6))
df.columns


new_df= preprocessing.StandardScaler().fit_transform(df)

new_df = pd.DataFrame(new_df, columns=['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium','Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols','Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline','Customer_Segment'])
new_df.head()
new_df=new_df.drop(["Customer_Segment"],axis=1)

new_df.head()
new_df.head()
new_df.columns
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))



ax1.set_title('Before Scaling')

sns.kdeplot(df['Alcohol'], ax=ax1)

sns.kdeplot(df['Ash_Alcanity'], ax=ax1)

sns.kdeplot(df['Malic_Acid'], ax=ax1)

ax2.set_title('After Standard Scaler')

sns.kdeplot(new_df['Alcohol'], ax=ax2)

sns.kdeplot(new_df['Ash_Alcanity'], ax=ax2)

sns.kdeplot(new_df['Malic_Acid'], ax=ax2)

plt.show()
X=new_df.iloc[:,1:].values

Error =[]

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i).fit(X)

    kmeans.fit(X)

    Error.append(kmeans.inertia_)



plt.plot(range(1, 11), Error)

plt.title('Elbow method')

plt.xlabel('No of clusters')

plt.ylabel('Error')

plt.show()

kmeans3=KMeans(n_clusters = 3).fit(X)

y_pred3=kmeans3.fit_predict(X)

print(y_pred3)
kmeans3.cluster_centers_
plt.scatter(X[:,5],X[:,4],c=y_pred3,cmap="rainbow")