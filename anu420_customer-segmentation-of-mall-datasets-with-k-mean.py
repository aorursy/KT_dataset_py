# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt # data visualization

from sklearn.cluster import KMeans # Import Sklearn KMeans clustering 

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/mall-customers/Mall_Customers.csv')

# check first five rows

df.head()
df.describe()
df.isnull().sum()
df.dtypes
# Count Genre 

df.Genre.value_counts()
# Groupby Genre visualization

# df.groupby('Genre').Genre.value_counts().unstack().plot.bar()

sns.countplot(x='Genre', data=df)

plt.title('Customer gender density')

plt.show()
totalgenre = df.Genre.value_counts()

genrelabel = ['Female', 'Male']
plt.axis('equal') # For perfect circle

plt.pie(totalgenre, labels=genrelabel, radius=1.5, autopct='%0.2f%%', shadow=True, explode=[0, 0], startangle=45)

# radius increase the size, autopct for show percentage two decimal point

plt.title('Pie Chart Depicting Ratio of Female & Male')

plt.show() 

#remove extra text
df['Age'].describe()
my_bins=10

# Histogram used by deafult 10 bins . bins like range.

arr=plt.hist(df['Age'],bins=my_bins, rwidth=0.95) 

plt.xlabel('Age Class')

plt.ylabel('Frequency')

plt.title('Histogram to Show of Age Class')

for i in range(my_bins):

    plt.text(arr[1][i],arr[0][i],str(arr[0][i]))
plt.boxplot(df["Age"])

plt.title('Boxplot for Descriptive Analysis of Age')

plt.show()
df['Annual Income (k$)'].describe()
my_bins=10

# Histogram used by deafult 10 bins . bins like range.

arr=plt.hist(df['Annual Income (k$)'],bins=my_bins, rwidth=0.95) 

plt.xlabel('Age Class')

plt.ylabel('Frequency')

plt.title('Histogram to Show of Age Class')

for i in range(my_bins):

    plt.text(arr[1][i],arr[0][i],str(arr[0][i]))
# Density Plot and Histogram of all arrival delays

sns.distplot(df['Annual Income (k$)'], hist=True, kde=True, 

            color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})
# Check the summary of Spending Score of the Customers

df['Spending Score (1-100)'].describe()
my_bins=10

# Histogram used by deafult 10 bins . bins like range.

arr=plt.hist(df['Spending Score (1-100)'],bins=my_bins, rwidth=0.95) 

plt.xlabel('Spending Score Class')

plt.ylabel('Frequency')

plt.title('Histogram for Spending Score')

for i in range(my_bins):

    plt.text(arr[1][i],arr[0][i],str(arr[0][i]))
plt.boxplot(df["Spending Score (1-100)"])

plt.title('Boxplot for Descriptive Analysis of Spending Score')

plt.show()
km = KMeans(n_clusters=3)

km
y_predicted = km.fit_predict(df[['Annual Income (k$)', 'Spending Score (1-100)']])

# clustering

y_predicted
df['cluster'] = y_predicted

df.head()
#check centroid

km.cluster_centers_
df0 = df[df.cluster == 0]

df1 = df[df.cluster == 1]

df2 = df[df.cluster == 2]
plt.scatter(df0['Annual Income (k$)'], df0['Spending Score (1-100)'], color='green')

plt.scatter(df1['Annual Income (k$)'], df1['Spending Score (1-100)'], color='red')

plt.scatter(df2['Annual Income (k$)'], df2['Spending Score (1-100)'], color='black')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='yellow', marker='o', label='centroid', s=200)

plt.xlabel('Anual Income')

plt.ylabel('Spending Score')

plt.legend(['Cluster1', 'Cluster2', 'Cluster3', 'centroid'])
k_rng = range(1, 10)

sse = []

for k in k_rng:

    km = KMeans(n_clusters=k)

    km.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])

    sse.append(km.inertia_)

sse
plt.xlabel('K')

plt.ylabel('Sum of Square Error(SSE)')

plt.plot(k_rng, sse)
km = KMeans(n_clusters=5)

y_predicted = km.fit_predict(df[['Annual Income (k$)', 'Spending Score (1-100)']])

# clustering

y_predicted
df['cluster'] = y_predicted

df.head()
km.cluster_centers_
df0 = df[df.cluster == 0]

df1 = df[df.cluster == 1]

df2 = df[df.cluster == 2]

df3 = df[df.cluster == 3]

df4 = df[df.cluster == 4]

plt.scatter(df0['Annual Income (k$)'], df0['Spending Score (1-100)'], color='green')

plt.scatter(df1['Annual Income (k$)'], df1['Spending Score (1-100)'], color='red')

plt.scatter(df2['Annual Income (k$)'], df2['Spending Score (1-100)'], color='blue')

plt.scatter(df3['Annual Income (k$)'], df3['Spending Score (1-100)'], color='cyan')

plt.scatter(df4['Annual Income (k$)'], df4['Spending Score (1-100)'], color='magenta')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='yellow', marker='o', label='centroid', s=150)

plt.xlabel('Anual Income')

plt.ylabel('Spending Score')

plt.legend(['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4', 'Cluster5', 'centroid'])