# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt

import seaborn as sns



from scipy.cluster.hierarchy import dendrogram

from sklearn.cluster import AgglomerativeClustering



df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df
df.shape
df.head()
df.isnull().sum()
df[["name", "host_name"]] = df[["name", "host_name"]].fillna("None")
df[["last_review", "reviews_per_month"]] = df[["last_review", "reviews_per_month"]].fillna(0)
df.describe()
df.dtypes
#separate out numerical variables

a=pd.DataFrame(df.dtypes.copy())

b= a[a[0] != 'object'].reset_index()

#drop id and host id:

numeric_vars=b["index"].tolist()[2:]



fig = plt.figure(figsize=(14,14))

ax1 = fig.add_subplot(3, 3, 1)

ax2 = fig.add_subplot(3, 3, 2)

ax3 = fig.add_subplot(3, 3, 3)

ax4 = fig.add_subplot(3, 3, 4)

ax5 = fig.add_subplot(3, 3, 5)

ax6 = fig.add_subplot(3, 3, 6)

ax7 = fig.add_subplot(3, 3, 7)

ax8 = fig.add_subplot(3, 3, 8)



ax1.hist(df[numeric_vars[0]], bins=30)

ax1.set_ylabel("Frequency")

ax1.set_title(numeric_vars[0])



ax2.hist(df[numeric_vars[1]], bins=30)

ax2.set_ylabel("Frequency")

ax2.set_title(numeric_vars[1])



ax3.hist((df[numeric_vars[2]]), bins=30)

ax3.set_ylabel("Frequency")

ax3.set_title('price')



ax4.hist(df[numeric_vars[3]], bins=31)

ax4.set_ylabel("Frequency")

ax4.set_title(numeric_vars[3])



ax5.hist(df[numeric_vars[4]], bins=30)

ax5.set_ylabel("Frequency")

ax5.set_title("number of reviews")



ax6.hist(df[numeric_vars[5]], bins=30)

ax6.set_ylabel("Frequency")

ax6.set_title("last review")



ax7.hist(df[numeric_vars[6]], bins=30)

ax7.set_ylabel("Frequency")

ax7.set_title(numeric_vars[6])



ax8.hist(df[numeric_vars[7]])

ax8.set_ylabel("Frequency")

ax8.set_title(numeric_vars[7])

plt.show()
f,ax = plt.subplots(figsize=(15,6))

ax = sns.countplot(df.neighbourhood_group,palette="muted")

plt.show()
f,ax = plt.subplots(figsize=(16,8))

ax = sns.scatterplot(y=df.latitude,x=df.longitude,hue=df.neighbourhood_group,palette="coolwarm")

plt.show()
print(df.iloc[:len(df)//2,9:11].values)
x = df.iloc[:len(df)//2,9:11].values

model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model.fit(x)
def plot_dendrogram(model, **kwargs):

    # Create linkage matrix and then plot the dendrogram



    # create the counts of samples under each node

    counts = np.zeros(model.children_.shape[0])

    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):

        current_count = 0

        for child_idx in merge:

            if child_idx < n_samples:

                current_count += 1  # leaf node

            else:

                current_count += counts[child_idx - n_samples]

        counts[i] = current_count



    linkage_matrix = np.column_stack([model.children_, model.distances_,

                                      counts]).astype(float)



    # Plot the corresponding dendrogram

    dendrogram(linkage_matrix, **kwargs)
plt.title('Hierarchical Clustering Dendrogram')

# plot the top three levels of the dendrogram

plot_dendrogram(model, truncate_mode='level', p=3)

plt.xlabel("Number of points in node (or index of point if no parenthesis).")

plt.show()