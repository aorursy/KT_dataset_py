# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
data.head()
plt.figure(figsize=(10,6))

sns.countplot(data.gender)
# Almost 50-50
plt.figure(figsize=(10,6))

sns.countplot(data['race/ethnicity'])
plt.figure(figsize=(15,6))

sns.countplot(data['parental level of education'])
plt.figure(figsize=(10,6))

sns.countplot(data.lunch)
plt.figure(figsize=(10,6))

sns.countplot(data['test preparation course'])
plt.figure(figsize=(12,8))

sns.kdeplot(data['math score'])

sns.kdeplot(data['reading score'])

sns.kdeplot(data['writing score'])
# Math a bit more difficult
data['mean_score'] = (data['math score'] + data['reading score'] + data['writing score']) / 3
plt.figure(figsize=(12,8))

sns.kdeplot(data[data.gender == 'female']['mean_score'])

sns.kdeplot(data[data.gender == 'male']['mean_score'])

plt.legend(labels=['Female', 'Male'])
data.groupby('gender')['mean_score'].mean()
# Better scores for females
data.head()
# Grades



# Above 80 = A Grade

# 70 to 80 = B Grade

# 60 to 70 = C Grade

# 50 to 60 = D Grade

# 40 to 50 = E Grade

# Below 40 = F Grade (Fail) 
# Change scores mean to grade

for i, row in data.iterrows():

    if row['mean_score'] >= 80 :

        data.loc[i, 'mean_score'] = 'A'

    if (row['mean_score'] >= 70) & (row['mean_score'] < 80):

        data.loc[i, 'mean_score'] = 'B'

    if (row['mean_score'] >= 60) & (row['mean_score'] < 70):

        data.loc[i, 'mean_score'] = 'C'

    if (row['mean_score'] >= 50) & (row['mean_score'] < 60):

        data.loc[i, 'mean_score'] = 'D'

    if (row['mean_score'] >= 40) & (row['mean_score'] < 50):

        data.loc[i, 'mean_score'] = 'E'

    if (row['mean_score'] < 40):

        data.loc[i, 'mean_score'] = 'F'
grades = data[['mean_score']]
grades['mean_score cat'] = grades['mean_score'].astype('category').cat.codes
# regroup some high shcool with high school

data['parental level of education'] = data['parental level of education'].str.replace('some high school', 'high school')
data['gender'] = data.gender.astype('category').cat.codes

data['race/ethnicity'] = data['race/ethnicity'].astype('category').cat.codes

data['lunch'] = data.lunch.astype('category').cat.codes

data['test preparation course'] = data['test preparation course'].astype('category').cat.codes

data['mean_score'] = data['mean_score'].astype('category').cat.codes
# parental education with map for level of education

data['parental level of education'] = data['parental level of education'].map({'some college': 0, 'high school': 1, 

                                                                               'associate\'s degree': 2, 'bachelor\'s degree': 3,

                                                                              'master\'s degree':4})
data.head()
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
data = data.drop(['math score', 'reading score', 'writing score'], axis=1)
# Calculate sum of squared distances

ssd = []

K = range(1,10)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(data)

    ssd.append(km.inertia_)
# Plot sum of squared distances / elbow method

plt.figure(figsize=(10,6))

plt.plot(K, ssd, 'bx-')

plt.xlabel('k')

plt.ylabel('ssd')

plt.title('Elbow Method For Optimal k')

plt.show()
# The best number of clusters is 3
# Create and fit model

kmeans = KMeans(n_clusters=3)

model = kmeans.fit(data)
pred = model.labels_

data['cluster'] = pred
# Create PCA for data visualization / Dimensionality reduction to 2D graph

from sklearn.decomposition import PCA



pca = PCA(n_components=2)

pca_model = pca.fit_transform(data)

data_transform = pd.DataFrame(data = pca_model, columns = ['PCA1', 'PCA2'])

data_transform['Cluster'] = pred
data_transform.head()
plt.figure(figsize=(10,10))

g = sns.scatterplot(data=data_transform, x='PCA1', y='PCA2', palette=sns.color_palette()[:3], hue='Cluster')

title = plt.title('Students Clusters with PCA')
groups = data.groupby('cluster').mean().reset_index()
df = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')

df = df[['parental level of education']]

df['parental level of education'] = df['parental level of education'].str.replace('some high school', 'high school')

df['parental level of education cat'] = df['parental level of education'].map({'some college': 0, 'high school': 1, 

                                                                               'associate\'s degree': 2, 'bachelor\'s degree': 3,

                                                                              'master\'s degree':4})
df.drop_duplicates()
grades.drop_duplicates()
plt.figure(figsize=(20,10))



plt.plot(groups.iloc[0,1:])

plt.plot(groups.iloc[1,1:])

plt.plot(groups.iloc[2,1:])



l = plt.legend(groups.cluster)

t = plt.xticks(rotation=45)
groups