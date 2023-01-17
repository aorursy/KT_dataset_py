# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
import pandas as pd

df_raw = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
df_raw.describe(include='all')
# Gender 

df = df_raw.copy()

df['gender'] = df['gender'].map({'female':0, 'male':1})



labels = 'female', 'male'

size = 518, 418



fig1, ax1 = plt.subplots()

ax1.pie(size, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal') 



plt.show()
fig, axs = plt.subplots(1, 3, figsize=(25, 5), sharey=True)

axs[0].scatter(df.index, df['math score'], c = df['gender'], cmap='rainbow')

axs[1].scatter(df.index, df['reading score'], c = df['gender'], cmap='rainbow')

axs[2].scatter(df.index, df['writing score'], c = df['gender'], cmap='rainbow')

fig.suptitle('math scores, reading, writting')



# female = blue, male = red
df_clust = df.copy()

x = df_clust.iloc[:, 0::5]

x.head()

from sklearn.cluster import KMeans

kmeans = KMeans(4)

kmeans.fit(x)
ready_cluster = kmeans.fit_predict(x)

df_clust = df.copy()
df_clust['Cluster'] = ready_cluster

df_clust
plt.scatter(df_clust.index, df_clust['math score'],c = df_clust['Cluster'], cmap='rainbow' )



plt.show()
kmeans.inertia_
wcss = []



for i in range(1,8):

    kmeans = KMeans(i)

    kmeans.fit(x)

    wcss_iter = kmeans.inertia_

    wcss.append(wcss_iter)
wcss
number_clust = range(1,8)

plt.plot(number_clust, wcss)
df['race/ethnicity'] = df['race/ethnicity'].map({'group B':1, 'group C':2, 'group A':3, 'group D':4, 'group E':5})

df['parental level of education'] = df['parental level of education'].map({'some high school':1, 'high school':2, "associate's degree":3, 'some college':4, "bachelor's degree":5, "master's degree":6})

df['lunch'] = df['lunch'].map({r'standard':0, "free/reduced":1})

df['test preparation course'] = df['test preparation course'].map({'none':0, 'completed':1})

df.head()
df.corr()
colormap = plt.cm.RdBu

plt.figure(figsize=(10,10))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(df.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
g = sns.pairplot(df[[u'gender', u'race/ethnicity', u'parental level of education', u'lunch', u'test preparation course', u'math score', u'reading score',

       u'writing score']], hue='gender', palette = 'seismic',size=2.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=20) )

g.set(xticklabels=[])