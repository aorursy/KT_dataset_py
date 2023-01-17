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
f = open("/kaggle/input/big-five-personality-test/IPIP-FFM-data-8Nov2018/codebook.txt", "r")

print(f.read())
data = pd.read_csv('/kaggle/input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv', sep='\t')
data.head()
len(data)
# Remove test where one of the answers is 0 (did not answer the question)

# As there are many columns, we will do it like this

for i in range(0,50):

    data = data[data[data.columns[i]] != 0]
# Remove NaN values

data = data.dropna()
len(data)
countries = data[data.country != 'NONE']

countries.country.value_counts()
# Top 10 countries

plt.figure(figsize=(10,6))

sns.countplot(countries.country, order=countries.country.value_counts().iloc[:10].index)
data['EXT1_E'].min()
# Filter on time spent on questions (let's say 1 min per question, to have accurate answers & time positive)

# 1 min = 60s = 60000 milliseconds

for i in range(50,100):

    data = data[(data[data.columns[i]] < 60000) & (data[data.columns[i]] > 0)]
# Sum all questions time

data['timetest'] = data.iloc[:, 51:100].sum(axis=1)



# Convert in seconds

data['timetest'] = data['timetest'] / 1000
plt.figure(figsize=(10,6))

sns.kdeplot(data.timetest)
# Associate text to answer

labels = ['I am the life of the party.', 'I don\'t talk a lot.', 'I feel comfortable around people.', 'I keep in the background.', 

          'I start conversations.', 'I have little to say.', 'I talk to a lot of different people at parties.', 

          'I don\'t like to draw attention to myself.', 'I don\'t mind being the center of attention.', 'I am quiet around strangers.', 

          'I get stressed out easily.', 'I am relaxed most of the time.', 'I worry about things.', 'I seldom feel blue.', 

          'I am easily disturbed.', 'I get upset easily.', 'I change my mood a lot.', 'I have frequent mood swings.', 'I get irritated easily.', 

          'I often feel blue.', 'I feel little concern for others.', 'I am interested in people.', 'I insult people.', 

          'I sympathize with others\' feelings.', 'I am not interested in other people\'s problems.', 'I have a soft heart.', 

          'I am not really interested in others.', 'I take time out for others.', 'I feel others\' emotions.', 'I make people feel at ease.', 

          'I am always prepared.', 'I leave my belongings around.', 'I pay attention to details.', 'I make a mess of things.', 

          'I get chores done right away.', 'I often forget to put things back in their proper place.', 'I like order.', 'I shirk my duties.', 

          'I follow a schedule.', 'I am exacting in my work.', 'I have a rich vocabulary.', 'I have difficulty understanding abstract ideas.',

          'I have a vivid imagination.', 'I am not interested in abstract ideas.', 'I have excellent ideas.', 

          'I do not have a good imagination.', 'I am quick to understand things.', 'I use difficult words.', 

          'I spend time reflecting on things.', 'I am full of ideas.']
fig, axes = plt.subplots(5, 2,figsize=(20,18))



ax1 = sns.countplot(data[data.columns[0]], ax=axes[0,0])

ax1.set_xlabel('')

ax2 = sns.countplot(data[data.columns[1]],ax=axes[0,1])

ax2.set_xlabel('')

ax3 = sns.countplot(data[data.columns[2]], ax=axes[1,0])

ax3.set_xlabel('')

ax4 = sns.countplot(data[data.columns[3]],ax=axes[1,1])

ax4.set_xlabel('')

ax5 = sns.countplot(data[data.columns[4]], ax=axes[2,0])

ax5.set_xlabel('')

ax6 = sns.countplot(data[data.columns[5]],ax=axes[2,1])

ax6.set_xlabel('')

ax7 = sns.countplot(data[data.columns[6]], ax=axes[3,0])

ax7.set_xlabel('')

ax8 = sns.countplot(data[data.columns[7]],ax=axes[3,1])

ax8.set_xlabel('')

ax9 = sns.countplot(data[data.columns[8]], ax=axes[4,0])

ax9.set_xlabel('')

ax10 = sns.countplot(data[data.columns[9]],ax=axes[4,1])

ax10.set_xlabel('')



ax1.title.set_text(labels[0])

ax2.title.set_text(labels[1])

ax3.title.set_text(labels[2])

ax4.title.set_text(labels[3])

ax5.title.set_text(labels[4])

ax6.title.set_text(labels[5])

ax7.title.set_text(labels[6])

ax8.title.set_text(labels[7])

ax9.title.set_text(labels[8])

ax10.title.set_text(labels[9])



plt.show()
fig, axes = plt.subplots(5, 2,figsize=(20,18))



ax1 = sns.countplot(data[data.columns[10]], ax=axes[0,0])

ax1.set_xlabel('')

ax2 = sns.countplot(data[data.columns[11]],ax=axes[0,1])

ax2.set_xlabel('')

ax3 = sns.countplot(data[data.columns[12]], ax=axes[1,0])

ax3.set_xlabel('')

ax4 = sns.countplot(data[data.columns[13]],ax=axes[1,1])

ax4.set_xlabel('')

ax5 = sns.countplot(data[data.columns[14]], ax=axes[2,0])

ax5.set_xlabel('')

ax6 = sns.countplot(data[data.columns[15]],ax=axes[2,1])

ax6.set_xlabel('')

ax7 = sns.countplot(data[data.columns[16]], ax=axes[3,0])

ax7.set_xlabel('')

ax8 = sns.countplot(data[data.columns[17]],ax=axes[3,1])

ax8.set_xlabel('')

ax9 = sns.countplot(data[data.columns[18]], ax=axes[4,0])

ax9.set_xlabel('')

ax10 = sns.countplot(data[data.columns[19]],ax=axes[4,1])

ax10.set_xlabel('')



ax1.title.set_text(labels[10])

ax2.title.set_text(labels[11])

ax3.title.set_text(labels[12])

ax4.title.set_text(labels[13])

ax5.title.set_text(labels[14])

ax6.title.set_text(labels[15])

ax7.title.set_text(labels[16])

ax8.title.set_text(labels[17])

ax9.title.set_text(labels[18])

ax10.title.set_text(labels[19])



plt.show()
fig, axes = plt.subplots(5, 2,figsize=(20,18))



ax1 = sns.countplot(data[data.columns[20]], ax=axes[0,0])

ax1.set_xlabel('')

ax2 = sns.countplot(data[data.columns[21]],ax=axes[0,1])

ax2.set_xlabel('')

ax3 = sns.countplot(data[data.columns[22]], ax=axes[1,0])

ax3.set_xlabel('')

ax4 = sns.countplot(data[data.columns[23]],ax=axes[1,1])

ax4.set_xlabel('')

ax5 = sns.countplot(data[data.columns[24]], ax=axes[2,0])

ax5.set_xlabel('')

ax6 = sns.countplot(data[data.columns[25]],ax=axes[2,1])

ax6.set_xlabel('')

ax7 = sns.countplot(data[data.columns[26]], ax=axes[3,0])

ax7.set_xlabel('')

ax8 = sns.countplot(data[data.columns[27]],ax=axes[3,1])

ax8.set_xlabel('')

ax9 = sns.countplot(data[data.columns[28]], ax=axes[4,0])

ax9.set_xlabel('')

ax10 = sns.countplot(data[data.columns[29]],ax=axes[4,1])

ax10.set_xlabel('')



ax1.title.set_text(labels[20])

ax2.title.set_text(labels[21])

ax3.title.set_text(labels[22])

ax4.title.set_text(labels[23])

ax5.title.set_text(labels[24])

ax6.title.set_text(labels[25])

ax7.title.set_text(labels[26])

ax8.title.set_text(labels[27])

ax9.title.set_text(labels[28])

ax10.title.set_text(labels[29])



plt.show()
fig, axes = plt.subplots(5, 2,figsize=(20,18))



ax1 = sns.countplot(data[data.columns[30]], ax=axes[0,0])

ax1.set_xlabel('')

ax2 = sns.countplot(data[data.columns[31]],ax=axes[0,1])

ax2.set_xlabel('')

ax3 = sns.countplot(data[data.columns[32]], ax=axes[1,0])

ax3.set_xlabel('')

ax4 = sns.countplot(data[data.columns[33]],ax=axes[1,1])

ax4.set_xlabel('')

ax5 = sns.countplot(data[data.columns[34]], ax=axes[2,0])

ax5.set_xlabel('')

ax6 = sns.countplot(data[data.columns[35]],ax=axes[2,1])

ax6.set_xlabel('')

ax7 = sns.countplot(data[data.columns[36]], ax=axes[3,0])

ax7.set_xlabel('')

ax8 = sns.countplot(data[data.columns[37]],ax=axes[3,1])

ax8.set_xlabel('')

ax9 = sns.countplot(data[data.columns[38]], ax=axes[4,0])

ax9.set_xlabel('')

ax10 = sns.countplot(data[data.columns[39]],ax=axes[4,1])

ax10.set_xlabel('')



ax1.title.set_text(labels[30])

ax2.title.set_text(labels[31])

ax3.title.set_text(labels[32])

ax4.title.set_text(labels[33])

ax5.title.set_text(labels[34])

ax6.title.set_text(labels[35])

ax7.title.set_text(labels[36])

ax8.title.set_text(labels[37])

ax9.title.set_text(labels[38])

ax10.title.set_text(labels[39])



plt.show()
fig, axes = plt.subplots(5, 2,figsize=(20,18))



ax1 = sns.countplot(data[data.columns[40]], ax=axes[0,0])

ax1.set_xlabel('')

ax2 = sns.countplot(data[data.columns[41]],ax=axes[0,1])

ax2.set_xlabel('')

ax3 = sns.countplot(data[data.columns[42]], ax=axes[1,0])

ax3.set_xlabel('')

ax4 = sns.countplot(data[data.columns[43]],ax=axes[1,1])

ax4.set_xlabel('')

ax5 = sns.countplot(data[data.columns[44]], ax=axes[2,0])

ax5.set_xlabel('')

ax6 = sns.countplot(data[data.columns[45]],ax=axes[2,1])

ax6.set_xlabel('')

ax7 = sns.countplot(data[data.columns[46]], ax=axes[3,0])

ax7.set_xlabel('')

ax8 = sns.countplot(data[data.columns[47]],ax=axes[3,1])

ax8.set_xlabel('')

ax9 = sns.countplot(data[data.columns[48]], ax=axes[4,0])

ax9.set_xlabel('')

ax10 = sns.countplot(data[data.columns[49]],ax=axes[4,1])

ax10.set_xlabel('')



ax1.title.set_text(labels[40])

ax2.title.set_text(labels[41])

ax3.title.set_text(labels[42])

ax4.title.set_text(labels[43])

ax5.title.set_text(labels[44])

ax6.title.set_text(labels[45])

ax7.title.set_text(labels[46])

ax8.title.set_text(labels[47])

ax9.title.set_text(labels[48])

ax10.title.set_text(labels[49])



plt.show()
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
# We  want only the answers



df = data.iloc[:,:50]
# Calculate sum of squared distances

ssd = []

K = range(1,10)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(df)

    ssd.append(km.inertia_)
# Plot sum of squared distances / elbow method

plt.figure(figsize=(10,6))

plt.plot(K, ssd, 'bx-')

plt.xlabel('k')

plt.ylabel('ssd')

plt.title('Elbow Method For Optimal k')

plt.show()
# Best number of clusters is 5 (and the study is about identifying 5 personality groups)
# Create and fit model

kmeans = KMeans(n_clusters=5)

model = kmeans.fit(df)
pred = model.labels_

df['Cluster'] = pred
df.head()
# Create PCA for data visualization / Dimensionality reduction to 2D graph

from sklearn.decomposition import PCA



pca = PCA(n_components=2)

pca_model = pca.fit_transform(df)

df_transform = pd.DataFrame(data = pca_model, columns = ['PCA1', 'PCA2'])

df_transform['Cluster'] = pred
df_transform.head()
plt.figure(figsize=(15,15))

g = sns.scatterplot(data=df_transform, x='PCA1', y='PCA2', palette=sns.color_palette()[:5], hue='Cluster')

title = plt.title('Personality Clusters with PCA')
# Explore personalities

# Sum of groups (EXT, EST, AGR, CSN, OPN)

# Rename groups (from https://ipip.ori.org/newBigFive5broadKey.htm)



groups = pd.DataFrame()

groups['Extraversion'] = df[df.columns[:10]].sum(axis=1)/10

groups['Emotional_Stability'] = df[df.columns[10:20]].sum(axis=1)/10

groups['Agreeableness'] = df[df.columns[20:30]].sum(axis=1)/10

groups['Conscientiousness'] = df[df.columns[30:40]].sum(axis=1)/10

groups['Imagination'] = df[df.columns[40:50]].sum(axis=1)/10

groups['Cluster'] = pred
groups = groups.groupby('Cluster').mean()

groups = groups.reset_index()
groups
plt.figure(figsize=(15,8))



plt.plot(groups.columns[1:], groups.iloc[:, 1])

plt.plot(groups.columns[1:], groups.iloc[:, 2])

plt.plot(groups.columns[1:], groups.iloc[:, 3])

plt.plot(groups.columns[1:], groups.iloc[:, 4])

plt.plot(groups.columns[1:], groups.iloc[:, 5])



l = plt.legend(groups.Cluster)

t = plt.xticks(rotation=45)
# Cluster 1 is very different from the others, it represents introvert, relaxed people, less organized with small imagination 