# Importing Libraries

import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from kmodes.kmodes import KModes

import warnings

warnings.filterwarnings("ignore") 
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

bank = pd.read_csv(r"/kaggle/input/bank-marketing-propensity-data/bank-additional-full.csv")
bank.head()
bank.columns
bank_cust = bank[['age','job', 'marital', 'education', 'default', 'housing', 'loan','contact','month','day_of_week','poutcome']]
bank_cust.head()
bank_cust['age_bin'] = pd.cut(bank_cust['age'], [0, 20, 30, 40, 50, 60, 70, 80, 90, 100], 

                              labels=['0-20', '20-30', '30-40', '40-50','50-60','60-70','70-80', '80-90','90-100'])
bank_cust.head()
bank_cust  = bank_cust.drop('age',axis = 1)
bank_cust.head()
bank_cust.info()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

bank_cust = bank_cust.apply(le.fit_transform)

bank_cust.head()
# Checking the count per category

job_df = pd.DataFrame(bank_cust['job'].value_counts())

job_df.head()
plt.figure(figsize=(10, 12))

ax=sns.barplot(x=job_df.index, y=job_df['job'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set_yscale('log')

plt.show()
# Checking the count per category

age_df = pd.DataFrame(bank_cust['age_bin'].value_counts())

age_df.head()
plt.figure(figsize=(10, 9))

ax=sns.barplot(x=age_df.index, y=age_df['age_bin'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.2))

ax.set_yscale('log')

plt.show()
km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)

fitClusters_cao = km_cao.fit_predict(bank_cust)
# Predicted Clusters

fitClusters_cao
clusterCentroidsDf = pd.DataFrame(km_cao.cluster_centroids_)

clusterCentroidsDf.columns = bank_cust.columns
# Mode of the clusters

clusterCentroidsDf
km_huang = KModes(n_clusters=2, init = "Huang", n_init = 1, verbose=1)

fitClusters_huang = km_huang.fit_predict(bank_cust)
# Predicted clusters

fitClusters_huang
cost = []

for num_clusters in list(range(1,5)):

    kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)

    kmode.fit_predict(bank_cust)

    cost.append(kmode.cost_)
y = np.array([i for i in range(1,5,1)])

plt.plot(y,cost);
## Choosing K=2
km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)

fitClusters_cao = km_cao.fit_predict(bank_cust)
fitClusters_cao
bank_cust = bank_cust.reset_index()

clustersDf = pd.DataFrame(fitClusters_cao)

clustersDf.columns = ['cluster_predicted']

combinedDf = pd.concat([bank_cust, clustersDf], axis = 1).reset_index()

combinedDf = combinedDf.drop(['index', 'level_0'], axis = 1)
combinedDf.head()
# Data for Cluster1

cluster1 = combinedDf[combinedDf.cluster_predicted==1]
# Data for Cluster0

cluster0 = combinedDf[combinedDf.cluster_predicted==0]
cluster1.info()
cluster0.info()
# Checking the count per category for JOB

job1_df = pd.DataFrame(cluster1['job'].value_counts())

job1_df.head()
job0_df = pd.DataFrame(cluster0['job'].value_counts())
job0_df.head()
fig, ax =plt.subplots(1,2,figsize=(20,8))



a=sns.barplot(x=job1_df.index, y=job1_df['job'], ax=ax[0])

for p in a.patches:

    a.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

a.set_yscale('log')

b=sns.barplot(x=job0_df.index, y=job0_df['job'], ax=ax[1])

for p in b.patches:

    b.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

b.set_yscale('log')

fig.show()
age1_df = pd.DataFrame(cluster1['age_bin'].value_counts())

age1_df.head()
age0_df = pd.DataFrame(cluster0['age_bin'].value_counts())
age0_df.head()
fig, ax =plt.subplots(1,2,figsize=(20,8))

a=sns.barplot(x=age1_df.index, y=age1_df['age_bin'], ax=ax[0])

for p in a.patches:

    a.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

a.set_yscale('log')

b=sns.barplot(x=age0_df.index, y=age0_df['age_bin'], ax=ax[1])

for p in b.patches:

    b.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

b.set_yscale('log')

fig.show()
cluster1['marital'].value_counts()
cluster0['marital'].value_counts()
cluster1['education'].value_counts()

cluster0['education'].value_counts()