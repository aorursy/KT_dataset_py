# supress warnings

import warnings

warnings.filterwarnings('ignore')



# Importing all required packages

import numpy as np

import pandas as pd



# Data viz lib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from matplotlib.pyplot import xticks
bank = pd.read_csv('../input/bankmarketing.csv')
bank.head()
bank.columns
# Importing Categorical Columns
bank_cust = bank[['age','job', 'marital', 'education', 'default', 'housing', 'loan','contact','month','day_of_week','poutcome']]
bank_cust.head()
# Converting age into categorical variable.
bank_cust['age_bin'] = pd.cut(bank_cust['age'], [0, 20, 30, 40, 50, 60, 70, 80, 90, 100], 

                              labels=['0-20', '20-30', '30-40', '40-50','50-60','60-70','70-80', '80-90','90-100'])

bank_cust  = bank_cust.drop('age',axis = 1)
bank_cust.head()
bank_cust.shape
bank_cust.describe()
bank_cust.info()
# Checking Null values

bank_cust.isnull().sum()*100/bank_cust.shape[0]

# There are no NULL values in the dataset, hence it is clean.
# Data is clean.
# First we will keep a copy of data

bank_cust_copy = bank_cust.copy()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

bank_cust = bank_cust.apply(le.fit_transform)

bank_cust.head()
# Importing Libraries



from kmodes.kmodes import KModes
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

plt.plot(y,cost)
## Choosing K=2
km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)

fitClusters_cao = km_cao.fit_predict(bank_cust)
fitClusters_cao
bank_cust = bank_cust_copy.reset_index()
clustersDf = pd.DataFrame(fitClusters_cao)

clustersDf.columns = ['cluster_predicted']

combinedDf = pd.concat([bank_cust, clustersDf], axis = 1).reset_index()

combinedDf = combinedDf.drop(['index', 'level_0'], axis = 1)
combinedDf.head()
cluster_0 = combinedDf[combinedDf['cluster_predicted'] == 0]

cluster_1 = combinedDf[combinedDf['cluster_predicted'] == 1]
cluster_0.info()
cluster_1.info()
# Job
plt.subplots(figsize = (15,5))

sns.countplot(x=combinedDf['job'],order=combinedDf['job'].value_counts().index,hue=combinedDf['cluster_predicted'])

plt.show()
# Marital
plt.subplots(figsize = (5,5))

sns.countplot(x=combinedDf['marital'],order=combinedDf['marital'].value_counts().index,hue=combinedDf['cluster_predicted'])

plt.show()
# Education
plt.subplots(figsize = (15,5))

sns.countplot(x=combinedDf['education'],order=combinedDf['education'].value_counts().index,hue=combinedDf['cluster_predicted'])

plt.show()
# Default
f, axs = plt.subplots(1,3,figsize = (15,5))

sns.countplot(x=combinedDf['default'],order=combinedDf['default'].value_counts().index,hue=combinedDf['cluster_predicted'],ax=axs[0])

sns.countplot(x=combinedDf['housing'],order=combinedDf['housing'].value_counts().index,hue=combinedDf['cluster_predicted'],ax=axs[1])

sns.countplot(x=combinedDf['loan'],order=combinedDf['loan'].value_counts().index,hue=combinedDf['cluster_predicted'],ax=axs[2])



plt.tight_layout()

plt.show()
f, axs = plt.subplots(1,2,figsize = (15,5))

sns.countplot(x=combinedDf['month'],order=combinedDf['month'].value_counts().index,hue=combinedDf['cluster_predicted'],ax=axs[0])

sns.countplot(x=combinedDf['day_of_week'],order=combinedDf['day_of_week'].value_counts().index,hue=combinedDf['cluster_predicted'],ax=axs[1])



plt.tight_layout()

plt.show()
f, axs = plt.subplots(1,2,figsize = (15,5))

sns.countplot(x=combinedDf['poutcome'],order=combinedDf['poutcome'].value_counts().index,hue=combinedDf['cluster_predicted'],ax=axs[0])

sns.countplot(x=combinedDf['age_bin'],order=combinedDf['age_bin'].value_counts().index,hue=combinedDf['cluster_predicted'],ax=axs[1])



plt.tight_layout()

plt.show()
# Above visualization can help in identification of clusters.