# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize ,StandardScaler
from sklearn.decomposition import PCA

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
creditcard_df=pd.read_csv('/kaggle/input/ccdata/CC GENERAL.csv')
creditcard_df

creditcard_df.info()
print('Average,Max,Min of BALANCE col:',creditcard_df['BALANCE'].mean()," ",creditcard_df['BALANCE'].max()," ",creditcard_df['BALANCE'].min())
creditcard_df['BALANCE'].describe()
creditcard_df.describe()
creditcard_df[(creditcard_df['ONEOFF_PURCHASES']==creditcard_df['ONEOFF_PURCHASES'].max())]
creditcard_df[(creditcard_df['CASH_ADVANCE']==creditcard_df['CASH_ADVANCE'].max())]
sns.heatmap(creditcard_df.isnull(),cmap='Blues')
#as mentioned before ,  'MINIMUM_PAYMENTS' having some missing values
# to figure out how many missing values we got in our dataset
creditcard_df.isnull().sum()
#fill missing data with the average value 
creditcard_df.loc[(creditcard_df['MINIMUM_PAYMENTS'].isnull()==True),'MINIMUM_PAYMENTS']=creditcard_df['MINIMUM_PAYMENTS'].mean()

#to cheak if it works
creditcard_df.isnull().sum()
creditcard_df.loc[(creditcard_df['CREDIT_LIMIT'].isnull()==True),'CREDIT_LIMIT']=creditcard_df['CREDIT_LIMIT'].mean()
#to check
creditcard_df.isnull().sum()
#Double cheak if it works
sns.heatmap(creditcard_df.isnull(),cmap='Blues')
creditcard_df.duplicated().sum()
creditcard_df.drop('CUST_ID',axis=1,inplace=True)
creditcard_df
plt.figure(figsize=(10,50))
for i in range(0,len(creditcard_df.columns)):
    plt.subplot(17,1,i+1)
    sns.distplot(creditcard_df[creditcard_df.columns[i]],kde_kws={'color':'b','lw':3,'label':'KDE','bw':0},hist_kws={'color':'g'})
    plt.title(creditcard_df.columns[i])


plt.tight_layout()
    

    
corrolations=creditcard_df.corr()
f,ax=plt.subplots(figsize=(20,10))
sns.heatmap(corrolations,cmap='Blues',annot=True)
scale=StandardScaler()
creditcard_df_scaled=scale.fit_transform(creditcard_df)
creditcard_df_scaled
scores_1=[]
range_values=range(1,20)
#to get WCSS in each iteration 
for i in range_values:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(creditcard_df_scaled)
    scores_1.append(kmeans.inertia_) 
plt.plot(scores_1)    
plt.xlabel('#K clusters')
plt.ylabel('WCSS')
            


kmeans=KMeans(n_clusters=7)
kmeans.fit(creditcard_df_scaled)
labels=kmeans.labels_
kmeans.cluster_centers_.shape
cluster_centers=pd.DataFrame(data=kmeans.cluster_centers_,columns=[creditcard_df.columns])
cluster_centers
# In order to understand what these numbers mean, let's perform inverse transformation
cluster_centers = scale.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers, columns = [creditcard_df.columns])
cluster_centers

# First Customers cluster (Transactors): Those are customers who pay least amount of intrerest charges and careful with their money, Cluster with lowest balance ($104) and cash advance ($303), Percentage of full payment = 23%
# Second customers cluster (revolvers) who use credit card as a loan (most lucrative sector): highest balance ($5000) and cash advance (~$5000), low purchase frequency, high cash advance frequency (0.5), high cash advance transactions (16) and low percentage of full payment (3%)
# Third customer cluster (VIP/Prime): high credit limit $16K and highest percentage of full payment, target for increase credit limit and increase spending habits
# Fourth customer cluster (low tenure): these are customers with low tenure (7 years), low balance 
labels.shape
labels.max()
labels.min()
k_means_predict=kmeans.fit_predict(creditcard_df_scaled)
k_means_predict
creditcard_df_clusters=pd.concat([creditcard_df,pd.DataFrame({'cluster':labels})],axis=1)
creditcard_df_clusters.head()

# Plot the histogram of various clusters
for i in creditcard_df.columns:
  plt.figure(figsize = (35, 5))
  for j in range(5):
    plt.subplot(1,7,j+1)
    cluster = creditcard_df_clusters[creditcard_df_clusters['cluster'] == j]
    cluster[i].hist(bins = 20)
    plt.title('{}    \nCluster {} '.format(i,j))
  
  plt.show()
