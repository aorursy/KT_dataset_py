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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import scipy.stats as stats

import pandas_profiling   #need to install using anaconda prompt (pip install pandas_profiling)



%matplotlib inline

plt.rcParams['figure.figsize'] = 10, 7.5

plt.rcParams['axes.grid'] = True



from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans



# center and scale the data

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# reading data into dataframe

CC_GENERAL= pd.read_csv("/kaggle/input/CC_GENERAL.csv")

CC_GENERAL
CC_GENERAL.head()
CC_GENERAL.tail()
CC_GENERAL.info()
CC_GENERAL.dtypes
CC_GENERAL.describe().T
CC_GENERAL.columns
CC_GENERAL.count()
CC_GENERAL.dtypes.value_counts()
CC_GENERAL["Monthly_avg_purchase"] = CC_GENERAL.PURCHASES/CC_GENERAL.TENURE

CC_GENERAL["Monthly_avg_purchase"]

CC_GENERAL
CC_GENERAL["Monthly_CASH_ADVANCE"] = CC_GENERAL.CASH_ADVANCE/CC_GENERAL.TENURE

CC_GENERAL["Monthly_CASH_ADVANCE"]

CC_GENERAL
CC_GENERAL["Purchases_by_type"] = CC_GENERAL.ONEOFF_PURCHASES+CC_GENERAL.INSTALLMENTS_PURCHASES

CC_GENERAL["Purchases_by_type"]

CC_GENERAL
CC_GENERAL["Avg_amt_per_purchase_cash_advance_trx"] = CC_GENERAL.CASH_ADVANCE_TRX + CC_GENERAL.PURCHASES_TRX

CC_GENERAL["Avg_amt_per_purchase_cash_advance_trx"]

CC_GENERAL
CC_GENERAL["Limit_usage"] = CC_GENERAL.BALANCE/CC_GENERAL.CREDIT_LIMIT

CC_GENERAL["Limit_usage"]

CC_GENERAL
CC_GENERAL["Payments_to_minimum_payments_ratio"] = CC_GENERAL.PAYMENTS/CC_GENERAL.MINIMUM_PAYMENTS

CC_GENERAL["Payments_to_minimum_payments_ratio"]

CC_GENERAL
conditions = [

    (CC_GENERAL['ONEOFF_PURCHASES'] == 0) & (CC_GENERAL['INSTALLMENTS_PURCHASES'] == 0),

    (CC_GENERAL['ONEOFF_PURCHASES'] > 0) & (CC_GENERAL['INSTALLMENTS_PURCHASES'] == 0),

    (CC_GENERAL['ONEOFF_PURCHASES'] == 0) & (CC_GENERAL['INSTALLMENTS_PURCHASES'] > 0),

     (CC_GENERAL['ONEOFF_PURCHASES'] > 0) & (CC_GENERAL['INSTALLMENTS_PURCHASES'] > 0)]

choices = ['None', 'One_of', 'Installment_Purchases','Both']

CC_GENERAL['Purchases_type'] = np.select(conditions, choices)

CC_GENERAL
for CUST_ID in CC_GENERAL.columns:

    if(CC_GENERAL[CUST_ID].dtype == 'object'):

        CC_GENERAL[CUST_ID]= CC_GENERAL[CUST_ID].astype('category')

        CC_GENERAL[CUST_ID] = CC_GENERAL[CUST_ID].cat.codes
CC_GENERAL
CC_GENERAL.describe().T
CC_GENERAL.dtypes
#Detailed profiling using pandas profiling



pandas_profiling.ProfileReport(CC_GENERAL)
numeric_var_names=[key for key in dict(CC_GENERAL.dtypes) if dict(CC_GENERAL.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]

cat_var_names=[key for key in dict(CC_GENERAL.dtypes) if dict(CC_GENERAL.dtypes)[key] in ['object']]

print(numeric_var_names)

print(cat_var_names)
data_num = CC_GENERAL[numeric_var_names]

data_num
data_cat = CC_GENERAL[cat_var_names]

data_cat
# Creating Data audit Report

# Use a general function that returns multiple values

def var_summary(x):

    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.dropna().quantile(0.01), x.dropna().quantile(0.05),x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75), x.dropna().quantile(0.90),x.dropna().quantile(0.95), x.dropna().quantile(0.99),x.max()], 

                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])



num_summary=data_num.apply(lambda x: var_summary(x)).T
num_summary
data_num.info()
import numpy as np

for col in data_num.columns:

    percentiles = data_num[col].quantile([0.01,0.99]).values

    data_num[col] = np.clip(data_num[col], percentiles[0], percentiles[1])
data_num
#Handling missings - Method2

def Missing_imputation(x):

    x = x.fillna(x.mean())

    return x



data_num=data_num.apply(lambda x: Missing_imputation(x))

data_num
data_num.corr()
# visualize correlation matrix in Seaborn using a heatmap

sns.heatmap(data_num.corr())
#Handling missings - Method2

def Cat_Missing_imputation(x):

    x = x.fillna(x.mode())

    return x



data_cat=data_cat.apply(lambda x: Cat_Missing_imputation(x))

data_cat
# An utility function to create dummy variable

def create_dummies( df, colname ):

    col_dummies = pd.get_dummies(df[colname], prefix=colname, drop_first=True)

    df = pd.concat([df, col_dummies], axis=1)

    df.drop( colname, axis = 1, inplace = True )

    return df



for c_feature in data_cat.columns:

    data_cat[c_feature] = data_cat[c_feature].astype('category')

    data_cat = create_dummies(data_cat , c_feature )
data_cat.head()
#car_sales=pd.concat(car_sales_num, car_sales_cat)

data_new = pd.concat([data_num, data_cat], axis=1)



data_new.head()
sc=StandardScaler()



data_new_scaled=sc.fit_transform(data_num)

data_new_scaled
pd.DataFrame(data_new_scaled).head()
pd.DataFrame(data_new_scaled).describe()
pc = PCA(n_components=23)
pc.fit(data_new_scaled)
pc.explained_variance_
#The amount of variance that each PC explains

var= pc.explained_variance_ratio_
var
#Cumulative Variance explains

var1=np.cumsum(np.round(pc.explained_variance_ratio_, decimals=4)*100)
var1
pc_final=PCA(n_components=7).fit(data_new_scaled)
pc_final.explained_variance_
reduced_cr=pc_final.fit_transform(data_new_scaled)  # the out put is Factors (F1, F2, ...F9)
dimensions = pd.DataFrame(reduced_cr)
dimensions.columns = ["C1", "C2", "C3", "C4", "C5", "C6","C7"]
dimensions.head()
#pc_final.components_



#print pd.DataFrame(pc_final.components_,columns=telco_num.columns).T



Loadings =  pd.DataFrame((pc_final.components_.T * np.sqrt(pc_final.explained_variance_)).T,columns=data_num.columns).T

Loadings
Loadings.to_csv("Loadings.csv")
list_var = ['PURCHASES','CASH_ADVANCE','Payments_to_minimum_payments_ratio','Monthly_avg_purchase',

'PURCHASES_INSTALLMENTS_FREQUENCY','TENURE','Purchases_by_type','BALANCE','Limit_usage']

list_var
data_new_scaled1=pd.DataFrame(data_new_scaled, columns=data_num.columns)

data_new_scaled1.head(5)



data_new_scaled2=data_new_scaled1[list_var]

data_new_scaled2.head(5)
km_3=KMeans(n_clusters=3,random_state=123)

km_3
km_3.fit(data_new_scaled2)

#km_4.labels_
km_3.labels_
km_3.cluster_centers_
pd.Series(km_3.labels_).value_counts()
pd.Series(km_3.labels_).value_counts()/sum(pd.Series(km_3.labels_).value_counts())
km_4=KMeans(n_clusters=4,random_state=123).fit(data_new_scaled2)

#km_5.labels_a



km_5=KMeans(n_clusters=5,random_state=123).fit(data_new_scaled2)

#km_5.labels_



km_6=KMeans(n_clusters=6,random_state=123).fit(data_new_scaled2)

#km_6.labels_



km_7=KMeans(n_clusters=7,random_state=123).fit(data_new_scaled2)

#km_7.labels_



km_8=KMeans(n_clusters=8,random_state=123).fit(data_new_scaled2)

#km_5.labels_
# Conactenating labels found through Kmeans with data 

#cluster_df_4=pd.concat([telco_num,pd.Series(km_4.labels_,name='Cluster_4')],axis=1)



# save the cluster labels and sort by cluster

data_num['cluster_3'] = km_3.labels_

data_num['cluster_4'] = km_4.labels_

data_num['cluster_5'] = km_5.labels_

data_num['cluster_6'] = km_6.labels_

data_num['cluster_7'] = km_7.labels_

data_num['cluster_8'] = km_8.labels_
data_num.head(5)
pd.Series(km_3.labels_).value_counts()/sum(pd.Series(km_3.labels_).value_counts())
pd.Series(km_4.labels_).value_counts()/sum(pd.Series(km_4.labels_).value_counts())
pd.Series(km_5.labels_).value_counts()/sum(pd.Series(km_5.labels_).value_counts())
pd.Series(km_6.labels_).value_counts()/sum(pd.Series(km_6.labels_).value_counts())
pd.Series(km_7.labels_).value_counts()/sum(pd.Series(km_7.labels_).value_counts())
pd.Series(km_8.labels_).value_counts()/sum(pd.Series(km_8.labels_).value_counts())
# calculate SC for K=3

from sklearn import metrics

metrics.silhouette_score(data_new_scaled2, km_8.labels_)
# calculate SC for K=3 through K=12

k_range = range(3, 9)

scores = []

for k in k_range:

    km = KMeans(n_clusters=k, random_state=123)

    km.fit(data_new_scaled2)

    scores.append(metrics.silhouette_score(data_new_scaled2, km.labels_))
scores
# plot the results

plt.plot(k_range, scores)

plt.xlabel('Number of clusters')

plt.ylabel('Silhouette Coefficient')

plt.grid(True)
data_num.cluster_3.value_counts()/sum(data_num.cluster_3.value_counts())
data_num.cluster_4.value_counts()/sum(data_num.cluster_4.value_counts())
data_num.cluster_5.value_counts()/sum(data_num.cluster_5.value_counts())
data_num.cluster_6.value_counts()/sum(data_num.cluster_6.value_counts())
data_num.cluster_7.value_counts()/sum(data_num.cluster_7.value_counts())
data_num.cluster_8.value_counts()/sum(data_num.cluster_8.value_counts())
cluster_range = range( 1, 20 )

cluster_errors = []



for num_clusters in cluster_range:

    clusters = KMeans( num_clusters )

    clusters.fit( data_new_scaled2 )

    cluster_errors.append( clusters.inertia_ )
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )



clusters_df[0:10]
# allow plots to appear in the notebook

%matplotlib inline

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
# DBSCAN with eps=1 and min_samples=3

from sklearn.cluster import DBSCAN

db = DBSCAN(eps=2.05, min_samples=10)

db.fit(data_new_scaled2)
pd.Series(db.labels_).value_counts()
# review the cluster labels

db.labels_
# save the cluster labels and sort by cluster

data_num['DB_cluster'] = db.labels_
# review the cluster centers

DBSCAN_clustering = data_num.groupby('DB_cluster').mean()

DBSCAN_clustering
DBSCAN_clustering.T
data_num.head()
data_num.cluster_3.value_counts()/1000
data_num.cluster_3.value_counts()*100/sum(data_num.cluster_3.value_counts())
pd.Series.sort_index(data_num.cluster_5.value_counts())
data_num.cluster_3.size
size=pd.concat([pd.Series(data_num.cluster_3.size), pd.Series.sort_index(data_num.cluster_3.value_counts()), pd.Series.sort_index(data_num.cluster_4.value_counts()),

           pd.Series.sort_index(data_num.cluster_5.value_counts()), pd.Series.sort_index(data_num.cluster_6.value_counts()),

           pd.Series.sort_index(data_num.cluster_7.value_counts()), pd.Series.sort_index(data_num.cluster_8.value_counts())])
size
Seg_size=pd.DataFrame(size, columns=['Seg_size'])

Seg_Pct = pd.DataFrame(size/data_num.cluster_3.size, columns=['Seg_Pct'])

Seg_size.T
Seg_Pct.T
data_num.head()
# Mean value gives a good indication of the distribution of data. So we are finding mean value for each variable for each cluster

Profling_output = pd.concat([data_num.apply(lambda x: x.mean()).T, data_num.groupby('cluster_3').apply(lambda x: x.mean()).T, data_num.groupby('cluster_4').apply(lambda x: x.mean()).T,

          data_num.groupby('cluster_5').apply(lambda x: x.mean()).T, data_num.groupby('cluster_6').apply(lambda x: x.mean()).T,

          data_num.groupby('cluster_7').apply(lambda x: x.mean()).T, data_num.groupby('cluster_8').apply(lambda x: x.mean()).T], axis=1)

Profling_output
Profling_output_final=pd.concat([Seg_size.T, Seg_Pct.T, Profling_output], axis=0)
Profling_output_final
#Profling_output_final.columns = ['Seg_' + str(i) for i in Profling_output_final.columns]

Profling_output_final.columns = ['Overall', 'KM3_1', 'KM3_2', 'KM3_3',

                                'KM4_1', 'KM4_2', 'KM4_3', 'KM4_4',

                                'KM5_1', 'KM5_2', 'KM5_3', 'KM5_4', 'KM5_5',

                                'KM6_1', 'KM6_2', 'KM6_3', 'KM6_4', 'KM6_5','KM6_6',

                                'KM7_1', 'KM7_2', 'KM7_3', 'KM7_4', 'KM7_5','KM7_6','KM7_7',

                                'KM8_1', 'KM8_2', 'KM8_3', 'KM8_4', 'KM8_5','KM8_6','KM8_7','KM8_8',]
Profling_output_final
Profling_output_final.to_csv('Profiling_output.csv')