import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = 10, 7.5
plt.rcParams['axes.grid'] = True
import seaborn as sns
import scipy.stats as stats
import pandas_profiling
train=pd.read_csv('../input/train.csv')
train.head()
train.columns
train.info()
print("The no of rows : {}".format(train.shape[0]))
print("The no of columns : {} ".format(train.shape[1]))
print(train.isnull().sum() ) 
#train_isnull=train.isnull().sum().to_csv("train_isnull.csv")
print("______________________________________________________________")

print("There are no missing value")
#Detailed profiling using pandas profiling
#pandas_profiling.ProfileReport(train)
numeric_var_names=[key for key in dict(train.dtypes) if dict(train.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]
cat_var_names=[key for key in dict(train.dtypes) if dict(train.dtypes)[key] in ['object']]
print(numeric_var_names)
print(cat_var_names)
numeric_var_names
train_num=train[numeric_var_names]
train_num.head()
# Creating Data audit Report
# Use a general function that returns multiple values
def var_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.dropna().quantile(0.01), x.dropna().quantile(0.05),x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75), x.dropna().quantile(0.90),x.dropna().quantile(0.95), x.dropna().quantile(0.99),x.max()], 
                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])
num_var_summary=train_num.apply(var_summary).T
num_var_summary
#Handling Outliers -
def outlier_treatment(x):
    x = x.clip_upper(x.quantile(0.99))
    x = x.clip_lower(x.quantile(0.01))
    return x
train_num=train_num.apply(outlier_treatment)
train_num.corr()
#train_correlation=train_num.corr().to_csv("train_correlation.csv")
train_cat=train[cat_var_names]
train_cat.head()
# An utility function to create dummy variable
def create_dummies( df, colname ):
    col_dummies = pd.get_dummies(df[colname], prefix=colname)
    col_dummies.drop(col_dummies.columns[0], axis=1, inplace=True)
    df = pd.concat([df, col_dummies], axis=1)
    df.drop( colname, axis = 1, inplace = True )
    return df
for c_feature in cat_var_names:
    train_cat = create_dummies( train_cat, c_feature )
train_cat.head()
from sklearn.preprocessing import StandardScaler
pc=StandardScaler()
train_num_scaled=pc.fit_transform(train_num)
train_num.columns
train_num=pd.DataFrame(train_num_scaled,columns=train_num.columns)
train_num.head()
# Clustering
from sklearn.cluster import KMeans
km_3=KMeans(n_clusters=3,random_state=123)
km_3.fit(train_num)
km_3.cluster_centers_
km_3.labels_
pd.Series(km_3.labels_).value_counts()
km_4=KMeans(n_clusters=4,random_state=123).fit(train_num)

#km_4.labels_

km_5=KMeans(n_clusters=5,random_state=123).fit(train_num)

#km_5.labels_

km_6=KMeans(n_clusters=6,random_state=123).fit(train_num)

#km_6.labels_

km_7=KMeans(n_clusters=7,random_state=123).fit(train_num)
#km_7.labels_

km_8=KMeans(n_clusters=8,random_state=123).fit(train_num)
#km_8.labels_
# save the cluster labels and sort by cluster
train_num['cluster_3'] = km_3.labels_
train_num['cluster_4'] = km_4.labels_
train_num['cluster_5'] = km_5.labels_
train_num['cluster_6'] = km_6.labels_
train_num['cluster_7'] = km_7.labels_
train_num['cluster_8'] = km_8.labels_
train_num.head()
pd.Series.sort_index(train_num.cluster_3.value_counts())
pd.Series(train_num.cluster_3.size)
size=pd.concat([pd.Series(train_num.cluster_3.size), pd.Series.sort_index(train_num.cluster_3.value_counts()), pd.Series.sort_index(train_num.cluster_4.value_counts()),
           pd.Series.sort_index(train_num.cluster_5.value_counts()), pd.Series.sort_index(train_num.cluster_6.value_counts()),
           pd.Series.sort_index(train_num.cluster_7.value_counts()), pd.Series.sort_index(train_num.cluster_8.value_counts())])
Seg_size=pd.DataFrame(size, columns=['Seg_size'])
Seg_Pct = pd.DataFrame(size/train_num.cluster_3.size, columns=['Seg_Pct'])
Seg_size.T
Seg_Pct.T
# Mean value gives a good indication of the distribution of data. So we are finding mean value for each variable for each cluster
Profling_output = pd.concat([train_num.apply(lambda x: x.mean()).T, train_num.groupby('cluster_3').apply(lambda x: x.mean()).T, train_num.groupby('cluster_4').apply(lambda x: x.mean()).T,
          train_num.groupby('cluster_5').apply(lambda x: x.mean()).T, train_num.groupby('cluster_6').apply(lambda x: x.mean()).T,
          train_num.groupby('cluster_7').apply(lambda x: x.mean()).T, train_num.groupby('cluster_8').apply(lambda x: x.mean()).T], axis=1)

Profling_output_final=pd.concat([Seg_size.T, Seg_Pct.T, Profling_output], axis=0)
#Profling_output_final.columns = ['Seg_' + str(i) for i in Profling_output_final.columns]
Profling_output_final.columns = ['Overall', 'KM3_1', 'KM3_2', 'KM3_3',
                                'KM4_1', 'KM4_2', 'KM4_3', 'KM4_4',
                                'KM5_1', 'KM5_2', 'KM5_3', 'KM5_4', 'KM5_5',
                                'KM6_1', 'KM6_2', 'KM6_3', 'KM6_4', 'KM6_5','KM6_6',
                                'KM7_1', 'KM7_2', 'KM7_3', 'KM7_4', 'KM7_5','KM7_6','KM7_7',
                                'KM8_1', 'KM8_2', 'KM8_3', 'KM8_4', 'KM8_5','KM8_6','KM8_7','KM8_8',]
Profling_output_final
Profling_output_final.to_csv('Profiling_output.csv')
# Elbow Plot
cluster_range = range( 1, 20 )
cluster_errors = []

for num_clusters in cluster_range:
    clusters = KMeans( num_clusters )
    clusters.fit( train_num )
    cluster_errors.append( clusters.inertia_ )
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

clusters_df[0:10]
%matplotlib inline
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
from sklearn import metrics
# calculate SC for K=3 through K=12
k_range = range(2, 12)
scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=1)
    km.fit(train_num)
    scores.append(metrics.silhouette_score(train_num, km.labels_))
scores
# The sc is maximum for k=2 so we will select the 2 as our optimum cluster
# plot the results
plt.plot(k_range, scores)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficient')
plt.grid(True)
