import numpy as np

import pandas as pd

import pandas_profiling

import matplotlib.pyplot as plt

import seaborn as sns

import warnings 

warnings.filterwarnings('ignore')

from sklearn import preprocessing

#from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans 

from sklearn.datasets.samples_generator import make_blobs

from sklearn.metrics import silhouette_score, silhouette_samples



import os

print(os.listdir("../input"))

df = pd.read_csv("../input/CreditCardUsage.csv")

df.head()
df.describe().T
df.info()
df.isna().sum()
df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].median())

df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median())
df.isna().sum()
plt.figure(figsize=(16, 6))

corr = df.corr()

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns,

        annot=True,

        linewidths=0.5,

        cmap="YlGnBu")

plt.title('CreditCardUsage Correlation Heatmap')
x = df2 = df.drop(columns="CUST_ID")

x.head()
# Create the Scaler object

ss = preprocessing.StandardScaler()

# Fit your data on the scaler object

ss_df = ss.fit_transform(x)

ss_df = pd.DataFrame(ss_df, columns=x.columns)

ss_df.head()
# normalize feature data

norm_df = preprocessing.normalize(x)

norm_df = pd.DataFrame(norm_df, columns=x.columns)

norm_df.head()
# Create the MinMaxscaler object

mms = preprocessing.MinMaxScaler()

# Fit your data on the Minmaxscaler object

mms_df = mms.fit_transform(x)

mms_df = pd.DataFrame(mms_df, columns=x.columns)

mms_df.head()
# Create the Robust scaler object

rs = preprocessing.RobustScaler()

# Fit your data on the Robust Scaler object

rs_df = rs.fit_transform(x)

rs_df = pd.DataFrame(rs_df, columns=x.columns)

rs_df.head()
# Create the power Transformer object

pt = preprocessing.PowerTransformer()

pt_df = pt.fit_transform(x)

pt_df = pd.DataFrame(pt_df, columns=x.columns)

pt_df.head()
qt = preprocessing.QuantileTransformer()

qt_df = qt.fit_transform(x)

qt_df = pd.DataFrame(qt_df, columns=x.columns)

qt_df.head()
i=1

f = plt.figure(figsize=(40,100))

for feature_name in x.columns:

    ax=f.add_subplot(x.shape[1],7,i)

    sns.distplot(x[feature_name])

    plt.xlabel(feature_name)

    i=i+1  

    ax=f.add_subplot(x.shape[1],7,i)

    sns.distplot(norm_df[feature_name])

    plt.xlabel(feature_name+" After Normalization")

    i=i+1 

    ax=f.add_subplot(x.shape[1],7,i)

    sns.distplot(mms_df[feature_name])

    plt.xlabel(feature_name+" After MinMaxScaler")

    i=i+1 

    ax=f.add_subplot(x.shape[1],7,i)

    sns.distplot(ss_df[feature_name])

    plt.xlabel(feature_name+" After StandardScaler")

    i=i+1 

    ax=f.add_subplot(x.shape[1],7,i)

    sns.distplot(rs_df[feature_name])

    plt.xlabel(feature_name+" After RobustScaler")

    i=i+1

    ax=f.add_subplot(x.shape[1],7,i)

    sns.distplot(pt_df[feature_name])

    plt.xlabel(feature_name+" After PowerTransformer")

    i=i+1

    ax=f.add_subplot(x.shape[1],7,i)

    sns.distplot(qt_df[feature_name])

    plt.xlabel(feature_name+" After QuantileTransformer")

    i=i+1
testrange = range(1,30)

kmeans = [KMeans(n_clusters=i) for i in testrange]

x_score = [kmeans[i].fit(x).score(x) for i in range(len(kmeans))]

norm_df_score = [kmeans[i].fit(norm_df).score(norm_df) for i in range(len(kmeans))]

mms_df_score = [kmeans[i].fit(mms_df).score(mms_df) for i in range(len(kmeans))]

ss_df_score = [kmeans[i].fit(ss_df).score(ss_df) for i in range(len(kmeans))]

rs_df_score = [kmeans[i].fit(rs_df).score(rs_df) for i in range(len(kmeans))]

pt_df_score = [kmeans[i].fit(pt_df).score(pt_df) for i in range(len(kmeans))]

qt_df_score = [kmeans[i].fit(qt_df).score(qt_df) for i in range(len(kmeans))]
plt.figure(figsize=(20, 20))

plt.subplot(431)

plt.plot(testrange,x_score,'bx-')

plt.xlabel('Number of Clusters X')

plt.ylabel('Score')

plt.subplot(432)

plt.plot(testrange,norm_df_score,'bx-')

plt.xlabel('Number of Clusters Normalization')

plt.ylabel('Score')

plt.subplot(433)

plt.plot(testrange,mms_df_score,'bx-')

plt.xlabel('Number of Clusters MinMaxScaler')

plt.ylabel('Score')

plt.subplot(434)

plt.plot(testrange,ss_df_score,'bx-')

plt.xlabel('Number of Clusters Standardscaler')

plt.ylabel('Score')

plt.subplot(435)

plt.plot(testrange,rs_df_score,'bx-')

plt.xlabel('Number of Clusters RobustScaler')

plt.ylabel('Score')

plt.subplot(436)

plt.plot(testrange,pt_df_score,'bx-')

plt.xlabel('Number of Clusters PowerTransformer')

plt.ylabel('Score')

plt.subplot(437)

plt.plot(testrange,qt_df_score,'bx-')

plt.xlabel('Number of Clusters QantileTransformer')

plt.ylabel('Score')

plt.suptitle('Elbow Curve - Score')

plt.show()
testrange = range(1,30)

kmeans = [KMeans(n_clusters=i) for i in testrange]

x_inertia = [kmeans[i].fit(x).inertia_ for i in range(len(kmeans))]

norm_df_inertia = [kmeans[i].fit(norm_df).inertia_ for i in range(len(kmeans))]

mms_df_inertia = [kmeans[i].fit(mms_df).inertia_ for i in range(len(kmeans))]

ss_df_inertia = [kmeans[i].fit(ss_df).inertia_ for i in range(len(kmeans))]

rs_df_inertia = [kmeans[i].fit(rs_df).inertia_ for i in range(len(kmeans))]

pt_df_inertia = [kmeans[i].fit(pt_df).inertia_ for i in range(len(kmeans))]

qt_df_inertia = [kmeans[i].fit(qt_df).inertia_ for i in range(len(kmeans))]
plt.figure(figsize=(20,20))

plt.subplot(431)

plt.plot(testrange,x_inertia,'bx-')

plt.xlabel('Number of Clusters X')

plt.ylabel('SSD')

plt.subplot(432)

plt.plot(testrange,norm_df_inertia,'bx-')

plt.xlabel('Number of Clusters Normalization')

plt.ylabel('SSD')

plt.subplot(433)

plt.plot(testrange,mms_df_inertia,'bx-')

plt.xlabel('Number of Clusters MinMaxScaler')

plt.ylabel('SSD')

plt.subplot(434)

plt.plot(testrange,ss_df_inertia,'bx-')

plt.xlabel('Number of Clusters Standardscaler')

plt.ylabel('SSD')

plt.subplot(435)

plt.plot(testrange,rs_df_inertia,'bx-')

plt.xlabel('Number of Clusters RobustScaler')

plt.ylabel('SSD')

plt.subplot(436)

plt.plot(testrange,pt_df_inertia,'bx-')

plt.xlabel('Number of Clusters PowerTransformer')

plt.ylabel('SSD')

plt.subplot(437)

plt.plot(testrange,qt_df_inertia,'bx-')

plt.xlabel('Number of Clusters QantileTransformer')

plt.ylabel('SSD')

plt.suptitle('Elbow Curve - SSD')

plt.show()
norm_score = mms_score = ss_score = rs_score = pt_score = qt_score = 0.0

save_norm_score = save_mms_score = save_ss_score = save_rs_score = save_pt_score = save_qt_score = 0.0

norm_cluster = mms_cluster = ss_cluster = rs_cluster = pt_cluster = qt_cluster = 0.0

for n_clusters in range(2,15):

    km = KMeans (n_clusters=n_clusters)

    norm_preds = km.fit_predict(norm_df)

    mms_preds = km.fit_predict(mms_df)

    ss_preds = km.fit_predict(ss_df)

    rs_preds = km.fit_predict(rs_df)

    pt_preds = km.fit_predict(pt_df)

    qt_preds = km.fit_predict(qt_df)

    #centers = km.cluster_centers_

    norm_score = silhouette_score(norm_df, norm_preds, metric='euclidean')

    mms_score = silhouette_score(mms_df, mms_preds, metric='euclidean')

    ss_score = silhouette_score(ss_df, ss_preds, metric='euclidean')

    rs_score = silhouette_score(rs_df, rs_preds, metric='euclidean')

    pt_score = silhouette_score(pt_df, pt_preds, metric='euclidean')

    qt_score = silhouette_score(qt_df, qt_preds, metric='euclidean')

    

    if save_norm_score < norm_score:

        save_norm_score = norm_score

        norm_cluster = n_clusters

    if save_mms_score < mms_score:

        save_mms_score = mms_score

        mms_cluster = n_clusters

    if save_ss_score < ss_score:

        save_ss_score = ss_score

        ss_cluster = n_clusters

    if save_rs_score < rs_score:

        save_rs_score = rs_score

        rs_cluster = n_clusters

    if save_pt_score < pt_score:

        save_pt_score = pt_score

        pt_cluster = n_clusters

    if save_qt_score < qt_score:

        save_qt_score = qt_score

        qt_cluster = n_clusters



print ("For normalization     optimal cluster = {}, silhouette score is {}".format(norm_cluster, norm_score))

print ("For MinMaxScaler      optimal cluster = {}, silhouette score is {}".format(mms_cluster, mms_score))

print ("For StandardScaler    optimal cluster = {}, silhouette score is {}".format(ss_cluster, ss_score))

print ("For RobustScaler      optimal cluster = {}, silhouette score is {}".format(rs_cluster, rs_score))

print ("For PowerTransform    optimal cluster = {}, silhouette score is {}".format(pt_cluster, pt_score))

print ("For QuantileTransform optimal cluster = {}, silhouette score is {}".format(qt_cluster, qt_score))
k_means = KMeans(n_clusters = norm_cluster, n_init = 12).fit(norm_df)

labels_norm = k_means.labels_

centroids_norm = k_means.cluster_centers_
k_means = KMeans(n_clusters = mms_cluster, n_init = 12).fit(mms_df)

labels_mms = k_means.labels_

centroids_mms = k_means.cluster_centers_
k_means = KMeans(n_clusters = ss_cluster, n_init = 12).fit(ss_df)

labels_ss = k_means.labels_

centroids_ss = k_means.cluster_centers_
k_means = KMeans(n_clusters = rs_cluster, n_init = 12).fit(rs_df)

labels_rs = k_means.labels_

centroids_rs = k_means.cluster_centers_
k_means = KMeans(n_clusters = pt_cluster, n_init = 12).fit(pt_df)

labels_pt = k_means.labels_

centroids_pt = k_means.cluster_centers_
k_means = KMeans(n_clusters = qt_cluster, n_init = 12).fit(qt_df)

labels_qt = k_means.labels_

centroids_qt = k_means.cluster_centers_
x['labels_norm'] = labels_norm

x['labels_mms'] = labels_mms

x['labels_ss'] = labels_ss

x['labels_rs'] = labels_rs

x['labels_pt'] = labels_pt

x['labels_qt'] = labels_qt

x.head()
x.groupby('labels_norm').mean().sort_values(by='BALANCE')
plt.scatter(x['BALANCE'], x['PURCHASES'], c=labels_norm.astype(np.float), alpha=0.8,s=50)

plt.show()