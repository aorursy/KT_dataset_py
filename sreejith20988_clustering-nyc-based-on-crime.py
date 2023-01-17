



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import requests



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import datetime

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import seaborn as sns

from matplotlib.pyplot import figure

from datetime import datetime



from sklearn.preprocessing import StandardScaler

import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering 

from sklearn.cluster import KMeans

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn import metrics

from sklearn.metrics import pairwise_distances

from sklearn.metrics import silhouette_score

from sklearn.metrics import silhouette_samples

from sklearn.cluster import DBSCAN

from sklearn.cluster import SpectralClustering

# Any results you write to the current directory are saved as output.



#file = '../input/nyc-crime/NYPD_Complaint_Data_Historic.csv'

#crime_original = pd.read_csv(file) 
#print(crime_original.shape)

#crime_sub = crime_original.sample(n = 25000, random_state = 99)

#print(crime_sub.shape)

#crime_sub.to_csv('nyc_crime_sample.csv')
file = '../input/sample/nyc_crime_sample.csv'

crime_sub = pd.read_csv(file)

print(crime_sub.shape)
print(crime_sub.columns)
print(crime_sub.describe())
print(crime_sub.isna().any())



# Complaint From Date & Time should be enough in our case 

# Reporting Date is not needed for the same reasons

# we can work with Classification code for now and then impute the offense description using the same later

# Level of offense should be useful; we should be able to impute it later

# Borough Name - Only 16 missing lines out of 10k. I think this is fine for now, imputation possible from Long/Lat proabably. 



# Age Group / Race / Sex - For this analysis, we can for now discard this as we are more interested in location wise and timewise patterns than people commiting 

# Long / Lat missing for 527; I think that's fine for now. We will stick to Lat Long instead of the X-Y coordinate system

# TRANSIT_DISTRICT & STATION NAME is mostly missing; doesn't seem to be much useful



# VICTIM AGE is missing but we can probably use SEX to check if there is more crimes against women

relevant_fields = ['CMPLNT_NUM','CMPLNT_FR_DT','CMPLNT_FR_TM','ADDR_PCT_CD','KY_CD','OFNS_DESC','PD_CD','PD_DESC','LAW_CAT_CD','BORO_NM','PREM_TYP_DESC',

                  'JURIS_DESC','Latitude','Longitude','PATROL_BORO','VIC_AGE_GROUP','VIC_SEX']



nyc_crime = crime_sub.loc[:,relevant_fields]

nyc_crime.shape
print(nyc_crime.head(5))
nyc_crime.tail(5)
nyc_crime.dtypes
nyc_crime['CMPLNT_FR_DT'] = pd.to_datetime(nyc_crime['CMPLNT_FR_DT'], errors='coerce')

Month = nyc_crime['CMPLNT_FR_DT'].dt.month

print(Month)


plt.hist(nyc_crime['CMPLNT_FR_DT'].dt.month)

#From the histogram we can see that the crime is maximum for the month if January and December.
plt.hist(pd.to_datetime(nyc_crime['CMPLNT_FR_TM']).dt.hour)

#From the histogram we can observe that most crimes take place at 3pm and 12 am.
sns.catplot(data = nyc_crime, kind = 'count', x = 'ADDR_PCT_CD', aspect = 5)
nyc_crime.columns
lat = nyc_crime['Latitude'].dropna().values

long = nyc_crime['Longitude'].dropna().values



plt.scatter(long, lat)

plt.show
nyc_geo = np.vstack((long, lat)).T



geo_cluster_model = KMeans(n_clusters = 4)

geo_cluster_model.fit(nyc_geo)



geo_clusters = geo_cluster_model.labels_

geo_centroids = geo_cluster_model.cluster_centers_



plt.scatter(long, lat, c = geo_clusters, alpha = 0.8)

plt.show()
precinct_pivot = nyc_crime.pivot_table(values = 'CMPLNT_NUM', index = 'ADDR_PCT_CD', aggfunc = pd.Series.nunique)

precinct_pivot.sort_values("ADDR_PCT_CD")

precinct_pivot = precinct_pivot.reset_index()
precinct_geo = nyc_crime[['ADDR_PCT_CD', 'Longitude', 'Latitude']].sort_values('ADDR_PCT_CD').dropna().drop_duplicates('ADDR_PCT_CD')

precinct_geo = precinct_geo.reset_index(drop = True)
nyc_cc = precinct_pivot['CMPLNT_NUM'].values

precinct_lat = precinct_geo['Latitude'].values

precinct_long = precinct_geo['Longitude'].values



nyc_geo_cc = np.vstack((precinct_long, precinct_lat, nyc_cc)).T
scaler = StandardScaler()

scaler.fit(nyc_geo_cc)

nyc_geo_cc_sampled = scaler.transform(nyc_geo_cc)
geo_cluster_cc_model = KMeans(n_clusters = 3)

geo_cluster_cc_model.fit(nyc_geo_cc_sampled)



geo_cc_clusters = geo_cluster_cc_model.labels_

geo_cc_centroids = geo_cluster_cc_model.cluster_centers_



plt.scatter(precinct_long, precinct_lat, c = geo_cc_clusters)

plt.show()
precinct_level = pd.crosstab(nyc_crime['ADDR_PCT_CD'], nyc_crime['LAW_CAT_CD'])

precinct_level = precinct_level.reset_index()



nyc_felony = precinct_level['FELONY'].values

nyc_mis = precinct_level['MISDEMEANOR'].values

nyc_vio = precinct_level['VIOLATION'] .values

nyc_level = np.vstack((nyc_felony, nyc_mis, nyc_vio)).T
scaler = StandardScaler()

nyc_level_sampled = scaler.fit_transform(nyc_level)



precinct_level_cluster_model = KMeans(n_clusters = 3)

precinct_level_cluster_model.fit(nyc_level_sampled)



precinct_level_clusters = precinct_level_cluster_model.labels_

precinct_level_centroids = precinct_level_cluster_model.cluster_centers_



plt.scatter(precinct_long, precinct_lat, c = precinct_level_clusters)

plt.show()
precinct_vic_sex = pd.crosstab(nyc_crime['ADDR_PCT_CD'], nyc_crime['VIC_SEX'])

precinct_vic_sex = precinct_vic_sex.reset_index()



nyc_males = precinct_vic_sex['M'].values

nyc_females = precinct_vic_sex['F'].values

nyc_business = precinct_vic_sex['D'] .values

nyc_vic_sex = np.vstack((nyc_males, nyc_females, nyc_business)).T
scaler = StandardScaler()

nyc_vic_sex_sampled = scaler.fit_transform(nyc_vic_sex)



precinct_vic_sex_cluster_model = KMeans(n_clusters = 3)

precinct_vic_sex_cluster_model.fit(nyc_vic_sex_sampled)



precinct_vic_sex_clusters = precinct_vic_sex_cluster_model.labels_

precinct_vic_sex_centroids = precinct_vic_sex_cluster_model.cluster_centers_



plt.scatter(precinct_long, precinct_lat, c = precinct_vic_sex_clusters)

plt.show()
precinct_vic_age = pd.crosstab(nyc_crime['ADDR_PCT_CD'], nyc_crime['VIC_AGE_GROUP'])

precinct_vic_age = precinct_vic_age.reset_index()



nyc_18_24 = precinct_vic_age['18-24'].values

nyc_25_44 = precinct_vic_age['25-44'].values

nyc_45_64 = precinct_vic_age['45-64'] .values

nyc_lt_18 = precinct_vic_age['<18'] .values

nyc_vic_age = np.vstack((nyc_18_24, nyc_25_44, nyc_45_64, nyc_lt_18)).T
precinct_vic_age_cluster_model = KMeans(n_clusters = 3)

precinct_vic_age_cluster_model.fit(nyc_vic_age)



precinct_vic_age_clusters = precinct_vic_age_cluster_model.labels_

precinct_vic_age_centroids = precinct_vic_age_cluster_model.cluster_centers_



plt.scatter(precinct_long, precinct_lat, c = precinct_vic_age_clusters)

plt.show()
nyc_X = np.vstack((precinct_long, precinct_lat, nyc_felony, nyc_mis, nyc_vio, nyc_18_24, nyc_25_44, nyc_45_64, nyc_lt_18,

                   nyc_males, nyc_females, nyc_business)).T
scaler = StandardScaler()

nyc_X = scaler.fit_transform(nyc_X)



precinct_cluster_model = KMeans(n_clusters = 3)

precinct_cluster_model.fit(nyc_X)



precinct_clusters = precinct_cluster_model.labels_

precinct_vic_age_centroids = precinct_cluster_model.cluster_centers_



plt.scatter(precinct_long, precinct_lat, c = precinct_clusters)

plt.show()
crime_tm = nyc_crime[["CMPLNT_NUM",'ADDR_PCT_CD',"CMPLNT_FR_DT","CMPLNT_FR_TM"]]

crime_tm['CMPLNT_HOUR']= pd.to_datetime(crime_tm['CMPLNT_FR_DT']).dt.month

crime_tm['CMPLNT_MONTH'] = pd.to_datetime(crime_tm['CMPLNT_FR_TM']).dt.hour





nyc_crime_tm = crime_tm.pivot_table(values = "CMPLNT_NUM", index = ['CMPLNT_MONTH','CMPLNT_HOUR'], aggfunc = pd.Series.nunique)

nyc_crime_tm = nyc_crime_tm.sort_values(['CMPLNT_MONTH','CMPLNT_HOUR']).reset_index()



nyc_crime_tm_cc = nyc_crime_tm['CMPLNT_NUM'].values.reshape(1,-1).T
dendrogram = sch.dendrogram(sch.linkage(nyc_crime_tm["CMPLNT_NUM"].values.reshape(1,-1).T, method  = "average"))

#plt.figure(figsize=(20,10))

plt.title('Dendrogram')

plt.xlabel('Month & Time')

plt.ylabel('cases')

plt.show()
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage ='average')

hc_clusters = hc.fit_predict(nyc_crime_tm)



nyc_crime_tm['Class'] = hc_clusters



plt.figure(figsize=(15,8))

ax= sns.heatmap(nyc_crime_tm.pivot("CMPLNT_HOUR","CMPLNT_MONTH","Class"))

nyc_crime_tm_class = pd.merge(crime_tm, nyc_crime_tm,on = ['CMPLNT_MONTH', 'CMPLNT_HOUR'])
precinct_time_class = nyc_crime_tm_class.pivot_table(index = 'ADDR_PCT_CD', values = 'Class', aggfunc = pd.Series.nunique)

precinct_time_class = precinct_time_class.reset_index()



nyc_tc = precinct_time_class['Class'].values
precinct_tc_cluster_model = KMeans(n_clusters = 3)

precinct_tc_cluster_model.fit(nyc_tc.reshape(1,-1).T)



precinct_tc_clusters = precinct_tc_cluster_model.labels_

precinct_tc_centroids = precinct_tc_cluster_model.cluster_centers_



plt.scatter(precinct_long, precinct_lat, c = precinct_tc_clusters)

plt.show()
nyc_X = np.vstack((precinct_long, precinct_lat, nyc_felony, nyc_mis, nyc_vio, nyc_18_24, nyc_25_44, nyc_45_64, nyc_lt_18, 

                   nyc_males, nyc_females, nyc_business, nyc_tc)).T
scaler = StandardScaler()

nyc_X = scaler.fit_transform(nyc_X)



precinct_cluster_model = KMeans(n_clusters = 3)

precinct_cluster_model.fit(nyc_X)



precinct_clusters = precinct_cluster_model.labels_

precinct_vic_age_centroids = precinct_cluster_model.cluster_centers_



plt.scatter(precinct_long, precinct_lat, c = precinct_clusters)

plt.show()
ks = range(1,6)

inertias = []



for k in ks:

    model = KMeans(n_clusters = k)

    model.fit(nyc_X)

    inertias.append(model.inertia_)

    

plt.plot(ks, inertias, '-o')

plt.xlabel('Number of clusters')

plt.ylabel('Inertia')

plt.xticks(ks)

plt.show()
model = TSNE(learning_rate=100)

transformed = model.fit_transform(nyc_X)



xs = transformed[:, 0]

ys = transformed[:, 1]



plt.scatter(xs, ys, c = precinct_clusters)

plt.show()
pca = PCA()

pca_components = pca.fit_transform(nyc_X)

#print(pca.explained_variance_ratio_)



plt.bar(range(pca.n_components_), pca.explained_variance_ratio_)

plt.xticks(range(pca.n_components_))

plt.ylabel('Variance')

plt.xlabel('Components')

plt.show()



PCA_components = pd.DataFrame(pca_components)
plt.scatter(PCA_components[0], PCA_components[1])

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

plt.show()
ks = range(1, 10)

inertias = []

for k in ks:

    

    model = KMeans(n_clusters=k)

    model.fit(PCA_components.iloc[:,:3])

    

    inertias.append(model.inertia_)

    

plt.plot(ks, inertias, '-o', color='black')

plt.xlabel('Number of clusters')

plt.ylabel('Inertia')

plt.xticks(ks)

plt.show()
pca_based_model = KMeans(n_clusters=3)

pca_based_model.fit(PCA_components.iloc[:,:3])

pca_clusters = pca_based_model.labels_



plt.scatter(PCA_components[0], PCA_components[1], c = pca_clusters, cmap = plt.cm.Paired)

plt.show()



silhouette_score(PCA_components.iloc[:,:3], pca_clusters, metric='euclidean')
pca_based_model = KMeans(n_clusters=5)

X = PCA_components.iloc[:,:3]

pca_based_model.fit_predict(X)

labels = pca_based_model.labels_



print(silhouette_score(X, labels, metric='euclidean'))
n_clusters = 5

def silhouette(X,labels):

    n_clusters = np.size(np.unique(labels));

    silhouette_values = silhouette_samples(X, labels)

    y_lower = 1

    

    for i in range(n_clusters):

        ith_cluster_silhouette_values = silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]

        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)

        plt.fill_betweenx(np.arange(y_lower, y_upper),0, ith_cluster_silhouette_values,facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle

        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        #Compute the new y_lower for next cluster

        y_lower = y_upper + 10 # 10 for the 0 samples

    

    plt.title("Silhouette plot for the various clusters.")

    plt.xlabel("Silhouette coefficient values")

#    plt.ylabel("Cluster label")

    plt.show()

    

silhouette(X,labels)
precinct_cluster_model = KMeans(n_clusters = 5)

precinct_cluster_model.fit(nyc_X)



labels = precinct_cluster_model.labels_



silhouette(nyc_X,labels)
dendrogram = sch.dendrogram(sch.linkage(nyc_X, method  = "average"))

#plt.figure(figsize=(20,10))

plt.title('Dendrogram')

plt.xlabel('Variables')

plt.ylabel('Distance')

plt.show()
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage ='average')

hc_clusters = hc.fit_predict(nyc_X)



plt.scatter(xs, ys, c = hc_clusters)

plt.show()
dbscan_model = DBSCAN(eps = 0.05, min_samples = 5, metric ='euclidean')

dbscan_model.fit(nyc_X)

dbscan_clusters = dbscan_model.labels_



plt.scatter(xs, ys, c = dbscan_clusters)

plt.show()
precinct_spectral = SpectralClustering(n_clusters = 4)

precinct_spectral_clusters = precinct_spectral.fit_predict(nyc_X)



plt.scatter(xs, ys, c = precinct_spectral_clusters)

plt.show()
precinct_cluster_model = KMeans(n_clusters = 3)

precinct_cluster_model.fit(nyc_X)



labels = precinct_cluster_model.labels_
precinct = precinct_geo['ADDR_PCT_CD'].values



nyc_X = np.vstack((precinct, precinct_long, precinct_lat, nyc_felony, nyc_mis, nyc_vio, nyc_18_24, nyc_25_44, nyc_45_64, nyc_lt_18, 

                   nyc_males, nyc_females, nyc_business, nyc_tc, labels)).T

nyc_X_pd = pd.DataFrame(nyc_X, columns = ['Precinct', 'Longitude','Latitude', 'Felony', 'Misdemeanour', 'Violation', '18-24', '25-44', '45-64', '<18', 'Male', 'Female', 'Business', 'Time Class', 'Label'])
nyc_X_pd.pivot_table(index = 'Label', values = ['18-24', '25-44', '45-64', '<18'], aggfunc = np.mean)
nyc_X_pd.pivot_table(index = 'Label', values = ['Felony', 'Misdemeanour', 'Violation', ], aggfunc = np.mean)
nyc_X_pd.pivot_table(index = 'Label', values = ['Male', 'Female', 'Business'], aggfunc = np.mean)
plt.scatter(precinct_long, precinct_lat, c = labels)

plt.show()