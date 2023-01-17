import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import KMeans

from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as sch
data = pd.read_csv( '/kaggle/input/crime.csv', encoding='latin-1')



data.head()
data.shape
data.describe()
data.info()
for column in data.columns : 

    print('Length of unique data for {0} is {1} '.format(column , len(data[column].unique())))

    
data.drop(['INCIDENT_NUMBER' , 'OCCURRED_ON_DATE' ,'STREET' , 'OFFENSE_CODE' , 'REPORTING_AREA','Location'],axis=1, inplace=True)



data.head()
for column in data.columns : 

    print('Length of unique data for {0} is {1} '.format(column , len(data[column].unique())))
data['Lat code'] = np.round(data['Lat'],2)

data['Long code'] = np.round(data['Long'],2)



data.drop(['Lat','Long'],axis=1, inplace=True)

data.head()
for column in data.columns : 

    print('Length of unique data for {0} is {1} '.format(column , len(data[column].unique())))
data.info()
data['SHOOTING'].unique()
data['shooting code'] = np.where(data['SHOOTING']=='Y' , 1 , 0)
print( 'Shooting Percentage  is {} %'.format(round((data['shooting code'].sum() / data.shape[0]) * 100,2)))



data.drop(['SHOOTING'],axis=1, inplace=True)





data.info()
data['DISTRICT'].unique()
data.DISTRICT.fillna('none', inplace=True)
data['DISTRICT'].unique()
data['UCR_PART'].unique()
data.UCR_PART.fillna('none', inplace=True)
data['UCR_PART'].unique()
lat_mean = data['Lat code'].sum() / 299074

long_mean =  data['Long code'].sum() / 299074

print(round(lat_mean,2))

print(round(long_mean,2))
data['Lat code'].fillna(round(lat_mean,2), inplace=True)

data['Long code'].fillna(round(long_mean,2), inplace=True)
data.info()
data.head()
enc  = LabelEncoder()

enc.fit(data['OFFENSE_CODE_GROUP'])

data['Offense Code'] = enc.transform(data['OFFENSE_CODE_GROUP'])

data.drop(['OFFENSE_CODE_GROUP'],axis=1, inplace=True)
data.head()
enc  = LabelEncoder()

enc.fit(data['OFFENSE_DESCRIPTION'])

data['Offense Desc Code'] = enc.transform(data['OFFENSE_DESCRIPTION'])

data.drop(['OFFENSE_DESCRIPTION'],axis=1, inplace=True)

data.head()
enc  = LabelEncoder()

enc.fit(data['DISTRICT'])

data['District Code'] = enc.transform(data['DISTRICT'])

data.drop(['DISTRICT'],axis=1, inplace=True)

data.head()
enc  = LabelEncoder()

enc.fit(data['DAY_OF_WEEK'])

data['Day Code'] = enc.transform(data['DAY_OF_WEEK'])

data.drop(['DAY_OF_WEEK'],axis=1, inplace=True)

data.head()
enc  = LabelEncoder()

enc.fit(data['UCR_PART'])

data['UCR Code'] = enc.transform(data['UCR_PART'])

data.drop(['UCR_PART'],axis=1, inplace=True)

data.head()
data.describe()
data.info()
X_train = data[:250000]

X_test = data[250000:]
print('X Train Shape is {}'.format(X_train.shape))

print('X Test Shape is {}'.format(X_test.shape))
KMeansModel = KMeans(n_clusters=5,init='k-means++', #also can be random

                     random_state=33,algorithm= 'auto') # also can be full or elkan

KMeansModel.fit(X_train)
print('KMeansModel centers are : ' , KMeansModel.cluster_centers_)

print('---------------------------------------------------')

print('KMeansModel labels are : ' , KMeansModel.labels_[:20])

print('---------------------------------------------------')

print('KMeansModel intertia is : ' , KMeansModel.inertia_)

print('---------------------------------------------------')

print('KMeansModel No. of iteration is : ' , KMeansModel.n_iter_)

print('---------------------------------------------------')

#Calculating Prediction

y_pred = KMeansModel.predict(X_test)

print('Predicted Value for KMeansModel is : ' , y_pred[:10])
NearestNeighborsModel = NearestNeighbors(n_neighbors=4,radius=1.0,algorithm='auto')#it can be:ball_tree,kd_tree,brute

NearestNeighborsModel.fit(X_train)
#Calculating Details

print('NearestNeighborsModel Train kneighbors are : ' , NearestNeighborsModel.kneighbors(X_train[: 5]))

print('----------------------------------------------------')

print('NearestNeighborsModel Train radius kneighbors are : ' , NearestNeighborsModel.radius_neighbors(X_train[:  1]))

print('----------------------------------------------------')

print('NearestNeighborsModel Test kneighbors are : ' , NearestNeighborsModel.kneighbors(X_test[: 5]))

print('----------------------------------------------------')

print('NearestNeighborsModel Test  radius kneighbors are : ' , NearestNeighborsModel.radius_neighbors(X_test[:  1]))

print('----------------------------------------------------')
AggClusteringModel = AgglomerativeClustering(n_clusters=5,affinity='euclidean',# it can be l1,l2,manhattan,cosine,precomputed

                                             linkage='ward')# it can be complete,average,single



y_pred_train = AggClusteringModel.fit_predict(X_train[:1000])

y_pred_test = AggClusteringModel.fit_predict(X_test[:1000])

#draw the Hierarchical graph for Training set

dendrogram = sch.dendrogram(sch.linkage(X_train[:30], method = 'ward'))# it can be complete,average,single

plt.title('Training Set')

plt.xlabel('X Values')

plt.ylabel('Distances')

plt.show()



#draw the Hierarchical graph for Test set

dendrogram = sch.dendrogram(sch.linkage(X_test[:30], method = 'ward'))# it can be complete,average,single

plt.title('Test Set')

plt.xlabel('X Value')

plt.ylabel('Distances')

plt.show()


