import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn
data_train = pd.read_csv("../input/atividade-regressao-PMR3508/train.csv",

        index_col=['Id'],engine='python', na_values="?")
data_train.shape
data_train.head()
data_train.info()
data_train.columns

data_train_copy = data_train.copy()
data_train.describe()
data_train.hist(bins=50,figsize=(20,15))

plt.show()
sns.set()

plt.figure(figsize=(10,8))#Figure size

plt.scatter('longitude','latitude',data=data_train)
plt.figure(figsize=(10,7))

data_train.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,

        s=data_train["population"]/100, label="population", figsize=(15,8),

        c="median_house_value", cmap=plt.get_cmap("jet"),colorbar=True,

    )

plt.legend()
data_train.columns
feat = ['median_age', 'total_rooms', 'total_bedrooms','median_income', 'median_house_value']

pd.plotting.scatter_matrix(data_train[feat],figsize=(15,8))
data_train.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.2, figsize=(10,10))
from sklearn.cluster import KMeans

X = data_train.loc[:,['latitude','longitude']]
X.head()
K_clusters = range(1,10)



kmeans = [KMeans(n_clusters=i) for i in K_clusters]



Y_axis = data_train[['latitude']]

X_axis = data_train[['longitude']]

score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]

# Visualize

plt.plot(K_clusters, score)

plt.xlabel('Number of Clusters')

plt.ylabel('Score')

plt.title('Elbow Curve')

plt.show()
kmeans = KMeans(n_clusters = 9, init ='k-means++')



kmeans.fit(X[X.columns])

X['cluster_places'] = kmeans.fit_predict(X[X.columns])

centers = kmeans.cluster_centers_ 

labels = kmeans.predict(X[X.columns[:2]]) 
X.head()
X = X['cluster_places']
data_train = data_train.merge(X, left_on='Id', right_on='Id')
data_train.head()
data_train.plot.scatter(x = 'latitude', y = 'longitude', c=labels, s=50, cmap='viridis')

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
data_train['rooms_per_household'] = data_train['total_rooms']/data_train['households']

data_train['bedrooms_per_room'] = data_train['total_bedrooms']/data_train['total_rooms']

data_train['population_per_household'] = data_train['population']/data_train['households']
data_train.head()
corr_matrix = data_train.corr()

corr_matrix['median_house_value'].sort_values(ascending=False)
data_train.columns
Y_train = data_train.pop('median_house_value')

X_train = data_train.drop(['latitude','longitude','total_rooms', 'total_bedrooms','households'], axis = 1)
from sklearn import preprocessing



names = X_train.columns



scaler = preprocessing.StandardScaler()



scaled_df = scaler.fit_transform(X_train)

X_train = pd.DataFrame(scaled_df, columns=names)
X_train.head()
def rmsle(y,y_pred):

    return np.sqrt(np.mean((np.log(np.abs(y_pred)+1) - np.log(np.abs(y)+1))**2))
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
X_train, X_test_split, Y_train, Y_test_split = train_test_split(X_train, Y_train, test_size=0.20)
from sklearn.linear_model import LinearRegression



reg = LinearRegression()

reg.fit(X_train, Y_train)



Y_pred = reg.predict(X_test_split)



lin_reg = rmsle(Y_pred, Y_test_split)

print("RMSLE:", lin_reg)
from sklearn.linear_model import RidgeCV



reg = RidgeCV(cv=10)

reg.fit(X_train, Y_train)



Y_pred = reg.predict(X_test_split)



rid_reg = rmsle(Y_pred, Y_test_split)

print("RMSLE:", rid_reg)
from sklearn.neighbors import KNeighborsRegressor



bestScore = 100



for i in range (1,51):

  knn = KNeighborsRegressor(n_neighbors= i)

  knn.fit(X_train, Y_train)

  Y_pred = knn.predict(X_test_split)

  knn_reg = rmsle(Y_pred, Y_test_split)



  if knn_reg < bestScore:

    bestScore = knn_reg

    knnKey = i



print("Acurácia com cross validation:", bestScore.mean())

print("Melhor K:",knnKey)
reg = KNeighborsRegressor(n_neighbors= 21)

reg.fit(X_train, Y_train)



Y_pred = reg.predict(X_test_split)



rid_reg = rmsle(Y_pred, Y_test_split)

print("RMSLE:", rid_reg)
data_test = pd.read_csv("../input/atividade-regressao-PMR3508/test.csv",

        index_col=['Id'],engine='python', na_values="?")
data_test.columns
###Tratando os locais###



X = data_test.loc[:,['latitude','longitude']]



kmeans = KMeans(n_clusters = 10, init ='k-means++')



kmeans.fit(X[X.columns])

X['cluster_places'] = kmeans.fit_predict(X[X.columns])

centers = kmeans.cluster_centers_ 

labels = kmeans.predict(X[X.columns[:2]]) 



X = X['cluster_places']

data_test['cluster_places'] = X







###Esclarecendo os dados###



data_test['rooms_per_household'] = data_test['total_rooms']/data_test['households']

data_test['bedrooms_per_room'] = data_test['total_bedrooms']/data_test['total_rooms']

data_test['population_per_household'] = data_test['population']/data_test['households']





###Normalizando os dados###



data_test = data_test.drop(['latitude','longitude','total_rooms', 'total_bedrooms','households'], axis = 1)





from sklearn import preprocessing



names = data_test.columns



scaler = preprocessing.StandardScaler()



scaled_df = scaler.fit_transform(data_test)

X_test = pd.DataFrame(scaled_df, columns=names)
X_test.head()
reg = KNeighborsRegressor(n_neighbors= 21)

reg.fit(X_test_split, Y_test_split)



predictions = reg.predict(X_test)

submission = pd.DataFrame()

submission[0] = data_test.index

submission[1] = predictions

submission.columns = ['Id','median_house_value']

submission.to_csv('submission.csv',index=False)