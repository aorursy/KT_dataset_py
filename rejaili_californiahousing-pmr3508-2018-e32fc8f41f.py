import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
train_original = pd.read_csv("../input/train.csv",header=0, index_col=0)
test = pd.read_csv("../input/test.csv",header=0, index_col=0)
print(train_original.shape)
train_original.head()
train_original['pop_per_bedroom']=train_original.population/train_original.total_bedrooms
train_original['room_per_household']=train_original.total_rooms/train_original.households
train_original['persons_per_household']=train_original.population/train_original.households
train_original['income_per_household']=train_original.median_income*train_original.population/train_original.households
test['pop_per_bedroom']=test.population/test.total_bedrooms
test['room_per_household']=test.total_rooms/test.households
test['persons_per_household']=test.population/test.households
test['income_per_household']=test.median_income*test.population/test.households
train = train_original.iloc[:,[0,1,2,6,7,9,10,11,12,8]]
test = test.iloc[:,[0,1,2,6,7,8,9,10,11]]
train.head()
plt.hist(np.hstack(train.median_house_value.values))
centroids = []
for i in range(10):
    centroids = centroids + ([[np.pi*np.mean(train.loc[(train['median_house_value']>=i*50000)&(train['median_house_value']<(i+1)*50000)].latitude)/180,
                           np.pi*np.mean(train.loc[(train['median_house_value']>=i*50000)&(train['median_house_value']<(i+1)*50000)].longitude)/180]])
coords = np.fliplr(train.iloc[:,0:2].values*np.pi/180)
from sklearn.neighbors import DistanceMetric
dist = DistanceMetric.get_metric('haversine')
dists = 6371*dist.pairwise(centroids+coords.tolist())[:10,10:]
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
X = train.iloc[:,2:-1]
for i in range(10):
    X['dist_'+str(i)]=dists[i]
y = train.iloc[:,-1]
scores = []
for i in range(5,20):
    neigh = KNeighborsRegressor(i,p=1)
    scores.append(cross_val_score(neigh, X, np.log(y), cv=10, scoring='neg_mean_squared_log_error'))
print("KNN Regressor:")
print(np.amin([np.mean(scores[i]) for i in range(len(scores))]))
print(np.argmin([np.mean(scores[i]) for i in range(len(scores))])+5)
print(np.std(scores[np.argmin([np.mean(scores[i]) for i in range(len(scores))])]))
clf = linear_model.Lasso(.5)
scores = cross_val_score(clf, X, np.log(y), cv=10, scoring='neg_mean_squared_log_error')
print("Lasso Regressor:")
print(np.mean(scores))
print(np.std(scores))
clf = linear_model.Ridge()
scores = cross_val_score(clf, X, np.log(y), cv=10, scoring='neg_mean_squared_log_error')
print("Ridge Regressor:")
print(np.mean(scores))
print(np.std(scores))
test.head()
coords = np.fliplr(test.iloc[:,0:2].values*np.pi/180)
dist = DistanceMetric.get_metric('haversine')
dists = 6371*dist.pairwise(centroids+coords.tolist())[:10,10:]
X_test = test.iloc[:,2:]
for i in range(10):
    X_test['dist_'+str(i)]=dists[i]
prediction = pd.read_csv("../input/sample_sub_1.csv",header=0, index_col=0)
prediction.head()
clf = KNeighborsRegressor(p=1)
clf.fit(X,np.log(y))
prediction.median_house_value=np.exp(clf.predict(X_test))
#prediction.loc[prediction.median_house_value<=0]=np.amin(train.median_house_value)
prediction.to_csv("submission.csv")
