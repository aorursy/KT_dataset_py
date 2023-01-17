import seaborn as sns

import matplotlib.pyplot as plt
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
car = pd.read_csv("/kaggle/input/car-price/CarPrice_Assignment.csv")
car.shape
car
car.info()
car["price"] = car["price"].astype(int)
car.isnull().sum()
car["fueltype"].value_counts().plot.bar()

plt.show()
from sklearn import preprocessing

encoder = preprocessing.LabelEncoder().fit(car["fueltype"])

car["fueltype"] = encoder.transform(car["fueltype"])
car["fueltype"].value_counts().plot.bar()

plt.show()
car["aspiration"].value_counts().plot.bar()

plt.show()
encoder = preprocessing.LabelEncoder().fit(car["aspiration"])

car["aspiration"] = encoder.transform(car["aspiration"])
car["fueltype"].value_counts().plot.bar()

plt.show()
car["doornumber"].value_counts().plot.bar()

plt.show()
car["doornumber"] = car["doornumber"].map({"four":4,"two":2})

car["doornumber"].value_counts().plot.bar()

plt.show()
car["carbody"].value_counts().plot.bar()

plt.show()
encoder = preprocessing.LabelEncoder().fit(car["carbody"])

car["carbody"] = encoder.transform(car["carbody"])

car["carbody"].value_counts().plot.bar()

plt.show()
car["drivewheel"].value_counts().plot.bar()

plt.show()
encoder = preprocessing.LabelEncoder().fit(car["drivewheel"])

car["drivewheel"] = encoder.transform(car["drivewheel"])

car["drivewheel"].value_counts().plot.bar()

plt.show()
car["enginelocation"].value_counts().plot.bar()

plt.show()
car["enginelocation"] = car["enginelocation"].map({"front":1,"rear":2})

car["enginelocation"].value_counts().plot.bar()

plt.show()
car["enginetype"].value_counts().plot.bar()

plt.show()
encoder = preprocessing.LabelEncoder().fit(car["enginetype"])

car["enginetype"] = encoder.transform(car["enginetype"])

car["enginetype"].value_counts().plot.bar()

plt.show()
car["cylindernumber"].value_counts().plot.bar()

plt.show()
car["cylindernumber"] = car["cylindernumber"].map({"four":4,"two":2,"six":6,"five":5,"eight":8,"three":3,"twelve":12})

car["cylindernumber"].value_counts().plot.bar()

plt.show()
car["fuelsystem"].value_counts().plot.bar()

plt.show()
encoder = preprocessing.LabelEncoder().fit(car["fuelsystem"])

car["fuelsystem"] = encoder.transform(car["fuelsystem"])

car["fuelsystem"].value_counts().plot.bar()

plt.show()
car.info()
car_noname = car.drop("CarName",axis=1)

car_noname
plt.subplots(figsize=(25,25))

ax = plt.axes()

corr = car_noname.corr()

sns.heatmap(corr)
pd.set_option('display.max_columns',None)
corr
plt.scatter(car_noname["curbweight"],car_noname["price"])

plt.xlabel("curbweight")

plt.ylabel("price")
plt.scatter(car_noname["enginesize"],car_noname["price"])

plt.xlabel("enginesize")

plt.ylabel("price")
import statsmodels.formula.api as smf

car_noname.eval('hp_es = horsepower / enginesize',inplace = True)

results = smf.ols('price ~hp_es',data=car_noname).fit()

results.summary()
corr = car_noname.corr()

corr
car1 = car_noname[["price","drivewheel","enginelocation","wheelbase","carlength","carwidth","curbweight","cylindernumber",

                   "enginesize","fuelsystem","boreratio","horsepower","citympg","highwaympg"]]
car1.shape
car1.head(5)
features1 = car1.columns.drop('price')
train1 = car1

train1.shape
train1_features = train1.drop("price",axis=1)

train1_target = train1["price"]

print(train1_features.shape,train1_target.shape)
from sklearn.model_selection import train_test_split

import eli5

X_train1,X_test1,Y_train1,Y_test1 = train_test_split(train1_features,train1_target,

                                                 test_size=0.2,shuffle=True,random_state = 133)

print(X_train1.shape,Y_train1.shape,X_test1.shape,Y_test1.shape)
testfeatures1 = X_test1.sample(n=10)

testdata1 = pd.merge(testfeatures1,Y_test1,left_index=True,right_index=True)

testdata1
#RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

RFR1 = RandomForestRegressor(n_estimators=200, max_depth=3, random_state=133).fit(X_train1, Y_train1)

score1 = RFR1.score(X_test1,Y_test1)

score1
#SVR

from sklearn.svm import SVR

l_svr = SVR(kernel='linear')

l_svr.fit(X_train1,Y_train1)

l_svr.score(X_test1,Y_test1)
#KNeighborsRegressor

from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(weights="uniform")

knn.fit(X_train1,Y_train1)

knn.score(X_test1,Y_test1)
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=111)

cv_results1 = cross_val_score(estimator=RFR1, X=train1_features, y=train1_target, cv=kf, scoring='r2', n_jobs=-1).mean()

cv_results2 = cross_val_score(estimator=l_svr, X=train1_features, y=train1_target, cv=kf, scoring='r2', n_jobs=-1).mean()

cv_results3 = cross_val_score(estimator=knn, X=train1_features, y=train1_target, cv=kf, scoring='r2', n_jobs=-1).mean()

print(cv_results1,cv_results2,cv_results3)
prediction = RFR1.predict(testfeatures1)

output = pd.DataFrame({"price":testdata1["price"],"prediction":prediction})

output
from sklearn.metrics import mean_absolute_error

predicts1 = RFR1.predict(X_test1)

mae1 = mean_absolute_error(Y_test1,predicts1)

mae1
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(RFR1, random_state=123).fit(X_train1, Y_train1)

eli5.show_weights(perm, feature_names = features1.tolist(), top=30)
car2 = car_noname[["price","aspiration","drivewheel","enginelocation","wheelbase","carlength","carwidth","carheight","curbweight",

                   "cylindernumber","enginesize","fuelsystem","boreratio","horsepower","fueltype","citympg","highwaympg","hp_es"]]

features2 = car2.columns.drop("price")

train2_features = car2.drop("price",axis=1)

train2_target = car2["price"]

print(train2_features.shape,train2_target.shape)
X_train2,X_test2,Y_train2,Y_test2 = train_test_split(train2_features,train2_target,

                                                 test_size=0.2,shuffle=True,random_state = 133)

print(X_train2.shape,Y_train2.shape,X_test2.shape,Y_test2.shape)
RFR2 = RandomForestRegressor(n_estimators=200, max_depth=3, random_state=133).fit(X_train2, Y_train2)

score2 = RFR2.score(X_test2,Y_test2)

score2
l_svr = SVR(kernel='linear')

l_svr.fit(X_train2,Y_train2)

l_svr.score(X_test2,Y_test2)
knn = KNeighborsRegressor(weights="uniform")

knn.fit(X_train2,Y_train2)

knn.score(X_test2,Y_test2)
predicts2 = RFR2.predict(X_test2)

mae2 = mean_absolute_error(Y_test2,predicts2)

mae2
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(RFR2, random_state=123).fit(X_train2, Y_train2)

eli5.show_weights(perm, feature_names = features2.tolist(), top=30)
car3 = car_noname

features3 = car3.columns.drop("price")

train3_features = car3.drop("price",axis=1)

train3_target = car3["price"]

X_train3,X_test3,Y_train3,Y_test3 = train_test_split(train3_features,train3_target,

                                                 test_size=0.2,shuffle=True,random_state = 133)

print(X_train3.shape,Y_train3.shape,X_test3.shape,Y_test3.shape)
RFR3 = RandomForestRegressor(n_estimators=200, max_depth=3, random_state=133).fit(X_train3, Y_train3)

score3 = RFR3.score(X_test3,Y_test3)

score3
predicts3 = RFR3.predict(X_test3)

mae3 = mean_absolute_error(Y_test3,predicts3)

mae3
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(RFR3, random_state=123).fit(X_train3, Y_train3)

eli5.show_weights(perm, feature_names = features3.tolist(), top=30)
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=111)

cv_results1 = cross_val_score(estimator=RFR1, X=train1_features, y=train1_target, cv=kf, scoring='r2', n_jobs=-1).mean()

cv_results2 = cross_val_score(estimator=RFR2, X=train2_features, y=train2_target, cv=kf, scoring='r2', n_jobs=-1).mean()

cv_results3 = cross_val_score(estimator=RFR3, X=train3_features, y=train3_target, cv=kf, scoring='r2', n_jobs=-1).mean()

print(cv_results1,cv_results2,cv_results3)
model = {'name':['RFR1','RFR2','RFR3'],'score':[score1,score2,score3],"MAE":[mae1,mae2,mae3],"CV_Results":[cv_results1,cv_results2,

                                                                                                           cv_results3]}

model_df = pd.DataFrame(model)

model_df
from sklearn.cluster import KMeans

car1_noprice = car1.drop("price",axis=1)

km = KMeans(n_clusters=5).fit(car1_noprice)

car1_noprice['cluster'] = km.labels_

car1_noprice.sort_values('cluster')

cluster_centers = km.cluster_centers_

car1_noprice.groupby("cluster").mean()
from pandas.plotting import scatter_matrix 

centers = car1_noprice.groupby("cluster").mean().reset_index

%matplotlib inline

import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 10

colors = np.array(['red','green','blue','yellow','purple'])
scatter_matrix(car1_noprice[["curbweight","cylindernumber","enginesize","horsepower"]],

               s=50,alpha=1,c=colors[car1_noprice["cluster"]],figsize=(10,10))
corr = car1_noprice[["curbweight","cylindernumber","enginesize","horsepower"]].corr()

corr
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = fig.add_subplot(111,projection='3d')

xs = car1_noprice["curbweight"]

ys = car1_noprice["horsepower"]

zs = car1_noprice["enginesize"]

ax.scatter(xs,ys,zs,c=colors[car1_noprice["cluster"]],s=20)

ax.set_xlabel('curbweight')

ax.set_ylabel('horsepower')

ax.set_zlabel('enginesize')

plt.show()
scatter_matrix(car1_noprice[["curbweight","enginesize","horsepower","highwaympg","citympg"]],

               s=50,alpha=1,c=colors[car1_noprice["cluster"]],figsize=(10,10))
car11 = car_noname[["price","drivewheel","enginelocation","wheelbase","carlength","carwidth","curbweight","cylindernumber",

                   "enginesize","fuelsystem","boreratio","horsepower","citympg","highwaympg","hp_es"]]

km = KMeans(n_clusters=5).fit(car11)

car11['cluster'] = km.labels_

car11.sort_values('cluster')

cluster_centers = km.cluster_centers_

car11.groupby("cluster").mean()
from pandas.plotting import scatter_matrix 

centers = car1_noprice.groupby("cluster").mean().reset_index

%matplotlib inline

import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 10

colors = np.array(['red','green','blue','yellow','purple'])

plt.scatter(car11["hp_es"],car11["price"],c=colors[car11["cluster"]])

plt.xlabel("hp_es")

plt.ylabel("price")
car11.groupby("cluster").mean()