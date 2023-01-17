# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #visualization

dataset = pd.read_csv('../input/listings_summary.csv')
# Calculating distance using latitude and longitude

from geopy.distance import great_circle

def distance_from_berlin(lat,lon):

    center=(52.5027778, 13.404166666666667)

    location=(lat,lon)

    return great_circle(center,location).km

dataset['distance'] = dataset.apply(lambda x: distance_from_berlin(x.latitude, x.longitude), axis=1)



features=["distance","accommodates","bathrooms","bedrooms","beds","minimum_nights","amenities","guests_included","number_of_reviews","availability_365","number_of_reviews","review_scores_rating","review_scores_accuracy","review_scores_cleanliness","review_scores_checkin","review_scores_communication","review_scores_location","review_scores_value"]

dataset.head(1)
dataset[features].info()
#Filling the missing data

dataset.bathrooms= dataset.bathrooms.fillna(1)

dataset.bedrooms= dataset.bedrooms.fillna(1)

dataset.beds= dataset.beds.fillna(1)

dataset.review_scores_rating= dataset.review_scores_rating.fillna(0)

dataset.review_scores_accuracy= dataset.review_scores_accuracy.fillna(0)

dataset.review_scores_cleanliness= dataset.review_scores_cleanliness.fillna(0)

dataset.review_scores_checkin= dataset.review_scores_checkin.fillna(0)

dataset.review_scores_communication= dataset.review_scores_communication.fillna(0)

dataset.review_scores_location= dataset.review_scores_location.fillna(0)

dataset.review_scores_value= dataset.review_scores_value.fillna(0)



#Calculating the number of amenities.

dataset.amenities= dataset.amenities.str.count(',')
# Adding Categorical variables

dataset["bed_type"]= pd.Categorical(dataset["bed_type"])

dataset['room_type'] = pd.Categorical(dataset['room_type'])

Dummies1 = pd.get_dummies(dataset['room_type'], prefix = 'category')

Dummies2 = pd.get_dummies(dataset['bed_type'], prefix = 'category')

dataset1= pd.concat([Dummies1,Dummies2], axis= 1)

X= pd.concat([dataset1,dataset[features]], axis=1)
dataset.price =  list(map(lambda x: float(str(x).replace(',','').replace('$','')),dataset.price)) 

y= dataset["price"]

# Final Features

X.info()
from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test= train_test_split(X,y,test_size=.25,random_state=0)

# Normalize data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test  = sc.transform(X_test)

# #PCA selection

# from sklearn.decomposition import PCA

# pca = PCA(n_components= 4)

# X_train = pca.fit_transform(X_train)

# X_test= pca.transform(X_test)

# explained_variance= pca.explained_variance_ratio_

# print(explained_variance)
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()

from sklearn.model_selection import GridSearchCV

param_grid = [{'max_depth': np.arange(1, 10),

              'min_samples_leaf': [1, 5, 10, 20, 50, 100],

              'min_weight_fraction_leaf': [0.0,0.1,0.3,0.5],

              'random_state':[1,3,4,7]}]

gridCV = GridSearchCV(estimator=regressor, param_grid=param_grid,cv=10)

gridCV = gridCV.fit(X_train, y_train)

print(gridCV.best_score_)

print(gridCV.best_params_)


md = gridCV.best_params_["max_depth"]

msl  = gridCV.best_params_["min_samples_leaf"]

mwfl = gridCV.best_params_["min_weight_fraction_leaf"]

rs = gridCV.best_params_["random_state"]
regressor = DecisionTreeRegressor(max_depth=md,min_samples_leaf=msl,min_weight_fraction_leaf=mwfl,random_state=rs)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
# from sklearn.metrics import mean_squared_error, r2_score

# RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

# print(f"RMSE: {round(RMSE, 4)}")



# r2 = r2_score(y_test, y_pred)

# print(f"r2: {round(r2, 4)}")
