import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from sklearn.preprocessing import PolynomialFeatures



import warnings

warnings.simplefilter('ignore')
path = '../input/'

rides = pd.read_csv(path + 'train.csv')

rides.head()
rides['datetime'].values[:5]
from datetime import datetime



# We extract 'month', 'hour', 'weekday' from the 'datetime' column

def extract_from_datetime(rides):

    rides["date"] = rides["datetime"].apply(lambda x : x.split()[0])

    rides["hour"] = rides["datetime"].apply(lambda x : x.split()[1].split(":")[0])

    rides["weekday"] = rides["date"].apply(lambda dateString : 

                            datetime.strptime(dateString,"%Y-%m-%d").weekday())

    rides["month"] = rides["date"].apply(lambda dateString : 

                            datetime.strptime(dateString,"%Y-%m-%d").month)

    return rides



# We one-hot encode the categorical features

def one_hot_encoding(rides):

    dummy_fields = ['season', 'weather', 'month', 'hour', 'weekday']

    for each in dummy_fields:

        dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)

        rides = pd.concat([rides, dummies], axis=1)

    return rides



# We drop the columns that are redundant now

def drop_features(rides):

    features_to_drop = ['datetime', 'date', 

                        'month', 'hour', 'weekday', 

                        'season', 'weather']



    rides = rides.drop(features_to_drop, axis=1)

    return rides



# Now we aggregate all the above defined functions inside a function

def feature_engineering(rides):

    rides = extract_from_datetime(rides)

    rides = one_hot_encoding(rides)

    rides = drop_features(rides)

    return rides



# Now we apply all the above defined functions to the rides dataframe

rides = feature_engineering(rides)
rides.head()
rides.columns
rides.shape
quantitative_features = ['temp', 'atemp', 'humidity', 'windspeed']



# Store scalings in a dictionary so we can convert back later

scaled_features = {}

for each in quantitative_features:

    mean, std = rides[each].mean(), rides[each].std()

    scaled_features[each] = [mean, std]

    rides.loc[:, each] = (rides[each] - mean)/std
# Next we extract the target variables from the dataframe

target = rides[['casual', 'registered', 'count']]

target = np.log1p(target)

rides = rides.drop(['casual', 'registered', 'count'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(rides, target,

                                        random_state = 0)
poly2 = PolynomialFeatures(degree=2)

X_poly2 = poly2.fit_transform(rides)

X_train_poly2, X_valid_poly2, y_train_poly2, y_valid_poly2 = train_test_split(X_poly2, 

                                                    target['count'], random_state = 0)
poly3 = PolynomialFeatures(degree=3)

X_poly3 = poly3.fit_transform(rides)

X_train_poly3, X_valid_poly3, y_train_poly3, y_valid_poly3 = train_test_split(X_poly3, 

                                                    target['count'], random_state = 0)
polyreg3 = LinearRegression().fit(X_train_poly3, y_train_poly3)



polyreg3_train_score = polyreg3.score(X_train_poly3, y_train_poly3)

polyreg3_valid_score = polyreg3.score(X_valid_poly3, y_valid_poly3)



print('R-squared score (training): {:.3f}'

     .format(polyreg3_train_score))

print('R-squared score (validation): {:.3f}'

     .format(polyreg3_valid_score))
def get_rmse(reg):

    y_pred_train = reg.predict(X_train_poly)

    train_rmse = np.sqrt(mean_squared_error(y_train_poly, y_pred_train))

    y_pred_valid = reg.predict(X_valid_poly)

    valid_rmse = np.sqrt(mean_squared_error(y_valid_poly, y_pred_valid))

    return train_rmse, valid_rmse