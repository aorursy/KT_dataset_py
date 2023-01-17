# Library required for failed attempt

# !pip install dirty_cat
# Import libraries in a separate cell for autocompletion

import numpy as np

import pandas as pd

import scipy

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction import DictVectorizer

from sklearn.feature_selection import mutual_info_classif

from sklearn.feature_selection import f_classif

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error

# import matplotlib.pyplot as plt



import os
# Read all the data to memory

raw_data = pd.read_csv('/kaggle/input/craigslistVehiclesFull.csv')
# Drop URLs since we are not going to access them anyway in this project

df = raw_data.drop(['url', 'image_url'], axis=1)

# Also drops Vehicle Identification Number.

# Because while it is very useful in real-life, only 22.7% of the entries have it

# in our dataset (LA only). Constructing the parser and map it to corresponding fields

# can be time-consuming and considered out of scope for this midterm.

# However, it would be interesting to see if having a VIN listed will impact the price

df.vin = df.vin.isna()

# Select the rows that have `city` equals `losangeles`

df = df[df.city == 'losangeles']

# Drop columns related to geolocation, since the region is now fixed

df.drop(list(df.filter(regex='state|county|city')), axis=1, inplace=True)

# Reset the index for easier processing

df.reset_index(drop=True, inplace=True)
cor = df.dropna().corr()

cor_target = abs(cor['price'])

cor_target.sort_values(ascending=False)
# Drop the badly performed numerical columns

df.drop(['weather', 'lat'], axis=1, inplace=True)
# Temoprarily drop NaN for feature selection

_df = df.dropna()

Xtrain = _df.drop(['price'], axis=1).T.to_dict().values()

Ytrain = _df['price']

# Vectorize the categorical values

dv = DictVectorizer()

dv.fit(Xtrain)



X_vec = dv.transform(Xtrain)
feature_scores = mutual_info_classif(X_vec, Ytrain)



for score, fname in sorted(zip(feature_scores, dv.get_feature_names()), reverse=True)[:10]:

    print(fname, score)
feature_scores = f_classif(X_vec, Ytrain)[0]



count = 0

for score, fname in sorted(zip(feature_scores, dv.get_feature_names()), reverse=True):

    if np.isinf(score):

        continue

    if count == 10:

        break

    print(fname, score)

    count += 1
df_lite = df[['price', 'odometer', 'long', 'year', 'condition', 'drive', 'type', 'make']]
df_cleaned = df_lite.dropna()

df_cleaned = df_cleaned[df_cleaned.price > 100]

df_cleaned.reset_index(drop=True, inplace=True)
# Preprocess the categorical columns

encoded = pd.get_dummies(df_cleaned.drop(['make'], axis=1))  # drop `make` column for now

X = encoded.drop(['price'], axis=1)

y = encoded['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=569)
parameters = {

    'max_depth':range(3,20),

    'min_samples_split':range(2,20),

    'min_samples_leaf':range(1,20),

    'max_features':[None, 'sqrt']

}

reg = GridSearchCV(DecisionTreeRegressor(), parameters, cv=10, n_jobs=4)

reg.fit(X=X_train, y=y_train)

tree_model = reg.best_estimator_

print (reg.best_score_, reg.best_params_) 
means = reg.cv_results_['mean_test_score']

stds = reg.cv_results_['std_test_score']

mean, std, params = list(zip(means, stds, reg.cv_results_['params']))[reg.best_index_]

print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
modelPred = tree_model.predict(X_test)

mse = mean_squared_error(y_test, modelPred)

print('MSE: ', mse)

rmse = np.sqrt(mse)

print('RMSE:', rmse)