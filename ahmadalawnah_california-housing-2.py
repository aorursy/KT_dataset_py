# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.impute import SimpleImputer

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import FeatureUnion

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPRegressor, MLPClassifier

from sklearn.model_selection import cross_val_score, train_test_split 

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout



from numpy.random import seed

import tensorflow

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
housing = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')

housing.dropna(axis=0)

housing['rooms_per_household'] = housing['total_rooms'] / housing['households']

housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]

housing["population_per_household"]=housing["population"]/housing["households"]

train_set, test_set = train_test_split(housing, test_size=0.2,random_state=70)

train_set_labels = train_set["median_house_value"].copy()

train_set = train_set.drop("median_house_value",axis=1)



test_set_labels = test_set["median_house_value"].copy()

test_set = test_set.drop("median_house_value",axis=1)
train_numeric = train_set.drop('ocean_proximity',axis=1)

test_numeric = test_set.drop('ocean_proximity',axis=1)
num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy="median")),

    ('std_scaler', StandardScaler()),])

num_attribs = list(train_numeric)

full_pipeline = ColumnTransformer([

    ("num", num_pipeline, num_attribs),

    ("cat", OneHotEncoder(), ['ocean_proximity']),])



train_prepared = full_pipeline.fit_transform(train_set)

print(train_prepared)
def create_model(lyrs=[10,10,10,10,10], act='relu', opt='Adam', dr=0.0):

    

    # set random seed for reproducibility

    seed(42)

    tensorflow.random.set_seed(42)

    

    model = Sequential()

    

    # create first hidden layer

    model.add(Dense(lyrs[0], input_dim=train_prepared.shape[1], activation=act))

    

    # create additional hidden layers

    for i in range(1,len(lyrs)):

        model.add(Dense(lyrs[i], activation=act))

    

    # add dropout, default is none

    model.add(Dropout(dr))

    

    # create output layer

    model.add(Dense(1, activation='sigmoid'))  # output layer

    

    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

    

    return model
num_attribs = list(test_numeric)

test_prepared = full_pipeline.fit_transform(test_set)

#model = KerasRegressor(build_fn=create_model, verbose=0, epochs = 1000)



#res = model.fit(train_prepared,train_set_labels)



#print(np.sqrt(np.mean(res.history['mean_squared_error'])))
model = KerasRegressor(build_fn=create_model, verbose=0)



activations = ['relu', 'tanh', 'sigmoid']



param_grid = dict(act=activations)



grid1 = GridSearchCV(estimator=model, 

                    param_grid=param_grid,

                    cv=4,

                    verbose=2,

                    scoring='neg_mean_squared_error')  



grid1_result = grid1.fit(train_prepared, train_set_labels)
print("Best: %f using %s" % (grid1_result.best_score_, grid1_result.best_params_))
model2 = KerasRegressor(build_fn=create_model, verbose=0)



epochs = [200, 500, 1000]

activation = ['relu']

param_grid = dict(epochs=epochs, act=activation)



grid2 = GridSearchCV(estimator=model2, 

                    param_grid=param_grid,

                    cv=4,

                    verbose=2)  



grid2_result = grid2.fit(train_prepared, train_set_labels)
print("Best: %f using %s" % (grid2_result.best_score_, grid2_result.best_params_))