# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from numpy.random import seed

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler, Normalizer, OrdinalEncoder, OneHotEncoder

from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

import keras.backend as K

from sklearn.metrics import explained_variance_score, max_error, r2_score, mean_poisson_deviance, mean_gamma_deviance

from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
housing = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')

housing.info()
housing=housing.dropna(axis=0)



housing['rooms_per_household'] = housing['total_rooms'] / housing['households']

housing["bedrooms_per_household"] = housing["total_bedrooms"]/housing["total_rooms"]

housing["population_per_household"]=housing["population"]/housing["households"]

housing.info()

housing = housing.drop('total_rooms', axis=1)

housing = housing.drop('total_bedrooms', axis=1)

housing = housing.drop('population', axis=1)
train = housing.drop('median_house_value',axis=1)

train_labels = housing['median_house_value'].copy()

train_numeric = train.drop('ocean_proximity',axis=1)
pipeline = Pipeline([('scaler', Normalizer())])



numeric_attributes= list(train_numeric)

full_pipeline = ColumnTransformer([

    ("num", pipeline, numeric_attributes),

    ("cat", OneHotEncoder(), ['ocean_proximity']),])



train_prepared = full_pipeline.fit_transform(train)

print(train_prepared)
def create_model(lyrs=[256,256,256,256,256,128,64,32,16], act='relu', opt='Adam', dr=0.0):

    

    # set random seed for reproducibility

    seed(42)

    tf.random.set_seed(42)

    

    model = Sequential()

    

    model.add(Dense(lyrs[0], input_dim=train_prepared.shape[1], activation=act))

    

    # create additional hidden layers

    for i in range(1,len(lyrs)):

        model.add(Dense(lyrs[i], activation=act))

    

    # add dropout, default is none

    model.add(Dropout(dr))

    

    # create output layer

    model.add(Dense(1, activation='sigmoid'))

    

    model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mean_absolute_percentage_error'])

    

    return model
model = create_model()



res = model.fit(train_prepared,train_labels, epochs=200)



print(np.mean(res.history['mean_absolute_percentage_error']))
#model = KerasRegressor(build_fn=create_model, verbose=0)



#activations = ['relu', 'tanh', 'sigmoid']



#param_grid = dict(act=activations)



#grid1 = GridSearchCV(estimator=model, 

                    #param_grid=param_grid,

                   # cv=4,

                   # verbose=2)  



#grid1_result = grid1.fit(train_prepared, train_labels)
#print("Best: %f using %s" % (grid1_result.best_score_, grid1_result.best_params_))
#model2 = KerasRegressor(build_fn=create_model, verbose=0)



#epochs = [200, 500, 1000]

#activation = ['relu']

#param_grid = dict(epochs=epochs, act=activation)



#grid2 = GridSearchCV(estimator=model2, 

#                    param_grid=param_grid,

#                    cv=4,

#                    verbose=2)  



#grid2_result = grid2.fit(train_prepared, train_labels)
#print("Best: %f using %s" % (grid2_result.best_score_, grid2_result.best_params_))