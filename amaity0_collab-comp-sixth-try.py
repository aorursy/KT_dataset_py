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
#reading the files

train = pd.read_csv("../input/learn-together/train.csv")

test = pd.read_csv("../input/learn-together/test.csv")



y = train.Cover_Type

test_id = test['Id']
##PREPROCESSING

import pickle



if os.path.isfile("X.pickle"):

    with open( "X.pickle", "rb" ) as fh1:

        X = pickle.load(fh1)

    with open('test.pickle', 'rb') as fh2:

        test = pickle.load(fh2)

else:

    #dropping Soil_Type7 and Soil_Type15

    train = train.drop(['Id','Soil_Type7', 'Soil_Type15'], axis = 1)

    test_id = test['Id']

    test = test.drop(['Id','Soil_Type7', 'Soil_Type15'], axis = 1)



    #prepare data for training the model

    X = train.drop(['Cover_Type'], axis = 1)



    #reducing Soil_Type cols to single col 

    X = X.iloc[:, :14].join(X.iloc[:, 14:].dot(range(1,39)).to_frame('Soil_Type1'))

    test = test.iloc[:, :14].join(test.iloc[:, 14:].dot(range(1,39)).to_frame('Soil_Type1'))

    #print(X.columns)

    #reducing Wilderness_Area to single col 

    X = X.iloc[:,:10].join(X.iloc[:,10:-1].dot(range(1,5)).to_frame('Wilderness_Area1')).join(X.iloc[:,-1])

    test = test.iloc[:,:10].join(test.iloc[:,10:-1].dot(range(1,5)).to_frame('Wilderness_Area1')).join(test.iloc[:,-1])



    #horizontal and vertical distance to hydrology can be easily combined

    cols = ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology']

    X['Distance_to_hydrology'] = X[cols].apply(np.linalg.norm, axis=1)

    X = X.drop(cols, axis = 1)

    test['Distance_to_hydrology'] = test[cols].apply(np.linalg.norm, axis=1)

    test = test.drop(cols, axis = 1)



    #shot in the dark - convert like colour tuples to grayscale

    cols = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']

    weights = pd.Series([0.299, 0.587, 0.114], index=cols)

    X['Hillshade'] = (X[cols]*weights).sum(1)

    X = X.drop(cols, axis = 1)

    test['Hillshade'] = (test[cols]*weights).sum(1)

    test = test.drop(cols, axis=1)



    #pickling data for quick access

    with open('X.pickle', 'wb') as fh1:

        pickle.dump(X, fh1)

    with open('test.pickle', 'wb') as fh2:

        pickle.dump(test, fh2)



print(X.columns)
#split data

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

model = RandomForestClassifier(n_estimators=100)

model.fit(X_train,y_train)

y_pred = model.predict(X_val)

val_mae = mean_absolute_error(y_val,y_pred)

print('Sixth try mae of RFClassifier with base parameters: ', val_mae)
print('Random search','-'*20)

#preparing model

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 11)]

#n_estimators = [100, 125, 150, 180, 200, 250]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

#max_depth =  [50, 60, 70, 80, 90, 100]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)
rf = RandomForestClassifier()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 

                    n_iter = 100, cv = 3, verbose=0, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(X_train, y_train)

print('rf_random.best_params_:',rf_random.best_params_)
def evaluate(model, test_features, test_labels):

    predictions = model.predict(test_features)

    errors = abs(predictions - test_labels)

    mape = 100 * np.mean(errors / test_labels)

    accuracy = 100 - mape

    print('Model Performance')

    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))

    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy



print('Base model','-'*20)

base_model = model

base_model.fit(X_train, y_train)

base_accuracy = evaluate(base_model, X_val, y_val)

print('Random search estimator','-'*20)

best_random = rf_random.best_estimator_

random_accuracy = evaluate(best_random,  X_val, y_val)
y_pred = best_random.predict(X_val)

val_mae = mean_absolute_error(y_val,y_pred)

print('Sixth try mae with RS RFClassifier: ', val_mae)
test_pred = best_random.predict(test)

output = pd.DataFrame({'Id': test_id, 'Cover_Type': test_pred.astype(int)})

output.to_csv('submission.csv', index=False)