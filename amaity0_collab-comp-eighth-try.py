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

print(train.columns)
#set aside the targets and id

y = train.Cover_Type

test_id = test['Id']
#dropping Ids

train = train.drop(['Id'], axis = 1)

test = test.drop(['Id'], axis = 1)



#prepare data for training the model

X = train.drop(['Cover_Type'], axis = 1)



#horizontal and vertical distance to hydrology can be easily combined

cols = ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology']

X['Distance_to_hydrology'] = X[cols].apply(np.linalg.norm, axis=1)

test['Distance_to_hydrology'] = test[cols].apply(np.linalg.norm, axis=1)



#adding a few combinations of distance features to help enhance the classification

cols = ['Horizontal_Distance_To_Roadways',

        'Horizontal_Distance_To_Fire_Points',

        'Horizontal_Distance_To_Hydrology']



def addDistFeatures(df):

    df['distance_mean'] = df[cols].mean(axis=1)

    df['distance_sum'] = df[cols].sum(axis=1)

    df['distance_dif_road_fire'] = df[cols[0]] - df[cols[1]]

    df['distance_dif_hydro_road'] = df[cols[2]] - df[cols[0]]

    df['distance_dif_hydro_fire'] = df[cols[2]] - df[cols[1]]

    return df



X = addDistFeatures(X)

test = addDistFeatures(test)



#persisting with the idea of adding simple combination of Hillshades

cols = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']

weights = pd.Series([0.299, 0.587, 0.114], index=cols)

X['Hillshade'] = (X[cols]*weights).sum(1)

test['Hillshade'] = (test[cols]*weights).sum(1)



print(X.columns)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
param_grid = {"n_estimators":  [int(x) for x in np.linspace(start = 10, stop = 200, num = 11)],

              "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],

              "min_samples_split": np.linspace(0.1, 1.0, 10, endpoint=True), #np.arange(1,150,1),

              "min_samples_leaf": np.linspace(0.1, 0.5, 5, endpoint=True),  #np.arange(1,60,1),

              "max_leaf_nodes": np.arange(2,60,1),

              "min_weight_fraction_leaf": np.arange(0.1,0.4, 0.1)}



#parameter tuning

import matplotlib.pyplot as plt

from collections import defaultdict

import warnings

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



clf = RandomForestClassifier(random_state=1)



def evaluate_param(clf, param_grid, metric, metric_abv):

    data = []

    for parameter, values in dict.items(param_grid):

        for value in values:

            d = {parameter:value}

            warnings.filterwarnings('ignore') 

            clf = RandomForestClassifier(**d)

            clf.fit(X_train, y_train)

            x_pred = clf.predict(X_train)

            train_score = metric(y_train, x_pred)

            y_pred = clf.predict(X_val)

            test_score = metric(y_val, y_pred)

            data.append({'Parameter':parameter, 'Param_value':value, 

            'Train_'+metric_abv:train_score, 'Test_'+metric_abv:test_score})

    df = pd.DataFrame(data)

    _, axes = plt.subplots(nrows=2, ncols=3, figsize=(10,5))

    for (parameter, group), ax in zip(df.groupby(df.Parameter), axes.flatten()):

        group.plot(x='Param_value', y=(['Train_'+metric_abv,'Test_'+metric_abv]),

        kind='line', ax=ax, title=parameter)

        ax.set_xlabel('')

    plt.tight_layout()

    plt.show()



evaluate_param(clf, param_grid, accuracy_score, 'ACC')
param_grid2 = {"n_estimators": [29,47,113,181],

                #'max_leaf_nodes': [150,None],

                #'max_depth': [20,None],

                #'min_samples_split': [2, 5], 

                #'min_samples_leaf': [1, 2],

              "max_features": ['auto','sqrt'],

              "bootstrap": [True, False]}



from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(clf, param_grid2, refit=True, cv=5, verbose=0)

grid.fit(X_train, y_train)

print('Best parameters: ',grid.best_params_)

print('Best estimator: ',grid.best_estimator_)

grid_predictions = grid.predict(X_val) 



from sklearn.metrics import classification_report

print(classification_report(y_val, grid_predictions))
test_pred = grid.predict(test)

output = pd.DataFrame({'Id': test_id, 'Cover_Type': test_pred.astype(int)})

output.to_csv('submission.csv', index=False)