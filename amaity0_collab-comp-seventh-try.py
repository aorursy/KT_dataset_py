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
#setting aside target values and index

y = train.Cover_Type

test_id = test['Id']
#do this preprocessing job only once

import pickle



if os.path.isfile("X.pickle"):

    with open( "X.pickle", "rb" ) as fh1:

        X = pickle.load(fh1)

    with open('test.pickle', 'rb') as fh2:

        test = pickle.load(fh2)

else:

    #dropping Soil_Type7 and Soil_Type15

    train = train.drop(['Id','Soil_Type7', 'Soil_Type15'], axis = 1)

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

    print(X.columns)



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
#split data

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
#approx measure of roc_auc for multiclass target

from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import roc_auc_score

def multiclass_roc_auc_score(test, pred, average='micro'):

    lb = LabelBinarizer()

    lb.fit(test)

    test = lb.transform(test)

    pred = lb.transform(pred)

    return roc_auc_score(test, pred, average=average)
#set parameter values

param_grid = {"n_estimators":  np.arange(2,50,2),

              "max_depth":  [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],

              "min_samples_split": np.linspace(0.1, 1.0, 10, endpoint=True), #np.arange(1,150,1),

              "min_samples_leaf": np.linspace(0.1, 0.5, 5, endpoint=True),  #np.arange(1,60,1),

              "max_leaf_nodes": np.arange(5,150,5),

              "min_weight_fraction_leaf": np.arange(0.1,0.4, 0.1)}
#set the classifier

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=1)
#a function for plotting score against each parameter

import matplotlib.pyplot as plt

%matplotlib inline

import warnings



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

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10,5))

    for (parameter, group), ax in zip(df.groupby(df.Parameter), axes.flatten()):

        group.plot(x='Param_value', y=(['Train_'+metric_abv,'Test_'+metric_abv]),

        kind='line', ax=ax, title=parameter)

        ax.set_xlabel('')

    plt.tight_layout()

    plt.show()
from sklearn.metrics import mean_absolute_error

#evaluate_param(clf, param_grid, mean_absolute_error, 'MAE')
#refining the number of search points

param_grid2 = {"n_estimators": [18,21],

               'max_leaf_nodes': [150,None],

               #'max_depth': [None],

                #'min_samples_split': [2, 5], 

                #'min_samples_leaf': [1, 2],

              "max_features": ['auto','sqrt'],

              "bootstrap": [True, False]}
from sklearn.model_selection import GridSearchCV



grid_search = GridSearchCV(clf, param_grid=param_grid2, cv=8, 

                            scoring='accuracy')

gs_result = grid_search.fit(X_train, y_train)

print(gs_result.best_params_)

best_clf = RandomForestClassifier(gs_result.best_params_)

y_pred = gs_result.predict(X_val)

val_mae = mean_absolute_error(y_val,y_pred)

print('Seventh try mae: ', val_mae)
test_pred = gs_result.predict(test)

output = pd.DataFrame({'Id': test_id, 'Cover_Type': test_pred.astype(int)})

output.to_csv('submission.csv', index=False)