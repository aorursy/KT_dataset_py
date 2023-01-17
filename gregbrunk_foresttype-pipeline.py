# imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier 

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import FunctionTransformer

from sklearn.pipeline import Pipeline



from IPython.display import SVG

from graphviz import Source

from IPython.display import display



import os

print(os.listdir("../input/forest-cover-type-kernels-only"))
# new dataframe onthe training dataset

train = pd.read_csv('../input/forest-cover-type-kernels-only/train.csv')
# wrap the feature engineering in a function for convenience and tuning

# this function results in a setting with copy warning in pandas - fix this later

pd.set_option('mode.chained_assignment', None)  # remove copy warnings for now

def add_features(df, split_aspect=True, aspect_slope=True, drop_aspects=True, 

                 elev_asp_slope=True, asp_slope_factor=10):

    if split_aspect:

        df['Aspect_N_S'] = np.cos(df.Aspect*np.pi/180)

        df['Aspect_E_W'] = np.sin(df.Aspect*np.pi/180)

        df.drop(columns = 'Aspect', inplace=True)

        

        if aspect_slope:

            df['Aspect_N_S_Slope'] = df['Aspect_N_S'] * df['Slope'] 

            df['Aspect_E_W_Slope'] = df['Aspect_E_W'] * df['Slope']

            if drop_aspects:

                df.drop(columns = ['Aspect_N_S', 'Aspect_E_W'], inplace=True)



            if elev_asp_slope:

                df['Elev_Asp_Slope'] = df['Aspect_N_S_Slope'] * asp_slope_factor +  df['Elevation']

    

    return df
pipeline = Pipeline([

        ('add_features', FunctionTransformer(add_features, validate=False, 

                                            kw_args={'split_aspect':True, 'aspect_slope':True, 

                                                     'drop_aspects':True, 'elev_asp_slope':True,

                                                     'asp_slope_factor':10}))

    ])
labels = train.columns.tolist()

labels.remove('Id')

labels.remove('Cover_Type')

labels
y = train.Cover_Type

X = train[labels]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)
estimator = RandomForestClassifier()

pipeline.fit_transform(X_train) #modifies X_train inplace

estimator.fit(X_train, y_train)
#calculate the percent correct

pipeline.transform(X_test)

sum(estimator.predict(X_test)==y_test)/len(y_test)
# Cross validation

estimator = RandomForestClassifier()

scores = cross_val_score(estimator, X_train, y_train, cv=10)



pd.Series(scores).describe()
estimator.get_params()
# Inputs for random search CV - todo: try using continuous distributions as inputs

# Number of trees in random forest

n_estimators = [3, 5, 10, 50, 100]

# Number of features to consider at every split

max_features = ['auto', None]

# Maximum number of levels in tree

max_depth = [3, 5, 10, 50, 100, None]

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4, 10]

# Method of selecting samples for training each tree

bootstrap = [True, False]# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}
# Use the random grid to search for best hyperparameters

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, # and use all available cores (not implemented)



# comment out to save compute time

'''

random_search = RandomizedSearchCV(estimator, param_distributions = random_grid, 

                               n_iter = 20, cv = 3, verbose=2, random_state=42)# Fit the random search model

random_search.fit(X_train, y_train)

'''
# print(random_search.best_params_)



# print(random_search.best_score_)



# print(random_search.best_estimator_)



# # see how it performs on the test set

# print(sum(random_search.predict(X_test)==y_test)/len(y_test))
pipeline = Pipeline([

        ('add_features', FunctionTransformer(add_features, validate=False, 

                                            kw_args={'split_aspect':True, 'aspect_slope':True, 

                                                     'drop_aspects':True, 'elev_asp_slope':True,

                                                     'asp_slope_factor':10})),

        ('random_forest', RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',

                       max_depth=100, max_features='auto', max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=2, min_samples_split=5,

                       min_weight_fraction_leaf=0.0, n_estimators=50,

                       n_jobs=None, oob_score=False, random_state=None,

                       verbose=0, warm_start=False))

    ])
y = train.Cover_Type

X = train[labels]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)



pipeline.fit(X_train, y_train)

sum(pipeline.predict(X_test)==y_test)/len(y_test)
pipeline.get_params()
#after some trial and error, this code is working, but there are a lot of dictionaries - not sure how to fix

y = train.Cover_Type

X = train[labels]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)



param_grid = [{

    'add_features__kw_args':[{'split_aspect':True},{'split_aspect':False}]

}]



grid_search_prep = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=4)
y = train.Cover_Type

X = train[labels]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)



param_grid = [{

    'add_features__kw_args':[{'split_aspect':False},

                             {'split_aspect':True, 'aspect_slope': False},

                             {'split_aspect':True, 'aspect_slope': True},

                             {'split_aspect':True, 'aspect_slope':True, 'drop_aspects':False}, 

                             {'asp_slope_factor':.5},

                              {'asp_slope_factor':1},

                              {'asp_slope_factor':10}],

    'random_forest__max_depth': [50, 100,200]

}]



grid_search_prep = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=4)
grid_search_prep.fit(X_train, y_train)
grid_search_prep.best_params_
grid_search_prep.cv_results_
for i, param in enumerate(grid_search_prep.cv_results_['params']):

    print(param,':\n     ', grid_search_prep.cv_results_['mean_test_score'][i])
sum(grid_search_prep.predict(X_test)==y_test)/len(y_test)
# a function that unonehots the data

def unonehot(df):

    df['Wilderness_Area'] = df.Wilderness_Area1 * 1 + df.Wilderness_Area2 * 2 + df.Wilderness_Area3 * 3 + df.Wilderness_Area4 *4

    df['Soil_Type'] = (df.Soil_Type1 * 1 + 

                    df.Soil_Type2 * 2 + 

                    df.Soil_Type3 * 3 + 

                    df.Soil_Type4 * 4 + 

                    df.Soil_Type5 * 5 + 

                    df.Soil_Type6 * 6 + 

                    df.Soil_Type7 * 7 + 

                    df.Soil_Type8 * 8 + 

                    df.Soil_Type9 * 9 + 

                    df.Soil_Type10 * 10 + 

                    df.Soil_Type11 * 11 + 

                    df.Soil_Type12 * 12 + 

                    df.Soil_Type13 * 13 + 

                    df.Soil_Type14 * 14 + 

                    df.Soil_Type15 * 15 + 

                    df.Soil_Type16 * 16 + 

                    df.Soil_Type17 * 17 + 

                    df.Soil_Type18 * 18 + 

                    df.Soil_Type19 * 19 + 

                    df.Soil_Type20 * 20 + 

                    df.Soil_Type21 * 21 + 

                    df.Soil_Type22 * 22 + 

                    df.Soil_Type23 * 23 + 

                    df.Soil_Type24 * 24 + 

                    df.Soil_Type25 * 25 + 

                    df.Soil_Type26 * 26 + 

                    df.Soil_Type27 * 27 + 

                    df.Soil_Type28 * 28 + 

                    df.Soil_Type29 * 29 + 

                    df.Soil_Type30 * 30 + 

                    df.Soil_Type31 * 31 + 

                    df.Soil_Type32 * 32 + 

                    df.Soil_Type33 * 33 + 

                    df.Soil_Type34 * 34 + 

                    df.Soil_Type35 * 35 + 

                    df.Soil_Type36 * 36 + 

                    df.Soil_Type37 * 37 + 

                    df.Soil_Type38 * 38 + 

                    df.Soil_Type39 * 39 + 

                    df.Soil_Type40 * 40)

    df.drop(columns = [ 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3','Wilderness_Area4', 

                     'Soil_Type1',

                     'Soil_Type2',

                     'Soil_Type3',

                     'Soil_Type4',

                     'Soil_Type5',

                     'Soil_Type6',

                     'Soil_Type7',

                     'Soil_Type8',

                     'Soil_Type9',

                     'Soil_Type10',

                     'Soil_Type11',

                     'Soil_Type12',

                     'Soil_Type13',

                     'Soil_Type14',

                     'Soil_Type15',

                     'Soil_Type16',

                     'Soil_Type17',

                     'Soil_Type18',

                     'Soil_Type19',

                     'Soil_Type20',

                     'Soil_Type21',

                     'Soil_Type22',

                     'Soil_Type23',

                     'Soil_Type24',

                     'Soil_Type25',

                     'Soil_Type26',

                     'Soil_Type27',

                     'Soil_Type28',

                     'Soil_Type29',

                     'Soil_Type30',

                     'Soil_Type31',

                     'Soil_Type32',

                     'Soil_Type33',

                     'Soil_Type34',

                     'Soil_Type35',

                     'Soil_Type36',

                     'Soil_Type37',

                     'Soil_Type38',

                     'Soil_Type39',

                     'Soil_Type40'], inplace=True)

    return df
# a function that makes columns for binary encoding to reduce dimensionality

def wilderness_bin_onehot(df):

    # make the lookup dicts

    bin0 = {}

    bin1 = {}

    bin2 = {}

    for i in range(5):

        binstr =format(i, '03b')

        bin0[i] = binstr[0]

        bin1[i] = binstr[1]

        bin2[i] = binstr[2]

    

    df['Wilderness_Area_bin0'] = df['Wilderness_Area'].map(bin0)

    df['Wilderness_Area_bin1'] = df['Wilderness_Area'].map(bin1)

    df['Wilderness_Area_bin2'] = df['Wilderness_Area'].map(bin2)

    

    df.drop(columns = 'Wilderness_Area', inplace = True)



    return df



# a function that makes columns for binary encoding to reduce dimensionality

def soil_type_bin_onehot(df):

    # make the lookup dicts

    bin0 = {}

    bin1 = {}

    bin2 = {}

    bin3 = {}

    bin4 = {}

    bin5 = {}

    for i in range(41):

        binstr =format(i, '06b')

        bin0[i] = binstr[0]

        bin1[i] = binstr[1]

        bin2[i] = binstr[2]

        bin3[i] = binstr[3]

        bin4[i] = binstr[4]

        bin5[i] = binstr[5]

    

    

    df['Soil_Type_bin0'] = df['Soil_Type'].map(bin0)

    df['Soil_Type_bin1'] = df['Soil_Type'].map(bin1)

    df['Soil_Type_bin2'] = df['Soil_Type'].map(bin2)

    df['Soil_Type_bin3'] = df['Soil_Type'].map(bin3)

    df['Soil_Type_bin4'] = df['Soil_Type'].map(bin4)

    df['Soil_Type_bin5'] = df['Soil_Type'].map(bin5)

    

    df.drop(columns = 'Soil_Type', inplace = True)



    return df



# wrap the previous functions up in a single function so that they can be added to the pipelinea function that makes columns for binary encoding to reduce dimensionality

def bin_onehot(df, perf_bin_onehot = True):

    if perf_bin_onehot: 

        unonehot(df)

        wilderness_bin_onehot(df)

        soil_type_bin_onehot(df)

    return df

pipeline = Pipeline([

        ('add_features', FunctionTransformer(add_features, validate=False, 

                                            kw_args={'split_aspect':True, 'aspect_slope':True, 

                                                     'drop_aspects':True, 'elev_asp_slope':True,

                                                     'asp_slope_factor':10})),

        ('bin_onehot', FunctionTransformer(bin_onehot, validate=False, 

                                            kw_args={'perf_bin_onehot':True})),

        ('random_forest', RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',

                       max_depth=100, max_features='auto', max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=2, min_samples_split=5,

                       min_weight_fraction_leaf=0.0, n_estimators=50,

                       n_jobs=None, oob_score=False, random_state=None,

                       verbose=0, warm_start=False))

    ])
y = train.Cover_Type

X = train[labels]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)



pipeline.fit(X_train, y_train)

sum(pipeline.predict(X_test)==y_test)/len(y_test)
y = train.Cover_Type

X = train[labels]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)



param_grid = [{

    'add_features__kw_args':[{'split_aspect':False}, 

                             {'asp_slope_factor':.5}],

    'bin_onehot__kw_args': [{'perf_bin_onehot':True},

                           {'perf_bin_onehot':False}],

    'random_forest__max_depth': [100,200]

}]



grid_search_prep = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=4)
grid_search_prep.fit(X_train, y_train)

print(grid_search_prep.best_params_)

for i, param in enumerate(grid_search_prep.cv_results_['params']):

    print(param,':\n     ', grid_search_prep.cv_results_['mean_test_score'][i])
sum(grid_search_prep.predict(X_test)==y_test)/len(y_test)
# new dataframe onthe training dataset

test = pd.read_csv('../input/forest-cover-type-kernels-only/test.csv')
labels = test.columns[1:]

labels
pred = grid_search_prep.predict(test[labels])
submit = pd.DataFrame(test.Id)

submit['Cover_Type'] = pred

submit.to_csv('submission.csv', index=False)

submit