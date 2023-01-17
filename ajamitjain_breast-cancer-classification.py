# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## Importing libraries for visualization 

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler



## Importing libraries for modeling and evaluation

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import RFE



from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import f1_score,confusion_matrix

from sklearn.metrics import accuracy_score
## Function to plot confusion matrix

def plot_confusion_matrix(prediction, actual):

    cm = confusion_matrix(prediction, actual)

    sns.heatmap(cm, annot=True, fmt="d")
## Function to printing the accuracy 

def print_accuracy(model_name, prediction, actual):

    print('Accuracy for {} classifier: {}% '.format(model_name, round(accuracy_score(prediction, actual), 2)*100))
## Importing the dataset

df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

df.head()
## Shape of the dataset

df.shape
df.info()
## Getting features from the dataset

not_predictor_list = ['Unnamed: 32','id','diagnosis']

features = df.drop(not_predictor_list, axis=1)

features.head()
## Details ( like mean, ranges, quantiles) of different features from above table.

features.describe()
## Creating a copy of the features dataframe

df_combined = features.copy()



## Standarding the dataset 

sc = StandardScaler()

df_combined = sc.fit_transform(df_combined)



## Converting dataset back to dataframe from numpy array (standard scaler returns array)

df_combined = pd.DataFrame(df_combined, columns=features.columns)



## Creating separate dataframes for mean, se and worst features

df_mean_combined = df_combined.loc[:, ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean']]

df_se_combined = df_combined.loc[:, ['radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se','compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se','fractal_dimension_se']]

df_worst_combined = df_combined.loc[:,['radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']]



## Adding diagnosis column to above dataframes

df_mean_combined['diagnosis'] = df.loc[:, 'diagnosis']

df_se_combined['diagnosis'] = df.loc[:, 'diagnosis']

df_worst_combined['diagnosis'] = df.loc[:, 'diagnosis']



## Melting the dataframe

df_mean_melted = pd.melt(df_mean_combined, id_vars = 'diagnosis',

                                 var_name = 'features',

                                 value_name = 'value')



df_se_melted = pd.melt(df_se_combined, id_vars = 'diagnosis',

                                 var_name = 'features',

                                 value_name = 'value')



df_worst_melted = pd.melt(df_worst_combined, id_vars = 'diagnosis',

                                 var_name = 'features',

                                 value_name = 'value')



## Ploting the box plot for different features to understand it's relation with diagnosis parameters (type of cell)

fig, ax = plt.subplots(figsize=(12,12))

sns.boxplot(x='features', y='value', hue='diagnosis', data=df_mean_melted, showfliers=False)

plt.xticks(rotation=90)
## Ploting the box plot for different features to understand it's relation with diagnosis parameters (type of cell)

fig, ax = plt.subplots(figsize=(12,12))

sns.boxplot(x='features', y='value', hue='diagnosis', data=df_se_melted, showfliers=False)

plt.xticks(rotation=90)
## Ploting the box plot for different features to understand it's relation with diagnosis parameters (type of cell)

fig, ax = plt.subplots(figsize=(12,12))

sns.boxplot(x='features', y='value', hue='diagnosis', data=df_worst_melted, showfliers=False)

plt.xticks(rotation=90)
corr_mat = features.corr()
fig, ax = plt.subplots(figsize = (18,18))

sns.heatmap(corr_mat, annot=True)
## Removing the correlated feature 

correlated_feature_list =  ['radius_mean','perimeter_mean','compactness_mean','concave points_mean',

                            'radius_se','perimeter_se', 'compactness_se','concave points_se',

                            'radius_worst','texture_worst', 'perimeter_worst','area_worst', 

                            'smoothness_worst', 'compactness_worst', 'concavity_worst','concave points_worst']



X = features.drop(correlated_feature_list, axis=1)

y = df[['diagnosis']]

## Encoding the categorical response variable

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

y = encoder.fit_transform(y.values.ravel())
## Displayng the final features dataset

X.head()
## Displaying the info about the features

X.info()
## Spiliting the dataset into train and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
## Using RFE & random forest classifier to select top 5 features 

random_forest_classifier = RandomForestClassifier()

rfe = RFE(estimator=random_forest_classifier, n_features_to_select=5, step=1)

rfe = rfe.fit(X_train, y_train)



## Printing list of top features

top_feature = X_train.columns[rfe.support_].tolist()

print('Top 5 feature by rfe and random forest:',top_feature)

## Limiting the features with top features

X_train = X_train.loc[:, top_feature]

X_test = X_test.loc[:, top_feature]
## Scaling before fitting it to model

sc_ = StandardScaler()

X_train = sc_.fit_transform(X_train)

X_test = sc_.transform(X_test)
## Applying the logistic regression classifier

from sklearn.linear_model import LogisticRegression



logistic_reg_classifier = LogisticRegression(random_state=10, max_iter=150)

logistic_reg_classifier.fit(X_train, y_train)
## Predicting training value

logistic_y_train_pred = logistic_reg_classifier.predict(X_train)



## Plotting a confusion matrix for logistic regression classifier

plot_confusion_matrix(logistic_y_train_pred, y_train)



## Printing the accuracy 

model_name = 'logistic regression'

print_accuracy(model_name, logistic_y_train_pred, y_train)
## Predicting the classifier output

logistic_y_pred = logistic_reg_classifier.predict(X_test)





## Plotting a confusion matrix for logistic regression classifier

plot_confusion_matrix(logistic_y_pred, y_test)



## Printing the accuracy 

model_name = 'logistic regression'

print_accuracy(model_name, logistic_y_pred, y_test)
## Applying the K nearest neighbors classifier

from sklearn.neighbors import KNeighborsClassifier



knn_classifier = KNeighborsClassifier(n_neighbors=5)

knn_classifier.fit(X_train, y_train)

## Predicting training value

knn_y_train_pred = knn_classifier.predict(X_train)



## Plotting a confusion matrix for logistic regression classifier

plot_confusion_matrix(knn_y_train_pred, y_train)



## Printing the accuracy 

model_name = 'knn'

print_accuracy(model_name, knn_y_train_pred, y_train)
## Predicting the classifier output

knn_y_pred = knn_classifier.predict(X_test)



## Plotting a confusion matrix for KNN 

plot_confusion_matrix(knn_y_pred, y_test)



## Printing the accuracy 

model_name = 'KNN'

print_accuracy(model_name, knn_y_pred, y_test)
## Applying the random forest classifier

from sklearn.ensemble import RandomForestClassifier



random_forest_classifier = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=0)

random_forest_classifier.fit(X_train, y_train)
## Predicting the classifier output

random_forest_y_pred = random_forest_classifier.predict(X_test)



## Plotting a confusion matrix for KNN 

plot_confusion_matrix(random_forest_y_pred, y_test)



## Printing the accuracy 

model_name = 'random forest'

print_accuracy(model_name, random_forest_y_pred, y_test)
## Predicting training value

random_forest_y_train_pred = random_forest_classifier.predict(X_train)



## Plotting a confusion matrix for logistic regression classifier

plot_confusion_matrix(random_forest_y_train_pred, y_train)



## Printing the accuracy 

model_name = 'random forest'

print_accuracy(model_name, random_forest_y_train_pred, y_train)
## Applying the logistic regression classifier

from xgboost import XGBClassifier



xgb_classifier = XGBClassifier()

xgb_classifier.fit(X_train, y_train)
## Predicting training value

xgb_y_train_pred = xgb_classifier.predict(X_train)



## Plotting a confusion matrix for logistic regression classifier

plot_confusion_matrix(xgb_y_train_pred, y_train)



## Printing the accuracy 

model_name = 'xgb'

print_accuracy(model_name, xgb_y_train_pred, y_train)
## Predicting the classifier output

xgb_y_pred = xgb_classifier.predict(X_test)



## Plotting a confusion matrix for XGB 

plot_confusion_matrix(xgb_y_pred, y_test)



## Printing the accuracy 

model_name = 'XGB'

print_accuracy(model_name, xgb_y_pred, y_test)
## Preparing the train and test set for Neural Network



## Spiliting the dataset into train and test set

X_ann_train, X_ann_test, y_ann_train, y_ann_test = train_test_split(X, y, test_size=0.3, random_state=0)



## Standradizing the train and test data

sc = StandardScaler()

X_ann_train = sc.fit_transform(X_ann_train)

X_ann_test = sc.transform(X_ann_test)
epochs = 100

batch_size = 1
## Applying the artificial neural network classifier

import keras

from keras.layers import Dense

from keras.layers import Dropout

from keras.models import Sequential



ann_classifier = Sequential()



## Adding hidden layers

ann_classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=14))

ann_classifier.add(Dropout(0.2))

ann_classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu')) 

ann_classifier.add(Dropout(0.2))



## Adding output layer                   

ann_classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

                   

## Compiling the classifier                

ann_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

           

## Fitting the classifier

ann_classifier.fit(X_ann_train, y_ann_train, batch_size = batch_size, epochs = epochs)                   
## Predicting training value

ann_y_train_pred = ann_classifier.predict(X_ann_train)

ann_y_train_pred = (ann_y_train_pred > 0.5)



## Plotting a confusion matrix for logistic regression classifier

plot_confusion_matrix(ann_y_train_pred, y_ann_train)



## Printing the accuracy 

model_name = 'artificial neural network'

print_accuracy(model_name, ann_y_train_pred, y_ann_train)
## Predicting the classifier output

ann_y_pred = ann_classifier.predict(X_ann_test)

ann_y_pred = (ann_y_pred > 0.5)



## Plotting a confusion matrix for KNN 

plot_confusion_matrix(ann_y_pred, y_ann_test)



## Printing the accuracy 

model_name = 'artificial neural network'

print_accuracy(model_name, ann_y_pred, y_ann_test)
### Accuracy comparision for all the above model

print_accuracy('logistic regression', logistic_y_pred, y_test)

print_accuracy('KNN', knn_y_pred, y_test)

print_accuracy('random forest', random_forest_y_pred, y_test)

print_accuracy('XGB', xgb_y_pred, y_test)

print_accuracy('artificial neural network', ann_y_pred, y_ann_test)
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.model_selection import GridSearchCV





## Creating the params for logistic regression

classifier_lr = LogisticRegression()

solver_list = ['newton-cg', 'lbfgs', 'liblinear']

penalty_list = ['l2']

c_list = [100, 10, 1.0, 0.1, 0.01]



# Defining the grid and folds

grid = dict(solver=solver_list,penalty=penalty_list,C=c_list)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)



## Creating the grid search 

grid_search = GridSearchCV(estimator=classifier_lr, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)

grid_result = grid_search.fit(X_train, y_train)



# Grid search results 

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("Test score : %f (%f) with parameters: %r" % (mean, stdev, param))
print("Best parameters: ", grid_result.best_params_)
## Applying the hyperparameter logistic regression classifier

logistic_reg_hp_classifier = LogisticRegression(C= 1, penalty='l2', solver= 'newton-cg', random_state=10, max_iter=150)

logistic_reg_hp_classifier.fit(X_train, y_train)
## Predicting training value

logistic_y_train_pred = logistic_reg_hp_classifier.predict(X_train)



## Plotting a confusion matrix for logistic regression classifier

plot_confusion_matrix(logistic_y_train_pred, y_train)



## Printing the accuracy 

model_name = 'logistic regression'

print_accuracy(model_name, logistic_y_train_pred, y_train)

## Predicting the classifier output

logistic_y_pred = logistic_reg_hp_classifier.predict(X_test)





## Plotting a confusion matrix for logistic regression classifier

plot_confusion_matrix(logistic_y_pred, y_test)



## Printing the accuracy 

model_name = 'logistic regression'

print_accuracy(model_name, logistic_y_pred, y_test)
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
## Creating the hyperparams list 

params ={'max_depth': hp.quniform("max_depth", 3, 18, 1),

        'gamma': hp.uniform ('gamma', 0,9),

        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),

        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),

        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),

        'n_estimators': 1000,

        'learning_rate':hp.quniform('learning_rate', 0.01, 0.1, 0.01)}
def objective(params):

    classifier = XGBClassifier(

                    n_estimators =params['n_estimators'], 

                    max_depth = int(params['max_depth']),

                    gamma = params['gamma'],

                    reg_alpha = params['reg_alpha'],

                    min_child_weight=params['min_child_weight'],

                    colsample_bytree=params['colsample_bytree'],

                    learning_rate=params['learning_rate'])

    

    evaluation = [( X_train, y_train), ( X_test, y_test)]

    

    classifier.fit(X_train, y_train,

            eval_set=evaluation, eval_metric="auc",

            early_stopping_rounds=10,verbose=False)

    

    pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    

    print('Socre:', accuracy)

    

    return {'loss': -accuracy, 'status': STATUS_OK }
trials = Trials()



best_hyperparams = fmin(fn = objective,

                        space = params,

                        algo = tpe.suggest,

                        max_evals = 500,

                        trials = trials)



print("Best parameters :", best_hyperparams)
xgb_hp_classifier = XGBClassifier( 

                    n_estimators = 1000, 

                    max_depth = int(round(best_hyperparams['max_depth'], 3)),

                    gamma = round(best_hyperparams['gamma'], 3),

                    reg_alpha = int(round(best_hyperparams['reg_alpha'], 3)),

                    min_child_weight=round(best_hyperparams['min_child_weight'], 3),

                    colsample_bytree=round(best_hyperparams['colsample_bytree'], 3),

                    learning_rate=best_hyperparams['learning_rate'])

    

evaluation = [( X_train, y_train), ( X_test, y_test)]

    

xgb_hp_classifier.fit(X_train, y_train,

            eval_set=evaluation, eval_metric="auc",

            early_stopping_rounds=10,verbose=False)

    
## Predicting the train data

y_pred = xgb_hp_classifier.predict(X_train)



## Plotting a confusion matrix for KNN 

plot_confusion_matrix(y_pred, y_train)



## Printing the accuracy 

model_name = 'XGB'

print_accuracy(model_name, y_pred, y_train)
## Predicting the test data

y_pred = xgb_hp_classifier.predict(X_test)



## Plotting a confusion matrix for KNN 

plot_confusion_matrix(y_pred, y_test)



## Printing the accuracy 

model_name = 'XGB'

print_accuracy(model_name, y_pred, y_test)
## Printing the accuracy 

model_name = 'XGB'

print_accuracy(model_name, y_pred, y_test)