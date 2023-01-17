import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, preprocessing, svm
from sklearn.preprocessing import StandardScaler, Normalizer
import math
import matplotlib
import seaborn as sns

import os
# print(os.listdir("../input/usedcarvaluation"))
print(os.listdir("../input/dataset"))

# pd.read_csv('../input/dataset/X_train.csv').head(5)


#Data preparation
df_x_train = pd.read_csv('../input/dataset/X_train.csv')
df_y_train = pd.read_csv('../input/dataset/y_train.csv')

df_X_test = pd.read_csv('../input/dataset/X_test.csv')

df_x_train.head(5)
# df_x_train.shape
# print("================")
# df_y_train.head(5)
df_y_train.head(5)
print("X shape: {}".format(df_x_train.shape))
print("Y shape: {}".format(df_y_train.shape))

print("X shape: {}".format(df_X_test.shape))
print(df_x_train.brand.unique())
print(df_x_train.engine_size.unique())
print(df_x_train.year.unique())
print(df_x_train.prev_owners.unique())
df_x_train.isnull().sum()
df_X_test.isnull().sum()
#VISUALIZATIONS
categories = ['prev_owners', 'brand']

for i, c in enumerate(categories):
    v = df_x_train[c].unique()
    
    g = df_x_train.groupby(by=c)[c].count().sort_values(ascending=False)
    r = range(min(len(v), 5))

    print( g.head())
    plt.figure(figsize=(5,3))
    plt.bar(r, g.head()) 
    #plt.xticks(r, v)
    plt.xticks(r, g.index)
    plt.show()
df_x_train['namelen'] = [min(70, len(n)) for n in df_x_train['id']]

df_X_test['namelen'] = [min(70, len(n)) for n in df_X_test['id']]
labels = ['id', 'brand']
les = {}

for l in labels:
    les[l] = preprocessing.LabelEncoder()
    les[l].fit(df_x_train[l])
    tr = les[l].transform(df_x_train[l]) 
    df_x_train.loc[:, l + '_feat'] = pd.Series(tr, index=df_x_train.index)

labeled = df_x_train[ ['year'
                        ,'age'
                        ,'engine_size'
                        ,'power'
                        ,'prev_owners'
                        , 'mileage'] 
                    + [x+"_feat" for x in labels]]


labels_test = ['id', 'brand']
les_test = {}

for l in labels_test:
    les_test[l] = preprocessing.LabelEncoder()
    les_test[l].fit(df_X_test[l])
    tr = les_test[l].transform(df_X_test[l]) 
    df_X_test.loc[:, l + '_feat'] = pd.Series(tr, index=df_X_test.index)

labeled_test = df_X_test[ ['year'
                        ,'age'
                        ,'engine_size'
                        ,'power'
                        ,'prev_owners'
                        , 'mileage'] 
                    + [x+"_feat" for x in labels_test]]


labeled.head(5)
labeled_test.head(5)
len(labeled['id_feat'].unique()) / len(labeled['id_feat'])
labeled.drop(['id_feat'], axis='columns', inplace=True)
labeled_test.drop(['id_feat'], axis='columns', inplace=True)
def category_values(dataframe, categories):
    for c in categories:
        print('\n', dataframe.groupby(by=c)[c].count().sort_values(ascending=False))
        print('Nulls: ', dataframe[c].isnull().sum())

def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )
plot_correlation_map(labeled)
labeled.corr()
#Prepare data for training
Y = df_y_train['value']
X = labeled
# X = labeled.drop(['price'], axis='columns', inplace=False)

X_TEST_DATA = labeled_test

matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"1. Before":Y, "2. After":np.log1p(Y)})
prices.hist()

Y = np.log1p(Y)
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score, train_test_split


def cv_rmse(model, x, y):
    r = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv = 5))
    return r

# Percent of the X array to use as training set. This implies that the rest will be test set
test_size = .20

#Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=test_size, random_state = 3)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

r = range(2003, 2017)
km_year = 10000

#Predicting using various models:
#LINEAR REGRESSION
linear = linear_model.LinearRegression()
# Percent of the X array to use as training set. This implies that the rest will be test set
test_size = .2
#Split into train and validation
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state = 3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
linear.fit(X_train, y_train)
# Explained variance score: 1 is perfect prediction
score = linear.score(X_test, y_test)
print('Variance score: %.2f' % score)
print("rmse on validation set", cv_rmse(linear, X_test, y_test).mean())
#Used grid search to set the optimal parameters for the regressors then train final modelRandom forests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

rf = RandomForestRegressor()
param_grid = { "criterion" : ["mse"]
              , "min_samples_leaf" : [3]
              , "min_samples_split" : [3]
              , "max_depth": [10]
              , "n_estimators": [500]}

gs = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, n_jobs=-1, verbose=1)
gs = gs.fit(X_train, y_train)

print(gs.best_score_)
print(gs.best_params_)
bp = gs.best_params_
forest = RandomForestRegressor(criterion=bp['criterion'],
                              min_samples_leaf=bp['min_samples_leaf'],
                              min_samples_split=bp['min_samples_split'],
                              max_depth=bp['max_depth'],
                              n_estimators=bp['n_estimators'])
forest.fit(X_train, y_train)
# Explained variance score: 1 is perfect prediction
print('Score: %.2f' % forest.score(X_val, y_val))
#NOW WITH COMPLETE GIVEN DATA
forest.fit(X, Y)
print('Score: %.2f' % forest.score(X_val, y_val))
print(forest.predict(X_TEST_DATA))
prediction = forest.predict(X_TEST_DATA)
import csv
for i in range(len(prediction)):
    row = [df_X_test['id'][i], prediction[i]]
    with open('output.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)

csvFile.close()