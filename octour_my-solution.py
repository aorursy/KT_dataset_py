import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D 

import math



from sklearn.decomposition import PCA

from sklearn import model_selection

from sklearn import metrics

from sklearn import preprocessing 

from sklearn import neighbors, tree, ensemble, linear_model



import xgboost as xgb



import warnings

warnings.filterwarnings("ignore")



pd.options.display.float_format = '{:.1f}'.format
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

subm_df_raw = pd.read_csv('../input/sample_submission.csv')
train_df['date'] = pd.to_datetime(train_df.date)

train_df['month'] = train_df.date.dt.month

train_df['year'] = train_df.date.dt.year

train_df['dayofweek'] = train_df.date.dt.dayofweek



test_df['date'] = pd.to_datetime(test_df.date)

test_df['month'] = test_df.date.dt.month

test_df['year'] = test_df.date.dt.year

test_df['dayofweek'] = test_df.date.dt.dayofweek



test_df = test_df.drop(['id'], 1)
print('train_df info')

print(train_df.info())

print('')

print('train_df describe')

print(train_df.describe())

print('')

print('test_df info')

print(test_df.info())

print('')

print('test_df describe')

print(test_df.describe())
plt.figure(figsize=(7, 10))



plt.subplot(311)

plt.xticks(range(0, 55, 5))

plt.grid()

plt.title('Sales by item')

_ = plt.plot(train_df.groupby(['item'])['sales'].sum())



plt.subplot(312)

plt.xticks(range(1, 11))

plt.grid()

plt.title('Sales by store')

_ = plt.plot(train_df.groupby(['store'])['sales'].sum())



plt.subplot(313)

plt.xticks(range(2013, 2018))

plt.grid()

plt.title('Sales by year')

_ = plt.plot(train_df.groupby(train_df.date.dt.year)['sales'].sum())
plt.figure(figsize=(15, 15))



plt.subplot(321)

for i in range(1, 11):

    plt.plot(train_df.sales[(train_df.store == i) ].groupby(train_df.date.dt.year).sum(), label='Store ' + str(i))

plt.title('Year, sum by store.')

plt.xticks(range(2013, 2018))

plt.grid()

plt.legend(loc='upper left')



plt.subplot(322) 

for i in range(1, 51):

    plt.plot(train_df.sales[(train_df.item == i) ].groupby(train_df.date.dt.year).sum(), label='Item' + str(i))

plt.title('Year, sum by item.')

plt.xticks(range(2013, 2018))

plt.grid()



plt.subplot(323)

for i in range(1, 11):

    plt.plot(train_df.sales[(train_df.store == i)].groupby(train_df.date.dt.month).sum(), label='Store ' + str(i))

plt.title('Month, sum by store.')

plt.xticks(range(1, 13))

plt.grid()

plt.legend(loc='upper left')



plt.subplot(324)

for i in range(1, 51):

    plt.plot(train_df.sales[(train_df.item == i)].groupby(train_df.date.dt.month).sum(), label='Item ' + str(i))

plt.title('Month, sum by item.')

plt.xticks(range(1, 13))

plt.grid()



plt.subplot(325)

for i in range(1, 11):

    plt.plot(train_df.sales[(train_df.store == i)].groupby(train_df.date.dt.dayofweek).sum(), label='Store ' + str(i))

plt.title('Day of week, sum by store.')

plt.grid()

plt.legend(loc='upper left')



plt.subplot(326)

for i in range(1, 51):

    _ = plt.plot(train_df.sales[(train_df.item == i)].groupby(train_df.date.dt.dayofweek).sum(), label='Item ' + str(i))

plt.title('Day of week, sum by item.')

plt.grid()
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111, projection='3d')



mark = ['', 'o', '^', '.', 'x', '^', 'o', '^', '.', 'x', 'o']

for i in range(1, 4):

    ax.scatter(train_df.date.dt.month[train_df.store == i+2], 

               train_df.item[train_df.store == i+2], 

               train_df.sales[train_df.store == i+2], marker=mark[i])



plt.xticks(range(1, 13))

plt.yticks(range(1, 55, 5))



ax.set_xlabel('Month')

ax.set_ylabel('item')

ax.set_zlabel('Sales')  

ax.view_init(elev=10., azim=45)
pca = PCA(n_components=3)

pca_df = train_df.drop(['date'], 1).copy()

pca_df = pd.DataFrame(preprocessing.StandardScaler().fit_transform(pca_df.values),

                      columns=pca_df.columns, index=pca_df.index)

pca_result = pca.fit_transform(pca_df)



pca_df['pca-one'] = pca_result[:,0]

pca_df['pca-two'] = pca_result[:,1] 

pca_df['pca-three'] = pca_result[:,2]



# For reproducability of the results

np.random.seed(42)

rndperm = np.random.permutation(pca_df.shape[0])
# 2d scatter

plt.figure(figsize=(16,10))

_ = sns.scatterplot(x="pca-one", y="pca-two",

                    hue="store",

                    palette=sns.color_palette("hls", 10),

                    data=pca_df.loc[rndperm, :],

                    alpha=0.3)
# 3d scatter

ax = plt.figure(figsize=(16,10)).gca(projection='3d')

ax.scatter(xs=pca_df.loc[rndperm,:]["pca-one"], 

           ys=pca_df.loc[rndperm,:]["pca-two"], 

           zs=pca_df.loc[rndperm,:]["pca-three"], 

           c=pca_df.loc[rndperm,:]["store"], 

           cmap='tab10')



ax.set_xlabel('pca-one')

ax.set_ylabel('pca-two')

ax.set_zlabel('pca-three')

plt.show()
pca_targ = train_df['sales']

pca_df = train_df.drop(['date'], 1).copy()

pca_df = pd.DataFrame(preprocessing.StandardScaler().fit_transform(pca_df.values),

                      columns=pca_df.columns, index=pca_df.index)



pca = PCA(n_components=5)

pca_result = pca.fit_transform(pca_df)

pca_features = pd.DataFrame(pca_result)



X_test = pd.DataFrame(preprocessing.StandardScaler().fit_transform(test_df.drop(['date'], 1).values),

                      index=test_df.drop(['date'], 1).index,

                      columns=test_df.drop(['date'], 1).columns)

pca_test = pd.DataFrame(pca.fit_transform(X_test))
features = train_df.drop(['date', 'sales'], 1)

targets = train_df['sales']



features = preprocessing.StandardScaler().fit_transform(features.values)



X_test = pd.DataFrame(preprocessing.StandardScaler().fit_transform(test_df.drop(['date'], 1).values),

                      index=test_df.drop(['date'], 1).index,

                      columns=test_df.drop(['date'], 1).columns)



folds = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
class SKLEstimator:    



    def __init__(self, estimator, param_grid, folds=folds, verbose=0):

        self.grid = model_selection.GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='neg_mean_squared_error', cv=folds, verbose=verbose, n_jobs=-1)

        self.estimator = estimator

    

  

    def fit_grid(self, X, y=targets, verbose=False):

        self.grid.fit(X, y)

        if verbose:

            print('')

            print('Best score: {:.2f}'.format(math.sqrt(abs(self.grid.best_score_))))

            print('Best parameters: {}'.format(self.grid.best_params_))

      

    def get_best_estimator(self):

      

        return self.grid.best_estimator_

 



    def best_params(self):

      

        return self.grid.best_params_

      

      

    def train_estimator(self, X, y=targets, folds=folds, verbose=False):

      

        scores = []



        for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):



            X_train, X_valid = X[train_index], X[valid_index]

            y_train, y_valid = y[train_index], y[valid_index]



            self.grid.best_estimator_.fit(X_train, y_train)

            y_pred_valid = self.grid.best_estimator_.predict(X_valid).reshape(-1)



            scores.append(metrics.mean_squared_error(y_valid, y_pred_valid))

      

        if verbose:

            print('Mean of CV MSE scores (on train/valid set): {0:.2f}'.format(math.sqrt(abs(np.mean(scores)))))  

      

        return scores

    

    def predict_targets(self, X_test):

      

        return self.grid.best_estimator_.predict(X_test)    
submit_df = subm_df_raw.copy()

def submit_preds(estimator, name, X_test=X_test, submit_df=submit_df, to_csv=False):



    submit_df['sales'] = estimator.predict(X_test).astype(int) 

    name_csv = name + ".csv"

    

    if to_csv:

        submit_df.to_csv(name_csv, index=False)

    

    return submit_df
grid_linreg = {'fit_intercept': [True, False]}

linreg_estim = SKLEstimator(linear_model.LinearRegression(), grid_linreg)

linreg_estim.fit_grid(features, verbose=True)
### Lasso estimator

grid_lasso = {'alpha': [1e-3, 1e-2, 1e-1, 1, 2],

              'fit_intercept': [True, False],

              'tol': [1e-4, 1e-3, 1e-1, 5e-1]}

lasso_estim = SKLEstimator(linear_model.Lasso(), grid_lasso)

#lasso_estim.fit_grid(features, verbose=True)
### Ridge estimator

grid_ridge = {'alpha': [1e-3, 1e-2, 1e-1],

              'fit_intercept': [True, False],

              #'tol': [1e-4, 1e-3, 1e-1, 5e-1],

              'solver': ['svd', 'lsqr']}

ridge_estim = SKLEstimator(linear_model.Ridge(), grid_ridge, verbose=1)

#ridge_estim.fit_grid(features, verbose=True)
### Elnet estimator

grid_elnet = {'alpha': [1e-3, 1e-1, 1, 2],

              'l1_ratio': [1e-3, 1e-2, 5e-1],

              'fit_intercept': [True, False]}

elnet_estim = SKLEstimator(linear_model.ElasticNet(), grid_elnet, verbose=1)

#elnet_estim.fit_grid(features, verbose=True)
grid_knn = {'n_neighbors': [3, 5, 10],

            'weights': ['uniform', 'distance'],

            'algorithm': ['ball_tree', 'kd_tree']}

knn_estim = SKLEstimator(neighbors.KNeighborsRegressor(), grid_knn, folds=folds, verbose=1)

#knn_estim.fit_grid(features, verbose=True)



# Best score: 11.60394831102819

# Best parameters: {'n_neighbors': 3, 'weights': 'uniform'}

  

# Best score: 8.826924270536434

# Best parameters: {'algorithm': 'kd_tree', 'n_neighbors': 10, 'weights': 'distance'}



# Best score: 8.826924270536434

# Best parameters: {'n_neighbors': 10, 'weights': 'distance'}
grid_dt = {'splitter': ['best', 'random'],

           'max_depth': [10, 20, 50],

           'min_samples_split': [10, 30],

           'min_samples_leaf': [5, 15],

           'min_weight_fraction_leaf': [0, 0.2],          

          }

dt_estim = SKLEstimator(tree.DecisionTreeRegressor(), grid_dt, verbose=1)

#dt_estim.fit_grid(features, verbose=True) # may take time to run!

#submit_df = submit_preds(dt_estim.get_best_estimator(), "DT_3", to_csv=True)



### PCA

#dt_estim = SKLEstimator(tree.DecisionTreeRegressor(), grid_dt, verbose=1)

#dt_estim.fit_grid(pca_features, pca_targ, verbose=True)

#submit_df = submit_preds(dt_estim.get_best_estimator(), "DT_pca1", pca_test, to_csv=True)





# Best score: 8.97

# Best parameters: {'max_depth': None, 'min_samples_leaf': 5, 'min_samples_split': 10, 'min_weight_fraction_leaf': 0, 'splitter': 'best'}
grid_rf = {'n_estimators': [10, 100],

           'max_depth': [2, 3, None],

           'min_samples_split': [2, 30],

           'min_samples_leaf': [1, 5],

           'min_weight_fraction_leaf': [0, 0.5]}



grid_rf = {'n_estimators': [100],

           'max_depth': [None],

           'min_samples_split': [2],

           'min_samples_leaf': [5],

           'min_weight_fraction_leaf': [0]}



rf_estim = SKLEstimator(ensemble.RandomForestRegressor(), grid_rf, folds=folds, verbose=1)

#rf_estim.fit_grid(features, verbose=True) # may take time to run!





### PCA:

#rf_estim = SKLEstimator(ensemble.RandomForestRegressor(), grid_rf, folds=folds, verbose=1)

#rf_estim.fit_grid(pca_features, pca_targ, verbose=True)

#submit_df = submit_preds(rf_estim.get_best_estimator(), "RF_pca1", pca_test, to_csv=True)



### Run 1:

#Best score: 7.652671941592358

#Best parameters: {'max_depth': None, 'min_samples_leaf': 5, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0, 'n_estimators': 100}
features_xgb = pd.DataFrame(preprocessing.StandardScaler().fit_transform(train_df.drop(['date', 'sales'], 1).values),

                            index=train_df.drop(['date', 'sales'], 1).index, 

                            columns=train_df.drop(['date', 'sales'], 1).columns)



dtrain = xgb.DMatrix(features_xgb, label=targets, feature_names=list(features_xgb.columns))

dtest = xgb.DMatrix(X_test, feature_names=list(X_test.columns))
# LB private / public: 24.77016 / 15.06425

xgb_grid = {'max_depth': 3, 'eta': 5e-1} # 'subsample': 1

#xgb_estim = xgb.train(xgb_grid, dtrain, 1500)

#submit_df = submit_preds(xgb_estim, "XGB_2", X_test=dtest, to_csv=True)
# LB private / public: 24.21677 / 14.81929

xgb_grid = {'max_depth': 5, 'eta': 5e-1}

#xgb_estim = xgb.train(xgb_grid, dtrain, 1500)

#submit_df = submit_preds(xgb_estim, "XGB_4", X_test=dtest, to_csv=True)
# LB private / public: 24.20192 / 14.78730

xgb_grid = {'max_depth': 5, 'eta': 2e-1}

xgb_estim = xgb.train(xgb_grid, dtrain, 1500)

submit_df = submit_preds(xgb_estim, "XGB_1", X_test=dtest, to_csv=True)
# LB private / public: 24.33419 / 14.78021

xgb_grid = {'max_depth': 5, 'eta': 1e-1}

#xgb_estim = xgb.train(xgb_grid, dtrain, 1500)

#submit_df = submit_preds(xgb_estim, "XGB_9", X_test=dtest, to_csv=True)
# LB private / public: 

xgb_grid = {'booster': 'dart', 

            'max_depth': 5, 

            'eta': 5e-1}

#xgb_estim = xgb.train(xgb_grid, dtrain, 1500)

#submit_df = submit_preds(xgb_estim, "XGB_11", X_test=dtest, to_csv=True)
dtrain_pca = xgb.DMatrix(pca_features, label=pca_targ)

dtest_pca = xgb.DMatrix(pca_test)
# LB private / public: 56.64127 / 56.36665

xgb_grid = {'max_depth': 5, 'eta': 2e-1}

#xgb_estim = xgb.train(xgb_grid, dtrain_pca, 1500)

#submit_df = submit_preds(xgb_estim, "XGB_pca_1", X_test=dtest_pca, to_csv=True)
# LB private / public: 23.41066 / 14.79618

grid_extree = {'n_estimators': [100],

               'max_depth': [None],

               'min_samples_split': [2],

               'min_samples_leaf': [1],

               'bootstrap': [False],

              }

extree_estim = SKLEstimator(ensemble.ExtraTreesRegressor(n_jobs=-1), grid_extree, verbose=1)

extree_estim.fit_grid(features, verbose=True) # may take time to run!

submit_df = submit_preds(extree_estim.get_best_estimator(), "extree_1", to_csv=True)





### PCA: 

#extree_estim = SKLEstimator(ensemble.ExtraTreesRegressor(n_jobs=-1), grid_extree, verbose=1)

#extree_estim.fit_grid(pca_features, pca_targ, verbose=True)

#submit_df = submit_preds(extree_estim.get_best_estimator(), "extree_pca_2", pca_test, to_csv=True)



#Best score: 12.03

#Best parameters: {'bootstrap': False, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
# LB private / public: 24.61912 / 15.31381

grid_gboost = {'learning_rate': [1e-2, 1e-1, 5e-1, 1],

               'n_estimators': [100, 200],

               'min_samples_split': [2, 3],

               'min_samples_leaf': [1, 3],

               'max_depth': [3, 5],

              }



best_params = {'learning_rate': [5e-1],

               'max_depth': [5],

               'min_samples_leaf': [3],

               'min_samples_split': [2],

               'n_estimators': [200]}



#gboost_estim = SKLEstimator(ensemble.GradientBoostingRegressor(), best_params, verbose=1)

#gboost_estim.fit_grid(features, verbose=True) # may take time to run!



#Best score: 16.00

#Best parameters: {'learning_rate': 0.1, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100}