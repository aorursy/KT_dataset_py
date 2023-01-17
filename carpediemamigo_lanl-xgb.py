import multiprocessing

n_jobs = multiprocessing.cpu_count()-1



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA as RandomizedPCA



from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.model_selection import learning_curve, validation_curve

from sklearn.model_selection import KFold, ShuffleSplit, StratifiedShuffleSplit



from time import time, ctime



import xgboost
def changeparam(value, param):

    if param == 'max_depth' or param == 'min_child_weight':

        if value >= 1 and value <= 10:

            return list(filter(lambda x: x != 0, [ value + i for i in [-1, 0, 1]]))

        else: 

            return value

    elif param == 'n_estimators':

        if value >= 1 and value < 10:

            return list(filter(lambda x: x != 0, [ value + i for i in [-1, 0, 1]]))

        elif value >= 10 and value < 100:

            return [ value + i for i in [-5, 0, 5]]

        elif value >= 100 and value < 1000:

            return [ value + i for i in [-20, -10, 0, 20, 10]]

        elif value >= 1000 and value < 10000:

            return [ value + i for i in [-200, -100, 0, 100, 200]]

        elif value >= 10000:

            return [ value + i for i in [-2000, -1000, 0, 1000, 2000]]



def plotCurves(model, X, Y, param_range, param_name, scoring):

    best_values = []

 

    for i, st_i in enumerate(scoring):

            fig, ax = plt.subplots( nrows = 1, ncols = 2, figsize=(18, 5))



            train_scores, test_scores = validation_curve(

                model, X, Y.values.ravel(), param_name=param_name, param_range=param_range,

                cv=cv, scoring=scoring[i], n_jobs = n_jobs, error_score=0)



            train_scores_mean = np.mean(train_scores, axis=1)

            train_scores_std = np.std(train_scores, axis=1)



            test_scores_mean = np.mean(test_scores, axis=1)

            test_scores_std = np.std(test_scores, axis=1)



            ax[0].plot(param_range, train_scores_mean, label="Training score", color="darkorange")

            ax[0].fill_between(param_range, train_scores_mean - train_scores_std,

                         train_scores_mean + train_scores_std, alpha=0.2,

                         color="darkorange")



            ax[0].plot(param_range, test_scores_mean, label="Cross-validation score", color="navy")

            ax[0].fill_between(param_range, test_scores_mean - test_scores_std,

                         test_scores_mean + test_scores_std, alpha=0.2,

                         color="navy")

            ax[0].legend(loc="best")



            ax[0].set_title('Validation Curve for '+scoring[i])

            ax[0].set_ylabel(scoring[i])

            ax[0].set_xlabel(param_name)



            ax[0].set_ylim(test_scores_mean[-1] - abs(6*test_scores_std[-1]), 

                           test_scores_mean[-1] + abs(6*test_scores_std[-1]))

            ax[0].grid(True)



            best_param = list(param_range)[np.argmax(test_scores_mean)]

            

            if param_name == 'n_estimators':

                model.n_estimators = best_param

            elif param_name == 'max_depth':

                model.max_depth = best_param

            elif param_name == 'min_child_weight':

                model.min_child_weight = best_param



            best_values.append(best_param)



            sizes = np.linspace(.1, 1.0, 10)

            train_sizes, train_scores, test_scores = learning_curve(model, 

                                                                    X, Y.values.ravel(), 

                                                                    cv = cv, 

                                                                    scoring=scoring[i],

                                                                    n_jobs = n_jobs, 

                                                                    train_sizes = sizes,

                                                                    error_score=0)

            train_scores_mean = np.mean(train_scores, axis=1)

            train_scores_std = np.std(train_scores, axis=1)

            test_scores_mean = np.mean(test_scores, axis=1)

            test_scores_std = np.std(test_scores, axis=1)



            ax[1].fill_between(sizes, train_scores_mean - train_scores_std,

                             train_scores_mean + train_scores_std, alpha=0.1,

                             color="r")

            ax[1].fill_between(sizes, test_scores_mean - test_scores_std,

                             test_scores_mean + test_scores_std, alpha=0.1, color="g")

            ax[1].plot(sizes, train_scores_mean, 'o-', color="r",

                     label="Training score")

            ax[1].plot(sizes, test_scores_mean, 'o-', color="g",

                     label="Cross-validation score")



            ax[1].set_title('Learning curve for '+scoring[i] +' for ' + param_name + ' = '+ str(best_param) )

            ax[1].set_ylabel(scoring[i])

            ax[1].set_xlabel('Number training observations')

            ax[1].grid(True)

            ax[1].legend(loc="best")



            plt.pause(0.01)

    fig.tight_layout()

    return best_values



def plotfig (ypred, yactual, strtitle):

    plt.scatter(ypred, yactual.values.ravel())

    plt.title(strtitle)

    plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])

    plt.xlim(0, 20)

    plt.ylim(0, 20)

    plt.xlabel('Predicted', fontsize=12)

    plt.ylabel('Actual', fontsize=12)

    plt.show()



# Utility function to report best scores

def report(results, n_top=3):

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")

            

def mode_custom(List): 

    dict = {} 

    count, itm = 0, '' 

    for item in reversed(List): 

        dict[item] = dict.get(item, 0) + 1

        if dict[item] >= count : 

            count, itm = dict[item], item 

    return(itm) 
XY = pd.read_csv('../input/LANL_train.csv')

X_TEST = pd.read_csv('../input/LANL_test.csv')

col = [c for c in XY.columns if c not in ['time_to_failure']]

X = XY[col]

Y = XY['time_to_failure']

print(X.shape, X_TEST.shape, Y.shape)
print(X.isnull().values.any())

print(X_TEST.isnull().values.any())
# Check the number of features with one unique value 

allunique = X.nunique().reset_index()

lst = [allunique.loc[i,'index'] for i in range(len(allunique)) if allunique.loc[i,0] == 1]

print(len(lst), lst)
# Drop colums with one unique value (in this case this is null)

X.drop(lst, axis = 1, inplace = True)

X_TEST.drop(lst, axis = 1, inplace = True)

print(X.shape, X_TEST.shape)
scaler = StandardScaler()



scaler.fit(X)

X = scaler.transform(X)



scaler.fit(X_TEST)

X_TEST = scaler.transform(X_TEST)
# features = [5,6,7,9,10,11,12,16,18,19,21,22,24,26,28,29,32,35,37,38,39,41,42,44,45,

#             46,47,48,49,50,51,52,54,55,56,57,58,62,64,66,68,69,71,72,73,74,76,78,79,

#             82,86,89,90,91,96,97,98,99,100,102,103,104,105,107,108,109,110,113,120,

#             121,123,126,127,128,129,130,131,133,134]

# X = X.iloc[:,features]

# X_TEST = X_TEST.iloc[:,features]

# print(X.shape, X_TEST.shape, Y.shape)



# X = scaler.fit(X).transform(X)

# X_TEST = scaler.fit(X_TEST).transform(X_TEST)
# pca = RandomizedPCA(copy = True, 

#                     iterated_power = 3,

#                     n_components = 55, 

#                     svd_solver='randomized', 

#                     random_state = 0, 

#                     whiten=False).fit(X)



# plt.plot(np.cumsum(pca.explained_variance_ratio_))

# plt.grid()



# X = pca.transform(X)

# print('Dimension of train dataset', X.shape)



# X_TEST = pca.transform(X_TEST)

# print('Dimension of test dataset', X_TEST.shape)
n_fold = 5



cv = ShuffleSplit(n_splits=n_fold, test_size=0.4, random_state = 0)

# cv = KFold(n_splits=n_fold, shuffle=True, random_state = 0)
# scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 

#            'neg_mean_squared_log_error','neg_median_absolute_error',

#            'explained_variance']



scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 

           'neg_median_absolute_error', 'explained_variance']



n_estimators_range = range(10, 5000, 500) 

max_depth_range = np.arange(1, 11, 2)

min_child_weight_range = np.arange(1, 11, 2)



model = xgboost.XGBRegressor(learning_rate = 0.01, 

                            criterion = 'mae',

                            subsample = 1,

                            colsample_bytree = 1,

                            max_depth = 1,

                            min_child_weight = 1

                            )



model
best_n_estimators = plotCurves(model, X, Y, n_estimators_range, 'n_estimators', scoring)
model.n_estimators = mode_custom(best_n_estimators)

model
best_max_depth = plotCurves(model, X, Y, max_depth_range, 'max_depth', scoring)
model.max_depth = mode_custom(best_max_depth)

model
best_min_child_weight = plotCurves(model, X, Y, min_child_weight_range, 'min_child_weight', scoring)
model.min_child_weight = mode_custom(best_min_child_weight)

ch_algorithm = model
# random_grid = {'n_estimators': n_estimators_range,

#                'learning_rate': [0.01],

#                'max_depth': max_depth_range,

#                'min_child_weight': min_child_weight_range,

#                'subsample' : [1],

#                'colsample_bytree' : [1],

#                'criterion' : ['mae'],

#               }



# model = xgboost.XGBRegressor()



# n_iter_search = 20

# random_search = RandomizedSearchCV(estimator = model, 

#                                     param_distributions = random_grid, 

#                                     n_iter = n_iter_search, 

#                                     cv = cv, 

#                                     scoring = 'neg_mean_absolute_error',

#                                     verbose = 10, 

#                                     error_score = 0,

#                                     random_state = 42,

#                                     n_jobs = n_jobs)



# start = time()

# random_search.fit(X, Y)

# print("RandomizedSearchCV took %.2f seconds for %d candidates"

#       " parameter settings." % ((time() - start), n_iter_search))

# report(random_search.cv_results_)



# ch_algorithm = random_search.best_estimator_
%%time



search_grid = {'n_estimators': changeparam(ch_algorithm.n_estimators, 'n_estimators'),

               'learning_rate': [0.01],

               'max_depth': changeparam(ch_algorithm.max_depth, 'max_depth'),

               'min_child_weight': changeparam(ch_algorithm.min_child_weight, 'min_child_weight'),

               'subsample' : [0.5, 1],

               'colsample_bytree' : [0.5, 1],

               'criterion' : ['mae'],

              }



n_iter_search = sum([len(value) for key, value in search_grid.items()])

model = xgboost.XGBRegressor()



grid = GridSearchCV(model, 

                    search_grid, 

                    cv=cv, 

                    scoring = 'neg_mean_absolute_error', 

                    verbose = 10,

                    error_score = 0,

                    n_jobs = n_jobs)



start = time()

grid.fit(X, Y)

print("RandomizedSearchCV took %.2f seconds for %d candidates"

      " parameter settings." % ((time() - start), n_iter_search))



report(grid.cv_results_)



ch_algorithm = grid.best_estimator_
ch_algorithm
ch_algorithm.n_estimators =ch_algorithm.n_estimators*2

# ch_algorithm.learning_rate = ch_algorithm.learning_rate / 2

ch_algorithm.n_jobs = n_jobs
ch_algorithm
%%time



submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

oof = np.zeros(len(X))

prediction = np.zeros(len(submission))

mae, r2 = [], []



for fold_n, (train_index, valid_index) in enumerate(cv.split(X)):

    print('\nFold', fold_n, 'started at', ctime())



    X_train = X[train_index]

    X_valid = X[valid_index]

    Y_train = Y.values.ravel()[train_index]

    Y_valid = Y.values.ravel()[valid_index]

       

    best_model = ch_algorithm.fit(X_train, Y_train)

    y_pred = best_model.predict(X_valid)   

  

    oof[valid_index] = y_pred



    mae.append(mean_absolute_error(Y_valid, y_pred))

    r2.append(r2_score(Y_valid, y_pred))



    print('MAE: ', mean_absolute_error(Y_valid, y_pred))

    print('R2: ', r2_score(Y_valid, y_pred))



    prediction += best_model.predict(X_TEST)

        

prediction /= n_fold



print('='*45)

print('CV mean MAE: {0:.4f}, std: {1:.4f}.'.format(np.mean(mae), np.std(mae)))

print('CV mean R2:  {0:.4f}, std: {1:.4f}.'.format(np.mean(r2), np.std(r2)))



plotfig(best_model.predict(X), Y, 'Predicted vs. Actual responses for XGB')



# non_zeros_ind = np.argwhere(oof != 0)[:, 0]

# oof     = oof[non_zeros_ind]

# Y_fixed = pd.DataFrame(Y.values[non_zeros_ind])

# plotfig(oof, Y_fixed, 'Predicted vs. Actual responses for XGB')

# print('\nMAE: ', mean_absolute_error(Y_fixed, oof))
# scores = cross_validate(best_model, X, Y, cv=cv, scoring = {'neg_mean_absolute_error', 'r2'})



# print('CV mean MAE: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores['test_neg_mean_absolute_error']), 

#                                                    np.std(scores['test_neg_mean_absolute_error'])))

# print('CV mean R2:  {0:.4f}, std: {1:.4f}.'.format(np.mean(scores['test_r2']), 

#                                                    np.std(scores['test_r2'])))
submission['time_to_failure'] = prediction 

print(submission.head())

submission.to_csv('submission.csv')