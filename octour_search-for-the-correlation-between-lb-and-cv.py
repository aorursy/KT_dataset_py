import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV

from sklearn import model_selection

from sklearn import metrics

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve    

from sklearn import linear_model 

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest, f_classif



import IPython



import warnings



warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

subm_df = pd.read_csv('../input/sample_submission.csv')
pd.set_option('max_rows', 105)

pd.set_option('max_columns', 105)

pd.set_option('max_colwidth', 150)

pd.options.display.float_format = '{:.3f}'.format
X_train = train_df.drop(['id', 'target'], axis=1)

y_train = train_df['target']

X_test = test_df.drop(['id'], axis=1)



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



n_fold = 20

folds = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
class MyEstimator:    



    def __init__(self, estimator, param_grid, folds):

        self.grid = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='roc_auc', cv=folds, verbose=0, n_jobs=-1)

        self.estimator = estimator





    def get_best_estimator(self):



        return self.grid.best_estimator_





    def best_params(self):



        return self.grid.best_params_





    def fit_grid(self, X, y, verbose=False):

        self.grid.fit(X, y)

        if verbose:

            print('Best Parameters', self.grid.best_params_)





    def train_estimator(self, X, X_test, y=y_train, folds=folds, verbose=False):



        scores = []



        for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):



            X_train, X_valid = X[train_index], X[valid_index]

            y_train, y_valid = y[train_index], y[valid_index]



            self.grid.best_estimator_.fit(X_train, y_train)

            y_pred_valid = self.grid.best_estimator_.predict(X_valid).reshape(-1)



            scores.append(roc_auc_score(y_valid, y_pred_valid))



            if verbose:

                print('CV mean AUC score (on train/valid set): {0:.4f}, std: {1:.4f}'.format(np.mean(scores), np.std(scores)))  



        return scores





    def predict_probabilities(self, X, target_value=1):



        return self.grid.best_estimator_.predict_proba(X)[:, target_value]





    def predict_targets(self, X):



        return self.grid.best_estimator_.predict(X)





    def plot_probabilities(self, preds_proba, num_rows=1, num_cols=2, i=1, y_scale=1000, title_sbp='Hist', verbose=False):



        if verbose:

            print('Proba mean: {0:5f}'.format(preds_proba.mean()))



        plt.subplot(num_rows, num_cols, i, facecolor='slategrey')

        plt.hist(preds_proba, bins=40, color='powderblue')

        plt.xticks(np.arange(0, 1.1, 0.1))

        plt.yticks(range(0, y_scale, 100))

        plt.grid(color='royalblue', linestyle=':', lw=1.3)

        plt.title(title_sbp)

# this dataframe should be used for sumbission. Such a way we'll not mess up with the original dataset.

submit_df = pd.DataFrame()

submit_df['id'] = subm_df['id']
# to convert to CSV, enable argument `to_csv`

# as parameters for estimator should be used k_dict

# currently function supports only logreg and elastic net estimators

def run_and_submit(k_dict, to_csv=False, estimator_type=linear_model.LogisticRegression(), X_train=X_train, y_train=y_train, X_test=X_test, folds=folds, submit_df=submit_df):



    for k_val in k_dict.keys():



        # if we have unique k values in dict, they will be int. 

        # But if there are several runs in the dict with the same k_value - we need

        # to diffirintiate them, converting in str type

        if type(k_val) == str:

            k_ind = int(k_val[:2]) # we're not gonna use more than 99 features here

        else:

            k_ind = k_val



        selector = SelectKBest(f_classif, k=k_ind)

        X_train_K = selector.fit_transform(X_train, y_train.values)

        X_test_K = selector.transform(X_test)



        params_ = k_dict[k_val][0]



        estimator = MyEstimator(estimator_type, params_, folds)

        estimator.fit_grid(X_train_K, y_train)

        scores = estimator.train_estimator(X_train_K, X_test_K) 



        if to_csv:



            if isinstance(estimator_type, linear_model.LogisticRegression):

                submit_df['target'] = estimator.predict_probabilities(X_test_K)

                name_csv = "subm_LR_Kbst{}_C{}_{}_CV{}.csv".format(str(k_ind), 

                                                                   str(estimator.grid.best_params_['C'])[1:],

                                                                   str(k_dict[k_val][0]['penalty'])[2:4],

                                                                   str(np.mean(scores))[1:6])

                submit_df.to_csv(name_csv, index=False)





            elif isinstance(estimator_type, linear_model.ElasticNet):



                submit_df['target'] = estimator.predict_targets(X_test_K)



                Lasso = False

                if isinstance(estimator_type, linear_model.Lasso):

                    Lasso = True



                if Lasso:

                    name_csv = "subm_Lasso_Kbst{}_alpha{}_CV{}.csv".format(str(k_ind), 

                                                                           str(estimator.grid.best_params_['alpha'])[1:],

                                                                           str(np.mean(scores))[1:6])

                else:

                    name_csv = "subm_ElNet_Kbst{}_alpha;l1_{};{}_CV{}.csv".format(str(k_ind), 

                                                                                  str(estimator.grid.best_params_['alpha'])[1:],

                                                                                  str(estimator.grid.best_params_['l1_ratio'])[1:],

                                                                                  str(np.mean(scores))[1:6])



                submit_df.to_csv(name_csv, index=False)





            else:

                print('Submission for this type of estimator is not avialiable yet')

        
# as parameters for estimator should be used k_dict

# currently function supports only logreg and elastic net estimators

def plot_results(k_dict, result_cols, estimator_type=linear_model.LogisticRegression(), figsize=(22, 12), rows=3, cols=3, y_scale=1000, X_train=X_train, y_train=y_train, X_test=X_test, folds=folds):



    # function will plot probabilities for certain parameters of estimator

    # returns dataframe with results

    i = 1

    result_df = pd.DataFrame(columns=result_cols)

    plt.figure(figsize=figsize)



    for k_val in k_dict.keys():

    

        # if we have unique k values in dict, they will be int. 

        # But if there are several runs in the dict with the same k_value - we need

        # to diffirintiate them, converting in str type

        if type(k_val) == str:

            k_ind = int(k_val[:2])

        else:

            k_ind = k_val

      

        selector = SelectKBest(f_classif, k=k_ind)

        X_train_K = selector.fit_transform(X_train, y_train.values)

        X_test_K = selector.transform(X_test)



        params_ = k_dict[k_val][0]



        estimator = MyEstimator(estimator_type, params_, folds)

        estimator.fit_grid(X_train_K, y_train)

        scores = estimator.train_estimator(X_train_K, X_test_K)

    

    

        if isinstance(estimator_type, linear_model.LogisticRegression):

            result_df.loc[len(result_df)] = [k_ind, k_dict[k_val][0]['penalty'][0], k_dict[k_val][0]['C'][0], np.mean(scores), k_dict[k_val][1], k_dict[k_val][0]]

            estimator.plot_probabilities(estimator.predict_probabilities(X_test_K), 

                                       num_rows=rows, num_cols=cols, i=i, y_scale=y_scale, 

                                       title_sbp='k = {}, C = {}, {}    CV = {:.3f}, LB = {:.3f} '.format(k_ind, k_dict[k_val][0]['C'][0], k_dict[k_val][0]['penalty'][0], 

                                                                                                          np.mean(scores), k_dict[k_val][1]))

        elif isinstance(estimator_type, linear_model.ElasticNet):

      

            Lasso = False

            if isinstance(estimator_type, linear_model.Lasso):

                Lasso = True



            if Lasso:

                result_df.loc[len(result_df)] = [k_ind, k_dict[k_val][0]['alpha'][0], np.mean(scores), k_dict[k_val][1], k_dict[k_val][0]]

                estimator.plot_probabilities(estimator.predict_targets(X_test_K), 

                                             num_rows=rows, num_cols=cols, i=i, y_scale=y_scale,

                                             title_sbp='k = {}, alpha = {},    CV = {:.3f}, LB = {:.3f} '.format(k_ind, k_dict[k_val][0]['alpha'][0],

                                                                                                                 np.mean(scores), k_dict[k_val][1]))

            else:

                result_df.loc[len(result_df)] = [k_ind, k_dict[k_val][0]['alpha'][0], k_dict[k_val][0]['l1_ratio'][0], np.mean(scores), k_dict[k_val][1], k_dict[k_val][0]]

                estimator.plot_probabilities(estimator.predict_targets(X_test_K), 

                                             num_rows=rows, num_cols=cols, i=i, y_scale=y_scale,

                                             title_sbp='k = {}, alpha/l1 = {}/{},    CV = {:.3f}, LB = {:.3f} '.format(k_ind, k_dict[k_val][0]['alpha'][0], k_dict[k_val][0]['l1_ratio'][0], 

                                                                                                                       np.mean(scores), k_dict[k_val][1]))      

        else:

            print('This type of estimator is not supported yet')

        

        i += 1



    return result_df
result_cols = ['k', 'penalty', 'C', 'CV', 'LB', 'params']

# key of the dict - number of features to be selected

# second value in tuple of dict values (0.736, 0.734, etc) is the LB score for that parameters set.

k_dict_all_l1 = {39: ({'C': [0.46], 'class_weight': [None], 'fit_intercept': [False], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.736),

                 40: ({'C': [0.46], 'class_weight': [None], 'fit_intercept': [False], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.734),

                 49: ({'C': [0.46], 'class_weight': [None], 'fit_intercept': [False], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.735),

                 46: ({'C': [0.46], 'class_weight': [None], 'fit_intercept': [False], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.729),

                 47: ({'C': [0.46], 'class_weight': [None], 'fit_intercept': [False], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.730),

                 52: ({'C': [0.46], 'class_weight': [None], 'fit_intercept': [False], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.731),

                 43: ({'C': [0.45], 'class_weight': [None], 'fit_intercept': [False], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.731) }



result_all_l1 = plot_results(k_dict_all_l1, result_cols)
result_all_l1
result_cols = ['k', 'penalty', 'C', 'CV', 'LB', 'params']

k_dict_myl1 = {'15r1':  ({'C': [0.336], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.839),

               '15r2':  ({'C': [0.276], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.840),

               '15r3':  ({'C': [0.2], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.842),

               '15r4':  ({'C': [0.19], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.843),

               '15r5':  ({'C': [0.18], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.843),

               '15r6':  ({'C': [0.17], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.844),

               '15r7':  ({'C': [0.16], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.844),

               '15r8':  ({'C': [0.15], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.845),

               '15r9':  ({'C': [0.14], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.845),

               '15r10': ({'C': [0.13], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.846),

               '15r11': ({'C': [0.125], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.846),

               '15r12': ({'C': [0.12], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.846),

               '15r13': ({'C': [0.11], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.846),

               '15r14': ({'C': [0.1], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.845),

               '15r15': ({'C': [0.097], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.845)}

result_myl1_df = plot_results(k_dict_myl1, result_cols, figsize=(22, 20), rows=5, cols=3)
result_myl1_df
result_cols = ['k', 'penalty', 'C', 'CV', 'LB', 'params']

k_dict_myl2 = {'15r1':  ({'C': [0.8390], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l2'], 'solver': ['liblinear']}, 0.833),

               '15r2':  ({'C': [0.276], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l2'], 'solver': ['liblinear']}, 0.832),

               '15r3':  ({'C': [0.2], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l2'], 'solver': ['liblinear']}, 0.832),

               '15r4':  ({'C': [0.19], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l2'], 'solver': ['liblinear']}, 0.832),

               '15r5':  ({'C': [0.18], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l2'], 'solver': ['liblinear']}, 0.832),

               '15r6':  ({'C': [0.17], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l2'], 'solver': ['liblinear']}, 0.832),

               '15r7':  ({'C': [0.16], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l2'], 'solver': ['liblinear']}, 0.832),

               '15r8':  ({'C': [0.15], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l2'], 'solver': ['liblinear']}, 0.831),

               '15r9':  ({'C': [0.14], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l2'], 'solver': ['liblinear']}, 0.831),

               '15r10': ({'C': [0.13], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l2'], 'solver': ['liblinear']}, 0.831),

               '15r11': ({'C': [0.125], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l2'], 'solver': ['liblinear']}, 0.831),

               '15r12': ({'C': [0.12], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l2'], 'solver': ['liblinear']}, 0.831),

               '15r13': ({'C': [0.11], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l2'], 'solver': ['liblinear']}, 0.831),

               '15r14': ({'C': [0.1], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l2'], 'solver': ['liblinear']}, 0.831),

               '15r15': ({'C': [0.097], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l2'], 'solver': ['liblinear']}, 0.831),

                    20: ({'C': [0.097], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l2'], 'solver': ['liblinear']}, 0.802),

                    25: ({'C': [0.02], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l2'], 'solver': ['liblinear']}, 0.797),

                    34: ({'C': [0.02], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l2'], 'solver': ['liblinear']}, 0.792),

                    36: ({'C': [0.01], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l2'], 'solver': ['liblinear']}, 0.785) }

result_myl2_df = plot_results(k_dict_myl2, result_cols, figsize=(22, 28), rows=7, cols=3)
result_myl2_df
k_dict_l1l2 = {'15r1': ({'C': [0.13], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.846),

               '15r2': ({'C': [0.12], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.846),

               '15r3': ({'C': [0.11], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l1'], 'solver': ['liblinear']}, 0.846),

               '15r4':  ({'C': [0.2], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l2'], 'solver': ['liblinear']}, 0.832),

               '15r5':  ({'C': [0.276], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l2'], 'solver': ['liblinear']}, 0.832),

               '15r6':  ({'C': [0.8390], 'class_weight': ['balanced'], 'fit_intercept': [True], 'penalty': ['l2'], 'solver': ['liblinear']}, 0.833)}

_ = plot_results(k_dict_l1l2, result_cols, figsize=(22, 8), rows=2, cols=3)                          
plt.figure(figsize=(20, 6), facecolor='lightgrey')



plt.subplot(121, facecolor='slategrey')

_ = plt.plot(result_myl1_df[result_myl1_df['k'] == 15]['C'], result_myl1_df[result_myl1_df['k'] == 15]['CV'], c='salmon', ls='--')

_ = plt.plot(result_myl1_df[result_myl1_df['k'] == 15]['C'], result_myl1_df[result_myl1_df['k'] == 15]['LB'], c='navy')

plt.xlabel('C value')

plt.ylabel('AUC score')

plt.xticks(np.arange(.1, .35, .05))

plt.yticks(np.arange(.765, .850, .005))

plt.grid(color='midnightblue', linestyle=':', lw=.7)

_ = plt.legend(('CV', 'LB'), loc='lower right')

_ = plt.title('[LB and CV] versus [C value] for l1, k = 15')





plt.subplot(122, facecolor='slategrey')

_ = plt.plot(result_myl2_df[(result_myl2_df['k'] == 15)]['C'], 

             result_myl2_df[(result_myl2_df['k'] == 15)]['CV'], c='salmon', ls='--') 



_ = plt.plot(result_myl2_df[result_myl2_df['k'] == 15]['C'], 

             result_myl2_df[result_myl2_df['k'] == 15]['LB'], c='navy')

plt.xlabel('C value')

plt.ylabel('AUC score')

plt.xticks(np.arange(.1, 0.8, .05))

plt.yticks(np.arange(.765, .850, .005))

plt.grid(color='midnightblue', linestyle=':', lw=.7)

_ = plt.legend(('CV', 'LB'), loc='lower right')

_ = plt.title('[LB and CV] versus [C value] for l2, k = 15')
dict_lasso = {15: ({'alpha': [0.03], 'tol': [1e-7], 'fit_intercept': [True]}, 0.847), 

              16: ({'alpha': [0.03], 'tol': [1e-7], 'fit_intercept': [False]}, 0.842), 

              13: ({'alpha': [0.03], 'tol': [1e-2], 'fit_intercept': [True]}, 0.839), 

              14: ({'alpha': [0.03], 'tol': [1e-7], 'fit_intercept': [False]}, 0.842)}
run_and_submit(dict_lasso, to_csv=False, estimator_type=linear_model.Lasso())
result_cols = ['k', 'alpha', 'CV', 'LB', 'params']

df_lasso = plot_results(dict_lasso, result_cols, estimator_type=linear_model.Lasso(), figsize=(16, 8), rows=2, cols=2, y_scale=1500)
df_lasso
# Reset before new run

list_k = []

list_mean = []

list_std = []

list_CV = []

list_C = []

list_params = []





resdict = {'k': list_k,

           'mean': list_mean,

           'std': list_std,

           'CV': list_CV,

           'alpha/l1_ratio': list_C,

           'params': list_params}



a = False # takes time to run



if a:

    for k_val in range(10, 21):

        selector = SelectKBest(f_classif, k=k_val)

        X_train_K = selector.fit_transform(X_train, y_train.values.astype(int))

        X_test_K = selector.transform(X_test)



        for alpha in range(5, 31):

            alpha /= 100



            grid_elnet = {'alpha': [alpha],

                          'l1_ratio': [0.25, 0.5, 0.75],

                          'fit_intercept': [True, False], 

                          'tol': [1e-5, 1e-4, 1e-3, 1e-1], 

                          'selection': ['cyclic', 'random']}



            elnet_estim = MyEstimator(linear_model.ElasticNet(), grid_elnet, folds)

            elnet_estim.fit_grid(X_train_K, y_train)



            scores = elnet_estim.train_estimator(X_train_K, X_test_K, verbose=False)

            preds = elnet_estim.predict_targets(X_test_K)



            list_k.append(k_val)

            list_mean.append(preds.mean())

            list_std.append(np.std(scores))

            list_CV.append(np.mean(scores))

            list_C.append([elnet_estim.grid.best_params_['alpha'], elnet_estim.grid.best_params_['l1_ratio']])

            list_params.append(str(elnet_estim.grid.best_params_)[1:len(str(elnet_estim.grid.best_params_))-1])



            print(k_val, alpha) # to track the process

            

    result_df = pd.DataFrame(resdict) # results



    ### filter results



    filter_df = result_df.sort_values('CV', ascending=False).head(50)

    #filter_df.to_csv('best50_elnet.csv', index=False)

    #filter_df.head(20)

  
# LB score (second value of the tuple) taken after submission

dict_elnet = {  '14r1': ({'alpha': [0.05], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['random'], 'tol': [1e-1]}, 0.839),

                '18r2': ({'alpha': [0.06], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['cyclic'], 'tol': [1e-5]}, 0.828),

                '18r3': ({'alpha': [0.08], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['cyclic'], 'tol': [1e-5]}, 0.831),

                '18r4': ({'alpha': [0.07], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['cyclic'], 'tol': [1e-5]}, 0.830),

                '20r5': ({'alpha': [0.05], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['cyclic'], 'tol': [1e-1]}, 0.821),

                '18r6': ({'alpha': [0.09], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['cyclic'], 'tol': [1e-5]}, 0.832),

                '16r7': ({'alpha': [0.05], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['random'], 'tol': [1e-1]}, 0.834),

                '18r8': ({'alpha': [0.05], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['random'], 'tol': [1e-1]}, 0.828),

                '15r9': ({'alpha': [0.09], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['random'], 'tol': [1e-1]}, 0.843),

                '15r10': ({'alpha': [0.05], 'fit_intercept': [True], 'l1_ratio': [0.50], 'selection': ['cyclic'], 'tol': [1e-5]}, 0.845),               

                '15r11': ({'alpha': [0.06], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['random'], 'tol': [1e-1]}, 0.841),

                '15r12': ({'alpha': [0.08], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['cyclic'], 'tol': [1e-1]}, 0.843),

                '16r13': ({'alpha': [0.09], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['random'], 'tol': [1e-1]}, 0.840),

                '14r14': ({'alpha': [0.07], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['cyclic'], 'tol': [1e-1]}, 0.841),

                '15r15': ({'alpha': [0.10], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['cyclic'], 'tol': [1e-5]}, 0.844),

                '14r16': ({'alpha': [0.08], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['cyclic'], 'tol': [1e-5]}, 0.842),

                '14r17': ({'alpha': [0.06], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['cyclic'], 'tol': [1e-5]}, 0.841),

                '19r18': ({'alpha': [0.08], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['cyclic'], 'tol': [1e-3]}, 0.828),

                '19r19': ({'alpha': [0.09], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['cyclic'], 'tol': [1e-5]}, 0.830),

                '17r20': ({'alpha': [0.07], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['random'], 'tol': [1e-1]}, 0.836)}              
run_and_submit(dict_elnet, to_csv=False, estimator_type=linear_model.ElasticNet())
result_cols = ['k', 'alpha', 'l1_ratio', 'CV', 'LB', 'params']

elnet_df = plot_results(dict_elnet, result_cols, estimator_type=linear_model.ElasticNet(), figsize=(22, 28), rows=7, cols=3, y_scale=1500)                          
elnet_df.sort_values('LB', ascending=False).head(6)
dict_elnet = {  '15r1': ({'alpha': [0.05], 'fit_intercept': [True], 'l1_ratio': [0.50], 'selection': ['cyclic'], 'tol': [1e-5]}, 0.845),

                '15r2': ({'alpha': [0.10], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['cyclic'], 'tol': [1e-5]}, 0.844),

                '15r3': ({'alpha': [0.08], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['cyclic'], 'tol': [1e-1]}, 0.843),

                '15r4': ({'alpha': [0.09], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['random'], 'tol': [1e-1]}, 0.843),

                '14r5': ({'alpha': [0.08], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['cyclic'], 'tol': [1e-5]}, 0.842),

                '15r6': ({'alpha': [0.06], 'fit_intercept': [True], 'l1_ratio': [0.25], 'selection': ['random'], 'tol': [1e-1]}, 0.841)}    

result_cols = ['k', 'alpha', 'l1_ratio', 'CV', 'LB', 'params']

_ = plot_results(dict_elnet, result_cols, estimator_type=linear_model.ElasticNet(), figsize=(22, 8), rows=2, cols=3, y_scale=1500)  
plt.figure(figsize=(20, 6), facecolor='lightgrey')



plt.subplot(121, facecolor='slategrey')

_ = plt.plot(elnet_df.sort_values('alpha')[elnet_df['k'] == 15]['alpha'], elnet_df.sort_values('alpha')[elnet_df['k'] == 15]['CV'], c='salmon', ls='--')

_ = plt.plot(elnet_df.sort_values('alpha')[elnet_df['k'] == 15]['alpha'], elnet_df.sort_values('alpha')[elnet_df['k'] == 15]['LB'], c='navy')

plt.xlabel('alpha')

plt.ylabel('AUC score')

plt.xticks(np.arange(.05, .10, .01))

plt.yticks(np.arange(.840, .915, .005))

plt.grid(color='midnightblue', linestyle=':', lw=.7)

_ = plt.legend(('CV', 'LB'), loc='lower right')

_ = plt.title('[LB and CV] versus [alpha] for ElNet, k = 15')