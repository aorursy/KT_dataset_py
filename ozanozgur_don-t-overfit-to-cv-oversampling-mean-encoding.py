import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import KBinsDiscretizer



import os

print(os.listdir("../input"))



train_data = pd.read_csv('../input/train.csv')

y_train = train_data.target

test_data = pd.read_csv('../input/test.csv')

test_IDs = test_data.ID_code



train_data.drop(['ID_code', 'target'], axis = 1, inplace = True)

test_data.drop(['ID_code'], axis = 1, inplace = True)



import warnings  

warnings.filterwarnings('ignore')
#Target encoding function

def mean_encode(train_data, test_data, columns, target_col, reg_method=None,

                alpha=0, add_random=False, rmean=0, rstd=0.1, folds=1):

    length_train = len(train_data)

    '''Returns a DataFrame with encoded columns'''

    encoded_cols = []

    target_mean_global = train_data[target_col].mean()

    for col in columns:

        # Getting means for test data

        nrows_cat = train_data.groupby(col)[target_col].count()

        target_means_cats = train_data.groupby(col)[target_col].mean()

        target_means_cats_adj = (target_means_cats*nrows_cat + 

                                 target_mean_global*alpha)/(nrows_cat+alpha)

        # Mapping means to test data

        encoded_col_test = test_data[col].map(target_means_cats_adj)

        # Getting a train encodings

        if reg_method == 'expanding_mean':

            train_data_shuffled = train_data.sample(frac=1, random_state=1)

            cumsum = train_data_shuffled.groupby(col)[target_col].cumsum() - train_data_shuffled[target_col]

            cumcnt = train_data_shuffled.groupby(col).cumcount()

            encoded_col_train = cumsum/(cumcnt)

            encoded_col_train.fillna(target_mean_global, inplace=True)

            if add_random:

                encoded_col_train = encoded_col_train + normal(loc=rmean, scale=rstd, 

                                                               size=(encoded_col_train.shape[0]))

        elif (reg_method == 'k_fold') and (folds > 1):

            kfold = StratifiedKFold(n_splits = folds, shuffle=True, random_state=1).split(train_data[target_col].values, train_data[target_col])

            parts = []

            for tr_in, val_ind in kfold:

                                # divide data

                    

                

                df_for_estimation, df_estimated = train_data.iloc[tr_in], train_data.iloc[val_ind]

                # getting means on data for estimation (all folds except estimated)

                nrows_cat = df_for_estimation.groupby(col)[target_col].count()

                target_means_cats = df_for_estimation.groupby(col)[target_col].mean()

                target_means_cats_adj = (target_means_cats*nrows_cat + 

                                         target_mean_global*alpha)/(nrows_cat+alpha)

                # Mapping means to estimated fold

                encoded_col_train_part = df_estimated[col].map(target_means_cats_adj)

                if add_random:

                    encoded_col_train_part = encoded_col_train_part + normal(loc=rmean, scale=rstd, 

                                                                             size=(encoded_col_train_part.shape[0]))

                # Saving estimated encodings for a fold

                parts.append(encoded_col_train_part)

            encoded_col_train = pd.concat(parts, axis=0)

            encoded_col_train.fillna(target_mean_global, inplace=True)

        else:

            encoded_col_train = train_data[col].map(target_means_cats_adj)

            if add_random:

                encoded_col_train = encoded_col_train + normal(loc=rmean, scale=rstd, 

                                                               size=(encoded_col_train.shape[0]))



        # Saving the column with means

        encoded_col = pd.concat([encoded_col_train, encoded_col_test], axis=0)

        encoded_col[encoded_col.isnull()] = target_mean_global

        encoded_cols.append(pd.DataFrame({'mean_'+target_col+'_'+col:encoded_col}))

    all_encoded = pd.concat(encoded_cols, axis=1)

    #Modified to reindex

    all_encoded = all_encoded.reset_index()

    return (all_encoded.iloc[:length_train].reset_index(drop = True), 

            all_encoded.iloc[length_train:].reset_index(drop = True)

           )
def RandomSearchModel(model, param_grid, num_runs = 1, random_state = None, n_folds = 2, process_fold = None, fold_data = None):

        #You must use random state 99999 to preprocess fold data.

        predictions_total = np.zeros(len(test_data))

        preds_train_total = np.zeros(len(train_data))

        

        #Separate model parameters from CV parameters

        cv_param_keys = ['has_eval_set', 'early_stopping', 'verbose']

        

        #Select random parameters from grid

        param = dict()

        for key in param_grid:

            param[key] = np.random.choice(param_grid[key])

        

        #Separate model parameters from CV parameters

        cv_params = dict()

        for cv_param in cv_param_keys:

            if not cv_param in list(param.keys()):

                print("CV parameter [{}] is required.".format(cv_param))

            cv_params[cv_param] = param[cv_param]

            param.pop(cv_param, None)



        print("parameters: {}".format(param))

        

        ### MULTIPLE RUNS ##############################################

        for i_run in range(num_runs):

            print('###> Run {}/{} <##################################'.format(i_run+1, num_runs))

            folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

                

            cv_scores = []

            best_score = 0

            preds_train = np.zeros(len(train_data))

            predictions = np.zeros(len(test_data))

            

            for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_data.values, y_train.values)):

                clf = model(**param)

                

                train_fold_x = None

                train_fold_y = None

                

                print("Fold {}".format(fold_))

                

                #Pass preprocessed fold data (oversampled or etc.)

                if fold_data == None:

                    train_fold_x = train_data.iloc[trn_idx]

                    train_fold_y = y_train.iloc[trn_idx].values.ravel()

                else:

                    #If there is a preprocessed data, use it

                    train_fold_x = fold_data[fold_][0]

                    train_fold_y = fold_data[fold_][1]

                    #May also add test,valid data

                

                valid_fold_x = train_data.iloc[val_idx]

                valid_fold_y = y_train.iloc[val_idx].values.ravel().copy()

                

                X_test_fold = None

                #Pass a function to process each fold data

                if process_fold != None:

                    train_fold_x, train_fold_y, valid_fold_x, X_test_fold = process_fold(train_fold_x, train_fold_y, valid_fold_x, test_data)

                    print(X_test_fold.shape)

                else:

                    X_test_fold = test_data

                    

                if cv_params['has_eval_set']:

                    clf.fit(train_fold_x, train_fold_y, eval_set = [(valid_fold_x, valid_fold_y)], early_stopping_rounds = cv_params['early_stopping'], verbose = cv_params['verbose'])

                else:

                    clf.fit(train_fold_x, train_fold_y)

                    print("Warning: Early stopping was not used! (Set [has_eval_set=1] if model allows.)")



                preds_train[val_idx] = clf.predict_proba(valid_fold_x)[:,1]

                

                score_fold = roc_auc_score(valid_fold_y, preds_train[val_idx])

                score_fold = np.abs(score_fold - 0.5) + 0.5



                cv_scores.append(score_fold)

                print('fold score: {}'.format(score_fold))



                # / folds.n_splits

                predictions += clf.predict_proba(X_test_fold)[:,1] / folds.n_splits

            

            predictions_total += predictions / num_runs

            preds_train_total += preds_train / num_runs

            

        overall_cv_score = roc_auc_score(y_train, preds_train_total)

        overall_cv_score = np.abs(overall_cv_score - 0.5) + 0.5

        

        return [overall_cv_score, param, cv_scores, predictions_total, preds_train_total]
#This function processes data at each fold, without using validation targets. Then we don't get false hopes by overfitting to CV.

def process_fold(train_fold_x, train_fold_y, valid_fold_x, X_test):

    #You can do whatever you want in this function and you won't overfit in CV results. => No bad surprise in submissions.

    #The reason for that is we don't have access to validation data target. Overfitting to CV results from making decisions based on

    #Validation set targets.

    

    #Things you can do here:

    # - Target encoding

    # - Binning

    # - Oversampling (preprocessing fold data is faster for that)

    # - Other FE

    

    #This didn't improve CV score, but you won't get false hopes by overfitting to CV.

    """cols_to_bin = np.random.choice(train_fold_x.columns, 10) #I chose these randomly, just to show that function works

    cols_to_encode = cols_to_bin

    

    print('Binning {} columns. (Replace)...'.format(len(cols_to_bin)))

    for col in cols_to_bin:

        est = KBinsDiscretizer(n_bins=25, encode='ordinal', strategy='quantile') #Can try different things

        est.fit(train_fold_x[col].values.reshape((-1,1)))

        # You may also fit to pd.concat([train_fold_x, valid_fold_x, X_test]) but I'm not sure which one works better



        train_fold_x[col] = est.transform(train_fold_x[col].values.reshape((-1,1)))

        valid_fold_x[col] = est.transform(valid_fold_x[col].values.reshape((-1,1)))

        X_test[col] = est.transform(X_test[col].values.reshape((-1,1)))

    

    #Cascaded mean encoding (This part is a little bit crappy, but I didn't have time to fix)

    #By encoding all columns this way, you can obtain a feature with 0.89 auc on its own without a model. But it didn't contribute to overall CV.

    

    print('Target encoding {} columns. (Add columns)...'.format(len(cols_to_encode)))

    num_valid = len(valid_fold_x)

    for col in cols_to_encode:

        train_fold_x['target'] = train_fold_y

        train_encoded, test_encoded = mean_encode(train_fold_x, pd.concat([valid_fold_x, X_test], axis = 0), [col], 'target', reg_method='k_fold',

                alpha=1, add_random=False, rmean=0, rstd=0.1, folds=4)

        train_fold_x.drop('target', axis = 1, inplace = True)

        

        train_encoded.drop('index', axis = 1, inplace = True)

        test_encoded.drop('index', axis = 1, inplace = True)

        

        train_fold_x.reset_index(drop = True, inplace = True)

        valid_fold_x.reset_index(drop = True, inplace = True)

        X_test.reset_index(drop = True, inplace = True)

        

        valid_encoded = test_encoded.iloc[:num_valid].reset_index(drop = True)

        test_encoded = test_encoded.iloc[num_valid:].reset_index(drop = True)

    

        train_fold_x = pd.concat([train_encoded, train_fold_x], axis = 1).reset_index(drop = True)

        valid_fold_x =  pd.concat([valid_encoded, valid_fold_x], axis = 1).reset_index(drop = True)

        X_test =  pd.concat([test_encoded, X_test], axis = 1).reset_index(drop = True)

    

    print('Fold processing done.')"""

    #Goes back into training

    return [train_fold_x, train_fold_y, valid_fold_x, X_test]
from imblearn.over_sampling import SMOTE



#This function processes fold data before starting training.

#This way you avoid the overhead of processing at each fold/run.

def preprocess_folds(random_state, n_folds, X_train, y_train):

    cols = X_train.columns

    # Oversample data fold by fold (More memory consumption, but faster)

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_data = []

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values, y_train.values)):

        print("Fold: {}".format(fold_))

        train_fold_x = X_train.iloc[trn_idx]

        train_fold_y = y_train.iloc[trn_idx].values.ravel()



        num_negative = len(train_fold_y[train_fold_y == 0])

        print("Oversampling...")

        train_fold_x, train_fold_y = SMOTE(sampling_strategy = {0: num_negative, 1:int(num_negative * 0.3)}).fit_resample(train_fold_x, train_fold_y)

        print('target=0 : {}'.format(len(train_fold_y[train_fold_y == 0])))

        print('target=1 : {}'.format(len(train_fold_y[train_fold_y == 1])))

        print("Oversampling done.")



        fold_data.append([pd.DataFrame(train_fold_x, columns = cols), pd.Series(train_fold_y)])

    

    return fold_data
#Oversample training sets in each fold. This way we won't overfit to CV.

n_folds = 5

random_state = 99999



fold_data = preprocess_folds(random_state = random_state,

                             n_folds = n_folds,

                             X_train = train_data,

                             y_train = y_train)
import lightgbm as lgb

#Random search gives its best results faster than bayesian search. Good for trying new things.

#But bayesian search performs better after many trials.



param_grid = dict(

         objective =  ['binary'],

         learning_rate = np.logspace(-3, -1, num=50, base=10.0),

         feature_fraction = np.logspace(-2, -1, num=50, base=10.0),

         num_leaves = np.arange(10,30,20),

         min_data_in_leaf = np.arange(30,150,50),

         bagging_fraction = [0.35, 0.32, 0.33, 0.37, 0.38, 0.39],

         bagging_freq = np.arange(3, 30, 27),

         max_depth = [-1],

         boosting_type = ['gbdt'],

         metric = ['auc'],

         min_sum_hessian_in_leaf = np.logspace(-4, 2, num=50, base=10.0),

         n_jobs = [-1],

         tree_learner = ['serial'],

         boost_from_average = [False],

         num_round = [30000],

         verbose_eval = [1000],

    

        #CV parameters

        verbose = [1],

        has_eval_set = [True],

        early_stopping = [3000]

)



cv_results = []



for i in range(1): #Increase this as much as you want

    #You can train model with different folds num_runs times and average predictions and results (CV score will be closer to LB score)

    #Note that you must set random_state to None while doing this.

    

    #In each call to RandomSearchModel, random parameters from parameter grid is selected.

    #You can provide preprocessed fold data [(fold1_train, fold1_y_train), (fold2_train, fold2_y_train) ...] Use this for oversampling.

    [cv_score, param, fold_scores, predictions_total, preds_train_total] = RandomSearchModel(lgb.LGBMClassifier,

                                                                         param_grid,

                                                                         num_runs = 1,

                                                                         random_state = random_state,

                                                                         n_folds = n_folds,

                                                                         process_fold = process_fold,

                                                                         fold_data = fold_data

                                                                        )

    cv_results.append((cv_score, param, predictions_total))



cv_results_df = pd.DataFrame(cv_results, columns = ['cv_score', 'parameters', 'predictions_total'])

cv_results_df.sort_values(by = 'cv_score', inplace = True)

display(cv_results_df)
"""my_submission = pd.DataFrame({'ID_code': test_IDs, 'target': cv_results_df.iloc[0].predictions_total})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)"""