import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from category_encoders import OrdinalEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import matplotlib.pyplot as plt

import seaborn as sns



from catboost import CatBoostClassifier



import os, gc, warnings, time



random_state = 42
def read_data(debug_mode = False):

    if debug_mode:

        nrows = 100000

    else:

        nrows = None

  

    data_path = '../input/'

    train_identity = pd.read_csv(os.path.join(data_path, 'train_identity.csv'))

    train_transaction = pd.read_csv(os.path.join(data_path, 'train_transaction.csv'), nrows = nrows)

    test_identity = pd.read_csv(os.path.join(data_path, 'test_identity.csv'),)

    test_transaction =pd.read_csv(os.path.join(data_path, 'test_transaction.csv'), nrows = nrows)

   

    train = pd.merge(train_transaction, train_identity, on= 'TransactionID', how = 'left')    

    test = pd.merge(test_transaction, test_identity, on= 'TransactionID', how = 'left')  

    del train_identity, train_transaction, test_identity, test_transaction

    gc.collect()

    

    

    print('Finished Reading Data')

    return train, test

    



def get_cat_num_cols(df):

    cat_cols = ['DeviceType', 'DeviceInfo', 'ProductCD', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain']

    cat_cols +=  ['M' + str(i) for i in range(1,10)]

    cat_cols += ['card' + str(i) for i in range(1,7)]

    cat_cols += ['id_' + str(i) for i in range(12,39)]

    

    all_cols = df.columns.tolist()

    num_cols = [x for x in all_cols if x not in cat_cols]

    num_cols.remove('TransactionID')

    num_cols.remove('isFraud')

    return cat_cols, num_cols



def set_ordinal_encoding(train, test, cat_cols):   

    oe = OrdinalEncoder( cols = cat_cols, handle_missing = 'return_nan')

    train[cat_cols] = oe.fit_transform(train[cat_cols])

    test[cat_cols] =   oe.transform(test[cat_cols])

    print('Finished: Ordinal Encoding')

    return train, test





def get_train_test(train, test):

#     X_train =  df[df['isFraud'].notnull()]

#     X_test  =  df[df['isFraud'].isnull()]

    y_train = train.isFraud

    sub = pd.DataFrame()

    sub['TransactionID'] = test['TransactionID']

    X_train = train.drop(['TransactionID', 'isFraud'], axis = 1)

    X_test =  test.drop(['TransactionID'], axis = 1)

    return X_train, X_test, y_train, sub



def plot_feature_imp(feature_imp, top_n = 30):

#     feature_imp = pd.DataFrame()

#     feature_imp['feature'] = model.feature_name()

#     feature_imp['importance']  = model.feature_importance()

    feature_imp = feature_imp.sort_values(['importance'], ascending = False)

    feature_imp_disp = feature_imp.head(top_n)

    plt.figure(figsize=(10, 12))

    sns.barplot(x="importance", y="feature", data=feature_imp_disp)

    plt.title('LightGBM Features')

    plt.show() 

    

    

def cv_results(y_valid, y_prob, verbose = True):   

    scores = {}                      

    y_pred_class =  [0  if x < 0.5 else 1 for x in y_prob]

    scores['cv_accuracy']  = accuracy_score(y_valid, y_pred_class)

    scores['cv_auc']       = roc_auc_score(y_valid, y_prob)

    scores['cv_f1']      =   f1_score(y_valid, y_pred_class, average = 'binary')

    if verbose:

        print('CV accuracy {:0.5f}'.format( scores['cv_accuracy'] ))

        print('CV AUC  {:0.5f}'.format( scores['cv_auc']   ))

        print('CV F1 %0.5f' %scores['cv_f1'] )

    return scores  



def run_lgb_with_cv(params, X_train, y_train, X_test, verbose_eval = 100):

    

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, 

                                                      random_state = random_state, stratify = y_train)

    print('Train shape{} Valid Shape{}, Test Shape {}'.format(X_train.shape, X_valid.shape, X_test.shape))



    lgb_train = lgb.Dataset(X_train, y_train)

    lgb_valid  = lgb.Dataset(X_valid, y_valid)

    early_stopping_rounds = 200

    lgb_results = {}

    

#     start_time = time.time()

    warnings.filterwarnings("ignore", message="categorical_feature in Dataset is overridden")

    model = lgb.train(params,

                      lgb_train,

                      num_boost_round = 10000,

                      valid_sets =  [lgb_train,lgb_valid],

                      early_stopping_rounds = early_stopping_rounds,                    

#                       categorical_feature = cat_cols,

                      evals_result = lgb_results,

                      verbose_eval = verbose_eval

                       )

    y_prob_valid = model.predict(X_valid)    

    cv_results(y_valid, y_prob_valid, verbose = True)

  

    feature_imp = pd.DataFrame()

    feature_imp['feature'] = model.feature_name()

    feature_imp['importance']  = model.feature_importance()

    return model, feature_imp





def run_lgb_simple(debug_mode = True):

    

    data= read_data(debug_mode = debug_mode)

    

    cat_cols, num_cols = get_cat_num_cols(data)

    

    data = set_ordinal_encoding(data, cat_cols) 

    

    X_train, X_test, y_train, sub = get_train_test(data)

    del data

    gc.collect()

    

    params = {}

    params['learning_rate'] = 0.1 #

    params['boosting_type'] = 'gbdt'

    params['objective'] = 'binary'

    params['seed'] =  random_state

    params['metric'] =    'auc'

    params['num_leaves'] =  60

    # params['bagging_fraction'] = 0.7

    # params['bagging_freq'] = 1

    # params['feature_fraction'] = 0.8

    # params['scale_pos_weight'] = 3

    # params['max_bin'] = 63



    model, feature_imp =  run_lgb_with_cv(params, X_train, y_train, X_test, cat_cols, verbose_eval = 100)

    

    plot_feature_imp(feature_imp, top_n = 50)

    

    y_prob_test = model.predict(X_test)

    sub['isFraud'] = y_prob_test

    sub.to_csv('lgb_sub.csv', index=False)

    

    
%%time

train, test = read_data(debug_mode = False)
# %%time

# #Frequency Encoding: Create a new column which counts combined occurance of each value in both train and test

# for col in cat_cols:

#     train[col + '_count'] = train[col].map(pd.concat([train[col], test[col]], ignore_index=True).value_counts(dropna=False))

#     test[col + '_count']  = test[col].map(pd.concat([train[col], test[col]], ignore_index=True).value_counts(dropna=False))
%%time

#For cat boost the categorical columns need not to be encoded to integers but NaN values must be imputed

cat_cols, num_cols = get_cat_num_cols(train)

mode = train.filter(cat_cols).mode()

train[cat_cols]= train[cat_cols].fillna(value=mode.iloc[0])

test[cat_cols]= test[cat_cols].fillna(value=mode.iloc[0])

train, test = set_ordinal_encoding(train, test, cat_cols)





# freq_cols = ['card1']

# for col in freq_cols:

#     cat_cols.remove(col)

#     train[col + '_count'] = train[col].map(pd.concat([train[col], test[col]], ignore_index=True).value_counts(dropna=False))

#     test[col + '_count']  = test[col].map(pd.concat([train[col], test[col]], ignore_index=True).value_counts(dropna=False))



# # Decimal part of Transaction Amount

# train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)

# test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)



X_train, X_test, y_train, sub = get_train_test(train, test)

del train, test

gc.collect()

%%time

X_tr, X_valid, y_tr, y_valid = train_test_split(X_train, y_train, test_size = 0.2, 

                                                      random_state = random_state, shuffle =False)



cat_cols_idx = [ X_train.columns.tolist().index(i) for i in  cat_cols ]



print('Train shape{} Valid Shape{}, Test Shape {}'.format(X_train.shape, X_valid.shape, X_test.shape))

%%time



model   =  CatBoostClassifier(  iterations= 20000,

                                loss_function='Logloss',

                                eval_metric='AUC',

                                grow_policy = 'Lossguide',

                                learning_rate = 0.05,

                                   max_leaves = 64,

                                task_type='GPU',

                                od_type = 'Iter',   #The type of the overfitting detector to use

                                od_wait = 300,      #Stop after n iterations

                                verbose = 200,

#                                 max_ctr_complexity = 1, #To improve Performance

#                                 border_count = 32, #To improve performance

                                random_seed = random_state  )



model.fit(

          X =  X_tr,

          y =  y_tr,

          cat_features=  cat_cols_idx ,    

          

          eval_set= (X_valid, y_valid),                       

          use_best_model = True ,

          logging_level = 'Verbose'

        )  



del X_tr, y_tr

gc.collect()


num_rounds = int(model.best_iteration_ + 0.15 * model.best_iteration_)

model   =  CatBoostClassifier(  iterations= num_rounds,

                                loss_function='Logloss',

                                 grow_policy = 'Lossguide',

                                 max_leaves = 64,

                                eval_metric='AUC',

                                learning_rate = 0.05,

                                task_type='GPU',

#                                 od_type = 'Iter',   #The type of the overfitting detector to use

#                                 od_wait = 300,      #Stop after n iterations

                                verbose = 200,

#                                 max_ctr_complexity = 1, #To improve Performance

#                                 border_count = 32, #To improve performance

                                random_seed = random_state  )



model.fit(

          X =  X_train,

          y =  y_train,

          cat_features=  cat_cols_idx ,    

          

#           eval_set= (X_valid, y_valid),                       

          use_best_model = False ,

          logging_level = 'Verbose'

        )            
y_prob_valid = model.predict(X_valid, prediction_type = 'Probability')[:,-1]    

cv_results(y_valid, y_prob_valid, verbose = True)

# plot_feature_imp(feature_imp, top_n = 50)

# feature_imp.to_csv('feature_imp.csv', index = False)



y_prob_test = model.predict(X_test, prediction_type = 'Probability')[:,-1]

sub['isFraud'] = y_prob_test

sub.to_csv('cb_sub.csv', index=False)