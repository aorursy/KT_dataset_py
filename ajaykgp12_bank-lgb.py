import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, gc, warnings

import seaborn as sns

import matplotlib.pyplot as plt

from plotly.subplots import make_subplots

import plotly.graph_objects as go

from imblearn.over_sampling import RandomOverSampler, SMOTE 

from sklearn.model_selection import  train_test_split

import lightgbm as lgb

from sklearn.metrics import accuracy_score, roc_auc_score,f1_score

from sklearn.model_selection import KFold, StratifiedKFold



pd.set_option('display.max_columns', 50)

pd.set_option('display.max_rows', 100)

random_state = 42



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



warnings.filterwarnings("ignore", message="categorical_feature in Dataset is overridden")

warnings.filterwarnings("ignore", message="F-score is ill-defined and being set to 0.0 due to no predicted samples")

warnings.simplefilter(action='ignore', category=FutureWarning)

    
def get_class_from_prob(y_prob):

        return  [0  if x < THR else 1 for x in y_prob]



def write_predictions(sub, y_prob, filename ):          

    sub_prob = sub.copy()

    sub_prob[TARGET] = y_prob    

    sub_prob.to_csv('test_prob.csv', index = False)

    

    y_pred =   get_class_from_prob(y_prob) 

    print('Test prediction:Postive class count {}, Percent {:0.2f}'.format(sum(y_pred), sum(y_pred) * 100 / len(y_pred)))

    sub[TARGET] = y_pred

    sub.to_csv(filename, index = False)

    print('Predictions Written to file', filename)

    



def get_train_test(df, features):

    X_train =  df[df[TARGET].notnull()]

    X_test  =  df[df[TARGET].isnull()]

    y_train = X_train[TARGET]

    sub = pd.DataFrame()

    sub['loan_id'] = X_test['loan_id']

    X_train = X_train[features]

    X_test = X_test[features]

    return X_train, X_test, y_train, sub



def set_ordinal_encoding(data, cat_cols):       

    for col in [x for x in cat_cols if data[x].dtype == 'object']:

        data[col], uniques = pd.factorize(data[col])

        #the factorize sets null values to -1, so convert them back to null, as we want LGB to handle null values

        data[col] = data[col].replace(-1, np.nan)

    print('Finished: Ordinal Encoding')

    return data



def run_lgb_with_cv(params, X_train, y_train, X_test,cat_cols,  test_size =0.2, verbose_eval = 100, ):    

    

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = test_size , 

                                                              random_state = random_state, stratify = y_train)   

  

    print('Train shape{} Valid Shape{}, Test Shape {}'.format(X_train.shape, X_valid.shape, X_test.shape))

    print('Number of Category Columns {}:'.format(len(cat_cols)))



    lgb_train = lgb.Dataset(X_train, y_train)

    lgb_valid  = lgb.Dataset(X_valid, y_valid)

    early_stopping_rounds = 200

    lgb_results = {}    



    

    model = lgb.train(params,

                      lgb_train,

                      num_boost_round = 10000,

                      valid_sets =  [lgb_train,lgb_valid],  #Including train set will do early stopiing for train instead of validation

                      early_stopping_rounds = early_stopping_rounds,                      

                      categorical_feature = cat_cols,

                      evals_result = lgb_results,

                      feval = lgb_f1_score,

                      verbose_eval = verbose_eval

                       )

    y_prob_valid = model.predict(X_valid) 

    y_pred =   get_class_from_prob(y_prob_valid)   

    print('Validation prediction:Postive class count {}, Percent {:0.2f}'.format(sum(y_pred), sum(y_pred) * 100 / len(y_pred)))

    cv_results(y_valid, y_prob_valid, verbose = True)

  

    feature_imp = pd.DataFrame()

    feature_imp['feature'] = model.feature_name()

    feature_imp['importance']  = model.feature_importance()

    feature_imp = feature_imp.sort_values(by = 'importance', ascending= False )

    return model, feature_imp



def cv_results(y_valid, y_prob, verbose = True):   

    scores = {}                      

    y_pred_class =  get_class_from_prob(y_prob)  

    scores['cv_accuracy']  = accuracy_score(y_valid, y_pred_class)

    scores['cv_auc']       = roc_auc_score(y_valid, y_prob)

    scores['cv_f1']      =   f1_score(y_valid, y_pred_class, average = 'binary')

    if verbose:

        print('CV accuracy {:0.6f}'.format( scores['cv_accuracy'] ))

        print('CV AUC  {:0.6f}'.format( scores['cv_auc']   ))

        print('CV F1 %0.6f' %scores['cv_f1'] )

    return scores



def plot_feature_imp(feature_imp, top_n = 30):

    feature_imp = feature_imp.sort_values(['importance'], ascending = False)

    feature_imp_disp = feature_imp.head(top_n)

    plt.figure(figsize=(10, 12))

    sns.barplot(x="importance", y="feature", data=feature_imp_disp)

    plt.title('LightGBM Features')

    plt.show() 

    

def lgb_f1_score(y_hat, data):

    y_true = data.get_label()

    y_pred =   get_class_from_prob(y_hat)

#     y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities

    return 'f1', f1_score(y_true, y_pred), True





def lgb_nfolds(params, X_train, y_train, X_test,  cat_cols = [], oversample = False,

               stratifiedKFold = False, nfolds = 5, verbose_eval = 100, predict_test_set = True ):

    if stratifiedKFold:

       folds = StratifiedKFold(n_splits = nfolds, shuffle = True, random_state = random_state)   

    else:

       folds = KFold(n_splits = nfolds, shuffle = True, random_state = random_state)



    oof_prob = np.zeros(shape=(X_train.shape[0])) 

    cv_score = np.zeros(shape = nfolds)

    test_prob = np.zeros(shape=(X_test.shape[0]))

    feature_imp = pd.DataFrame()   



    print('Train Shape {}, Test shape {}'.format(X_train.shape, X_test.shape )) 

    print('Number of Category Columns {}:'.format(len(cat_cols)))

    warnings.filterwarnings("ignore", message="categorical_feature in Dataset is overridden")

    warnings.filterwarnings("ignore", message="categorical_feature in param dict is overridden")



    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):

        train_x, train_y = X_train.iloc[train_idx],  y_train.iloc[train_idx]

        valid_x, valid_y = X_train.iloc[valid_idx],  y_train.iloc[valid_idx]   

        

        if oversample:           

#             ros = RandomOverSampler(random_state=42)

            sm = SMOTE(random_state= random_state)

            train_x, train_y = sm.fit_resample(train_x, train_y)

            train_x = pd.DataFrame(train_x, columns = X_train.columns.tolist())

        

        lgb_train = lgb.Dataset(train_x, train_y)

        lgb_valid  = lgb.Dataset(valid_x, valid_y)

        early_stopping_rounds = 250

        lgb_results = {}

       

        print('Train Shape {}, Valid shape {}'.format(train_x.shape, valid_x.shape )) 

        model = lgb.train(params,

                          lgb_train,

                          num_boost_round = 10000,

                          valid_sets =  [lgb_valid],

                          early_stopping_rounds = early_stopping_rounds,                    

                          categorical_feature = cat_cols,                    

                          evals_result = lgb_results,

                          feval = lgb_f1_score,

                          verbose_eval = verbose_eval

                           )



        oof_prob[valid_idx] = model.predict(valid_x) 

        y_pred =   get_class_from_prob(oof_prob[valid_idx])

        print('Valid Set:Postive class count {}, Percent {:0.2f}'.format(sum(valid_y), sum(valid_y) * 100 / len(valid_y)))

        print('Valid Predict :Postive class count {}, Percent {:0.2f}'.format(sum(y_pred), sum(y_pred) * 100 / len(y_pred)))

        if predict_test_set:                            

            #Average out the test set probablities  

            test_prob += model.predict(X_test) / folds.n_splits        

    

        fold_importance = pd.DataFrame()

        fold_importance["feature"] =  model.feature_name()

        fold_importance["importance"] = model.feature_importance()

        fold_importance["fold"] = n_fold + 1   

        feature_imp = pd.concat([feature_imp, fold_importance], axis=0)



        print('\nFold {} results'.format(n_fold+1))

        scores = cv_results(valid_y, oof_prob[valid_idx], verbose = False)

        cv_score[n_fold] = scores['cv_f1']

        print('Validation F1 {:0.6f}'.format(cv_score[n_fold]))  

           

    print('\nScores:', cv_score)

    print('Mean F1 Score{:0.6f}: '.format(np.mean(cv_score))) 

    print('Out of Fold  CV results') 

    cv_results(y_train, oof_prob, verbose = True)

    

    y_pred =   get_class_from_prob(oof_prob)   

    print('OOF prediction:Postive class count {}, Percent {:0.2f}'.format(sum(y_pred), sum(y_pred) * 100 / len(y_pred)))



    feature_imp = feature_imp[['feature', 'importance']].groupby(['feature']).mean().reset_index()

    feature_imp.sort_values(['importance'], ascending= False, inplace = True)

    return oof_prob, test_prob, feature_imp   

    
TARGET = 'm13'

data_path = '/kaggle/input/bank-fraud/data/'

# output_path = 'C:\\Users\\I056036\\Documents\\Docs\\Development\\datasets\\AV-Loan Default\\outputs'

train = pd.read_csv(os.path.join(data_path, 'train.csv'))

test =pd.read_csv(os.path.join(data_path, 'test.csv'))



# test['first_payment_date'] = test['first_payment_date'].map({'Apr-12': '04/2012', 'Mar-12': '05/2012', 'Feb-12':'02/2012'})

test['first_payment_date'] = test['first_payment_date'].map({'Apr-12': '04/2012', 'Mar-12': '03/2012', 

                                                                     'May-12':'05/2012', 'Feb-12':'02/2012'})



data = train.append(test, ignore_index = True, sort=False)
def calc_emi(X):

    P = X['unpaid_principal_bal']

    N = X['loan_term']

    R = X['interest_rate'] / 1200

    emi = (P * R * (1 + R) ** N ) / ((1 + R) ** N -1 )

    return emi 



def combine_cols(col1, col2, df):

    col = col1 + '_' + col2

    df[col] = df[col1] + '_'  + df[col2]

    return df, col

data['origination_month'] = pd.to_datetime(data['origination_date']).dt.month

cat_cols = ['source', 'financial_institution', 'loan_purpose', 'insurance_type', 'first_payment_date']
%%time





mcols = ['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12']



freq_cols = data.columns.tolist()

freq_cols.remove('loan_id')

freq_cols.remove(TARGET)

freq_cols.remove('origination_date')

for col in freq_cols:

    data[col + '_count'] = data[col].map(data[col].value_counts(dropna=False)) 

   



data['bal_term_ratio'] = data['unpaid_principal_bal'] / data['loan_term'] 

data['bal_borrowers_ratio'] = data['unpaid_principal_bal'] / data['number_of_borrowers']



data, col = combine_cols('source', 'financial_institution', data)

cat_cols.append(col)

data, col = combine_cols('source', 'loan_purpose', data)

cat_cols.append(col)

data, col = combine_cols('financial_institution', 'loan_purpose', data)

cat_cols.append(col)



data['emi'] = data.apply(calc_emi, axis = 1)

data['mtotal']= data.loc[:,mcols].sum(axis=1)

data['mavg']= data['mtotal'] / len(mcols)

data['mbool'] = data.apply(lambda x: 1 if x['mtotal'] > 0 else 0, axis = 1)

data['avg_credit_score'] = (data['borrower_credit_score'] + data['co-borrower_credit_score']) / 2



df = data.copy()

df['m12bool'] = df.apply(lambda x: 1 if x['m12'] > 0 else 0, axis = 1)

df['m11bool'] = df.apply(lambda x: 1 if x['m11'] > 0 else 0, axis = 1)

df['m10bool'] = df.apply(lambda x: 1 if x['m10'] > 0 else 0, axis = 1)

df['m9bool'] =  df.apply(lambda x: 1 if x['m9'] > 0 else 0, axis = 1)



data['msum'] = df['m12bool'] + df['m11bool'] + df['m10bool'] + df['m9bool']



features = data.columns.tolist()

features.remove('loan_id')

features.remove(TARGET)

features.remove('origination_date')



# features = ['source', 'financial_institution', 'interest_rate', 'unpaid_principal_bal', 'loan_term', 'first_payment_date', 

#             'loan_to_value', 'number_of_borrowers', 'debt_to_income_ratio', 'borrower_credit_score', 'loan_purpose', 'insurance_percent',

#             'co-borrower_credit_score', 'insurance_type', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 

#             'origination_month', 'unpaid_principal_bal_count', 'first_payment_date_count', 'loan_to_value_count', 'm10_count', 

#             'm12_count', 'msum']



# cat_cols = [x for x in cat_cols if x in features]



data = set_ordinal_encoding(data, cat_cols)

X_train, X_test, y_train, sub = get_train_test(data, features)

X_train.shape


# %%time

# THR = 0.5



# params = {}

# params['learning_rate'] = 0.02

# params['boosting_type'] = 'gbdt'

# params['objective'] = 'binary'

# params['seed'] =  random_state

# params['metric'] = 'None'

# params['num_leaves'] =  64



# cat_cols_idx = [ X_train.columns.tolist().index(i) for i in  cat_cols ]

# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2 , 

#                                                               random_state = random_state, stratify = y_train)   



# ros = RandomOverSampler(random_state=42)

# X_train, y_train = ros.fit_resample(X_train, y_train)

# X_train = pd.DataFrame(X_train, columns = features)

  

# print('Train shape{} Valid Shape{}, Test Shape {}'.format(X_train.shape, X_valid.shape, X_test.shape))

# print('Number of Category Columns {}:'.format(len(cat_cols)))



# lgb_train = lgb.Dataset(X_train, y_train)

# lgb_valid  = lgb.Dataset(X_valid, y_valid)

# early_stopping_rounds = 200

# lgb_results = {}    

# model = lgb.train(params,

#                   lgb_train,

#                   num_boost_round = 10000,

#                   valid_sets =  [lgb_valid],  #Including train set will do early stopiing for train instead of validation

#                   early_stopping_rounds = early_stopping_rounds,                      

#                   categorical_feature = cat_cols_idx,

#                   evals_result = lgb_results,

#                   feval = lgb_f1_score,

#                   verbose_eval = 100

#                    )



# y_prob_valid = model.predict(X_valid) 

# y_pred =   get_class_from_prob(y_prob_valid)   

# print('Validation prediction:Postive class count {}, Percent {:0.2f}'.format(sum(y_pred), sum(y_pred) * 100 / len(y_pred)))

# cv_results(y_valid, y_prob_valid, verbose = True)



# feature_imp = pd.DataFrame()

# feature_imp['feature'] = model.feature_name()

# feature_imp['importance']  = model.feature_importance()

# feature_imp = feature_imp.sort_values(by = 'importance', ascending= False )
# y_prob_test= model.predict(X_test)

# write_predictions(sub,y_prob_test, 'lgb_sub.csv' )

# plot_feature_imp(feature_imp, top_n = 30)

# print(params)
%%time



THR = 0.5

stratifiedKFold  = False

oversample = True



params = {}

params['bagging_fraction'] = 0.65

params['bagging_freq'] = 11

params['feature_fraction'] = 0.65

params['lambda_l1'] = 0.2

params['lambda_l2'] = 2.15

params['max_bin'] = 63

params['min_data_in_leaf'] = 44

params['min_gain_to_split'] = 6.25

params['min_sum_hessian_in_leaf'] = 7.0

# params['scale_pos_weight'] = 13



params['learning_rate'] =  0.061088469

params['num_leaves'] =  317

params['boosting_type'] = 'gbdt'

params['objective'] = 'binary'

params['seed'] =  random_state

params['metric'] = 'None'



oof_prob, y_prob_test, feature_imp  = lgb_nfolds(params, X_train, y_train, X_test,  cat_cols , oversample, 

                                                 stratifiedKFold , nfolds = 5, verbose_eval = 100, predict_test_set = True )



print(params)

write_predictions(sub,y_prob_test, 'lgb_sub.csv' )

plot_feature_imp(feature_imp, top_n = 30)