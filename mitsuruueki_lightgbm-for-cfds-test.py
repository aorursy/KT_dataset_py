import os

import numpy as np

import pandas as pd

from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import lightgbm as lgb

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display

%matplotlib inline 

plt.xkcd()



PATH = '/kaggle/input'

print(os.listdir(PATH))



# Random seed

SEED = 1234



TASK = 'regression' # 'regression'

categorical_encoding = 'ordinal' # 'one-hot'
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df





def import_data(file):

    """create a dataframe and optimize its memory usage"""

    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)

    df = reduce_mem_usage(df)

    return df
def count_categories(df):

    for col in df.columns:

        col_type = str(df[col].dtype)

        if col_type == 'category':

            print(col, len(df[col].unique()))
def extract_category_features(train, test, unique_num):

    train_extracted = train.copy()

    test_extracted = test.copy()

    

    for col in train.columns:

        col_type = str(train[col].dtype)

        if col == 'Country':

            continue

        elif col_type == 'category' and len(train[col].unique()) > unique_num:

            train_extracted.drop(col, axis=1, inplace=True)

            test_extracted.drop(col, axis=1, inplace=True)

    

    return train_extracted, test_extracted
train = import_data(os.path.join(PATH,'train.csv'))

test  = import_data(os.path.join(PATH,'test.csv'))

country_info = import_data(os.path.join(PATH,'country_info.csv'))
display(train.head())

display(test.head())

display(country_info.head())
train.describe()
test.describe()
train.info()
test.info()
count_categories(train)
count_categories(test)
unique_num = 20

train, test = extract_category_features(train, test, unique_num)
train.info()
test.info()
def impute_missing_values(train, test):

    for col in train.columns:

        if train[col].isnull().sum() > 0:

            if str(train[col].dtype) != 'category':

                train[col] = train[col].fillna(train[col].mean())

            else:

                train[col] = train[col].fillna(train[col].mode())

                

    for col in test.columns:                

        if test[col].isnull().sum() > 0:

            if str(test[col].dtype) != 'category':

                test[col] = test[col].fillna(test[col].mean())

            else:

                test[col] = test[col].fillna(test[col].mode())

                

    return train, test
train, test = impute_missing_values(train, test)
def calc_tfidf(train, test, feature_name, top_n):

    top_n *= -1

    

    tfidf_vectorizer = TfidfVectorizer(min_df=5, norm=None, stop_words='english')

    tfidf_vectorizer.fit(train[feature_name])

    train_tfidf_df = tfidf_vectorizer.transform(train[feature_name])

    max_value = train_tfidf_df.max(axis=0).toarray().ravel()

    sorted_by_tfidf = max_value.argsort()



    words_list = np.array(tfidf_vectorizer.get_feature_names())

    

    train_tfidf_df = pd.DataFrame(train_tfidf_df.toarray(), columns=words_list)

    train_tfidf_df = train_tfidf_df[words_list[sorted_by_tfidf[top_n:]]]

    train_tfidf_df.rename(columns=lambda s:feature_name+'_'+s, inplace=True)

    train_new = pd.concat([train, train_tfidf_df], axis=1)

    train_new.drop(feature_name, axis=1, inplace=True)

    

    

    test_tfidf_array = tfidf_vectorizer.transform(test[feature_name]).toarray()

    test_tfidf_df = pd.DataFrame(test_tfidf_array, columns=words_list)

    test_tfidf_df = test_tfidf_df[words_list[sorted_by_tfidf[top_n:]]]

    test_tfidf_df.rename(columns=lambda s:feature_name+'_'+s, inplace=True)

    test_new = pd.concat([test, test_tfidf_df], axis=1)

    test_new.drop(feature_name, axis=1, inplace=True)

    

    return train_new, test_new
#train, test = calc_tfidf(train, test, feature_name='Name', top_n=10)
# replace `Country` to `GDP`

train = pd.merge(train, country_info[['Country', 'GDP ($ per capita)']], on='Country', how='left')

test = pd.merge(test, country_info[['Country', 'GDP ($ per capita)']], on='Country', how='left')



train.drop('Country', axis=1, inplace=True)

test.drop('Country', axis=1, inplace=True)
def cat_to_dummy(train, test, target_feature):

    train_d = pd.get_dummies(train, drop_first=False)

    test_d = pd.get_dummies(test, drop_first=False)

    for i in train_d.columns:

        if i not in test_d.columns:

            if i != target_feature:

                train_d = train_d.drop(i, axis=1)

    for j in test_d.columns:

        if j not in train_d.columns:

            if j != target_feature:

                test_d = test_d.drop(j, axis=1)

    print('Memory usage of train increases from {:.2f} to {:.2f} MB'.format(train.memory_usage().sum() / 1024**2, 

                                                                            train_d.memory_usage().sum() / 1024**2))

    print('Memory usage of test increases from {:.2f} to {:.2f} MB'.format(test.memory_usage().sum() / 1024**2, 

                                                                            test_d.memory_usage().sum() / 1024**2))

    return train_d, test_d
def cat_to_int(train, test):

    mem_orig_train = train.memory_usage().sum() / 1024**2

    mem_orig_test  = test .memory_usage().sum() / 1024**2

    categorical_feats = [ f for f in train.columns if train[f].dtype == 'object' or train[f].dtype.name == 'category' ]

    print('---------------------')

    print(categorical_feats)

    for f_ in categorical_feats:

        train[f_], indexer = pd.factorize(train[f_])

        test[f_] = indexer.get_indexer(test[f_])

    print('Memory usage of train increases from {:.2f} to {:.2f} MB'.format(mem_orig_train, 

                                                                            train.memory_usage().sum() / 1024**2))

    print('Memory usage of test increases from {:.2f} to {:.2f} MB'.format(mem_orig_test, 

                                                                            test.memory_usage().sum() / 1024**2))

    return categorical_feats, train, test

if categorical_encoding == 'one-hot':

    target_feature = 'ConvertedSalary'

    train_ce, test_ce = cat_to_dummy(train, test, target_feature)

elif categorical_encoding == 'ordinal':

    categorical_feats, train_ce, test_ce = cat_to_int(train, test)
display(train_ce.head())

display(test_ce.head())
X_rus, y_rus = (train_ce.drop(['Respondent', 'ConvertedSalary'], axis=1),

                train_ce['ConvertedSalary'])
if TASK == 'classification':

    X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus, test_size=0.20, random_state=SEED, stratify=y_rus)

elif TASK == 'regression':

    X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus, test_size=0.20, random_state=SEED)

else:

    print('ERROR')
def learning_rate_010_decay_power_099(current_iter):

    base_learning_rate = 0.1

    lr = base_learning_rate  * np.power(.99, current_iter)

    return lr if lr > 1e-3 else 1e-3



def learning_rate_010_decay_power_0995(current_iter):

    base_learning_rate = 0.1

    lr = base_learning_rate  * np.power(.995, current_iter)

    return lr if lr > 1e-3 else 1e-3



def learning_rate_005_decay_power_099(current_iter):

    base_learning_rate = 0.05

    lr = base_learning_rate  * np.power(.99, current_iter)

    return lr if lr > 1e-3 else 1e-3
if TASK == 'classification':

    fit_params={"early_stopping_rounds":100, 

                "eval_metric" : 'auc', 

                "eval_set" : [(X_test,y_test)],

                'eval_names': ['valid'],

                #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],

                'verbose': 100,

                'categorical_feature': 'auto'}



elif TASK == 'regression':

    fit_params={"early_stopping_rounds":100, 

                "eval_metric" : 'l2', 

                "eval_set" : [(X_test,y_test)],

                'eval_names': ['valid'],

                #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],

                'verbose': 100,

                'categorical_feature': 'auto'}
param_test ={'num_leaves': sp_randint(6, 50), 

             'min_child_samples': sp_randint(100, 500), 

             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],

             'subsample': sp_uniform(loc=0.2, scale=0.8), 

             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),

             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],

             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}
#This parameter defines the number of HP points to be tested

n_HP_points_to_test = 10



if TASK == 'classification':

    clf = lgb.LGBMClassifier(max_depth=-1, random_state=SEED, silent=True, 

                             metric='None', n_jobs=-1, n_estimators=5000, importance_type='gain')

    gs = RandomizedSearchCV(estimator=clf, param_distributions=param_test, 

                            n_iter=n_HP_points_to_test, scoring='roc_auc',

                            cv=5, refit=True, random_state=SEED, verbose=True, n_jobs=-1)

elif TASK == 'regression':

    clf = lgb.LGBMRegressor(max_depth=-1, random_state=SEED, silent=True, 

                            metric='rmse', n_jobs=-1, n_estimators=5000, importance_type='gain')

    gs = RandomizedSearchCV(estimator=clf, param_distributions=param_test, 

                            n_iter=n_HP_points_to_test, scoring='neg_mean_squared_error',

                            cv=5, refit=True, random_state=SEED, verbose=True, n_jobs=-1)
hyp_param_optim = False



if hyp_param_optim == True:

    gs.fit(X_train, y_train, **fit_params)

    print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))
opt_parameters = {'colsample_bytree': 0.7790752106122303, 'min_child_samples': 102, 'min_child_weight': 1, 

                  'num_leaves': 45, 'reg_alpha': 1, 'reg_lambda': 10, 'subsample': 0.5422907926702556}
if TASK == 'classification':

    clf_sw = lgb.LGBMClassifier(**clf.get_params())

elif TASK == 'regression':

    clf_sw = lgb.LGBMRegressor(**clf.get_params())



#set optimal parameters

clf_sw.set_params(**opt_parameters)



#set importance_type

clf_sw.set_params(importance_type='gain')
if TASK == 'classification':

    clf_final = lgb.LGBMClassifier(**clf.get_params())

elif TASK == 'regression':

    clf_final = lgb.LGBMRegressor(**clf.get_params())



#set optimal parameters

clf_final.set_params(**opt_parameters)



#set importance_type

clf_final.set_params(importance_type='gain')



#Train the final model with learning rate decay

clf_final.fit(X_train, y_train, 

              **fit_params, callbacks=[lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)])
feat_imp = pd.Series(clf_final.feature_importances_, index=train_ce.drop(['Respondent', 'ConvertedSalary'], axis=1).columns)

(feat_imp.nlargest(20)/feat_imp.max()*100).plot(kind='barh', figsize=(8,10))

plt.show()
top_feat_list = list(feat_imp[(feat_imp/feat_imp.max()*100)>20].index)
if TASK == 'classification':

    clf_final_ext_feat = lgb.LGBMClassifier(**clf.get_params())

elif TASK == 'regression':

    clf_final_ext_feat = lgb.LGBMRegressor(**clf.get_params())



#set optimal parameters

clf_final.set_params(**opt_parameters)

    

#set importance_type

clf_final_ext_feat.set_params(importance_type='gain')



#Train the final model with learning rate decay

clf_final_ext_feat.fit(X_train[top_feat_list], y_train, 

                       callbacks=[lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)])
import math



# non-negative

def rmsle(y, y_pred):

    assert len(y) == len(y_pred)

    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]

    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
train_pred_val = clf_final_ext_feat.predict(train_ce.drop(['Respondent', 'ConvertedSalary'], axis=1)[top_feat_list])

train_pred_df = pd.DataFrame({

                            'Respondent': train_ce['Respondent'],

                            'ConvertedSalary':    [row for row in train_pred_val]

                           })

train_pred_df['ConvertedSalary'].hist()

plt.show()
print(train_pred_val.max(), train_pred_val.min()) 
pred_val = clf_final_ext_feat.predict(test_ce.drop(['Respondent'], axis=1)[top_feat_list])

submission = pd.DataFrame({

                            'Respondent': test_ce['Respondent'],

                            'ConvertedSalary':    [row for row in pred_val]

                           })

submission.to_csv("submission.csv", index=False)
submission['ConvertedSalary'].hist()

plt.show()