# For data analysis, model building, evaluating

from sklearn.model_selection import train_test_split, StratifiedKFold,KFold

from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve

from sklearn import preprocessing

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', 200) # before I forget



import warnings

warnings.filterwarnings("ignore")



#For loading data

import os



# For plots

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib import rcParams
print(os.listdir("../input"))



input_path = '../input'



%matplotlib inline



RANDOM_SEED = 42

nan_replace = -999
print(os.listdir('../input/previous-submissions'))
baseline_random_forest = pd.read_csv('../input/previous-submissions/baseline_random_forest.csv', index_col='TransactionID') # .878

baseline_xgboost = pd.read_csv('../input/previous-submissions/baseline_xgboost.csv', index_col='TransactionID') # .938

xgboost_with_tuning = pd.read_csv('../input/previous-submissions/xgboost_with_tuning.csv', index_col='TransactionID') # .928

preprocessed2_xgboost = pd.read_csv('../input/previous-submissions/preprocessed2_xgboost.csv', index_col='TransactionID') # .934
assert(baseline_random_forest.shape == baseline_xgboost.shape == xgboost_with_tuning.shape == preprocessed2_xgboost.shape)
ensemble = .025*baseline_random_forest + .05*xgboost_with_tuning + .075*preprocessed2_xgboost + .85*baseline_xgboost

ensemble.head()
ensemble.to_csv('ensemble6.csv')
# %%time

# train_identity =  pd.read_csv(os.path.join(input_path, 'train_identity.csv'), index_col='TransactionID')

# train_transaction = pd.read_csv(os.path.join(input_path, 'train_transaction.csv'), index_col='TransactionID')

# test_identity = pd.read_csv(os.path.join(input_path, 'test_identity.csv'), index_col='TransactionID')

# test_transaction = pd.read_csv(os.path.join(input_path, 'test_transaction.csv'), index_col='TransactionID')
# print('train_identity shape:', train_identity.shape)

# print('train_transaction shape:', train_transaction.shape)

# print('test_identity shape:', test_identity.shape)

# print('test_transaction shape:', test_transaction.shape)
# train_transaction.head()
# train_identity.head()
# train = pd.merge(train_transaction, train_identity, how='left', on='TransactionID')

# test = pd.merge(test_transaction, test_identity, how='left', on='TransactionID')



# # see if transaction and identity variables one train table (should be same for test)

# train.head()
# # clear up RAM

# del train_transaction, train_identity, test_transaction, test_identity
# print('train shape:', train.shape)

# print('test shape:', test.shape)
# num_train = train.shape[0]

# num_test = test.shape[0]

# num_features = test.shape[1]



# print('Test data is {:.2%}'.format(num_test/(num_train+num_test)), 'of total train/test data')
# import xgboost as xgb

# print('XGBoost version:', xgb.__version__)
# y_train = train['isFraud']

# X_train = train.drop('isFraud', axis=1)

# X_test = test





# X_train = X_train.fillna(nan_replace)

# X_test = X_test.fillna(nan_replace)



# del train, test



# # Label Encoding

# for f in X_train.columns:

#     if X_train[f].dtype=='object' or X_test[f].dtype=='object': 

#         lbl = preprocessing.LabelEncoder()

#         lbl.fit(list(X_train[f].values) + list(X_test[f].values))

#         X_train[f] = lbl.transform(list(X_train[f].values))

#         X_test[f] = lbl.transform(list(X_test[f].values))
# clf = xgb.XGBClassifier(n_estimators=500,

#     max_depth=9,

#     learning_rate=0.05,

#     subsample=0.9,

#     colsample_bytree=0.9,

#     missing=nan_replace,

#     random_state=RANDOM_SEED,

#     tree_method='gpu_hist')
# %%time

# clf.fit(X_train, y_train)
# sample_submission = pd.read_csv(os.path.join(input_path, 'sample_submission.csv'), index_col='TransactionID')
# del X_train, y_train
# y_pred = clf.predict_proba(X_test)



# sample_submission['isFraud'] = y_pred[:,1]

# sample_submission.to_csv('baseline_xgboost.csv')
# for feature in train.columns[:20]:

#     print(feature, '\t', train[feature].dtype)
# cat_features = ['ProductCD', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo', 'isFraud']



# # add card1-card6

# for i in range(1, 7):

#     cat_features.append('card'+str(i))

    

    

# # add M1-M9

# for i in range(1, 10):

#     cat_features.append('M'+str(i))

    

    

# # add id12-38

# for i in range(12, 39):

#     cat_features.append('id_'+str(i))
# # Convert categorical features to data type 'object'

# def convert_to_object(df, cat_features):

#     """

#     Converts features to data type 'object', so that all categorical features in dataframe are of type 'object'

    

#     Args:

#         df (pd.Dataframe)

#         cat_features (list): the categorical features as strings

        

#     Returns:

#         df (pd.Dataframe): where new df has categorical features as type 'object'

#     """

#     for feature in cat_features:

#         if feature not in df.columns:

#             print('ERROR:', feature)

#         else:

#             df[feature] = df[feature].astype('object')

                        

#     return df
# train = convert_to_object(train, cat_features)

# test = convert_to_object(test, cat_features)



# #Verify

# for feature in train.columns[:20]:

#     print(feature, '\t', train[feature].dtype)

    

# print('\n')

# print('-'*50)

# print('\nTest: \n')



# for feature in test.columns[:20]:

#     print(feature, '\t', test[feature].dtype)
# train.isnull().sum()[:15]
# test.isnull().sum()[:15]
# for feature in train.columns[:20]:

#     if train[feature].dtype == 'object':

#         print(feature, '\t Unique categories:', train[feature].describe()[1])

#         print('-'*40)
# y_train = train.isFraud

# train = train.drop(['isFraud'], axis=1)

# num_fraud = y_train.sum()



# print('# of fraudulent transactions:', num_fraud, '\n# of training examples:', num_train)
# plt.bar([1, 2], height=[num_fraud, num_train])

# plt.title('Class Imbalance')

# plt.show()
# train_fraud = train[y_train == 1]

# train_not_fraud = train[y_train == 0]



# train_fraud.head(10)
# def get_mean_of_feature(df, feature):

#     """

#     Calculates and returns mean value of a numerical feature variable

    

#     Args:

#         df (pd.DataFrame): the dataframe

#         feature (str): the name of the numerical feature/variable as a string

        

#     Returns:

#         mean (float)

#     """

#     return df[feature].mean()



# def get_categorical_distribution_of_feature(df, feature):

#     """

#     Calculates and returns distribution of a categorical feature variable

    

#     Args:

#         df (pd.DataFrame): the dataframe

#         feature (str): the name of the categorical feature/variable as a string

        

#     Returns:

#         categorical dist (pd.Series)

#     """

#     return df[feature].value_counts() / df[feature].value_counts().sum()
# def compare_dataframes(df1, df1_description, df2, df2_description):

#     """

#     Analyze each feature and compare the difference between fraud and not fraud table

    

#     Args:

#         train_fraud (pd.DataFrame): contains the fraudulent transactions

#         train_not_fraud (pd.DataFrame): contains the non-fraud transactions

        

#     Returns:

        

#     """

    

#     # features that look interesting from visual inspection

#     features = ['TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card4', 'card6', 

#                 'P_emaildomain', 'R_emaildomain', 'id_29', 'id_30', 'id_31', 'DeviceType', 'DeviceInfo']

    

#     # Use this if analyzing ALL features of dataframes

#     # make sure have same features in both dataframes

#     #assert(sorted(train_not_fraud.columns) == sorted(train_fraud.columns))

#     #features = train_fraud.columns 

    

#     for feature in features:

#         # numerical feature

#         if df1[feature].dtype == 'int64' or df1[feature].dtype == 'float64':

#             print('\nNumerical feature (' + str(df1_description), ')\tFeature name:', feature, '\nmean:', get_mean_of_feature(df1, feature))

#             print('\nNumerical feature (' + str(df2_description), ')\tFeature name:', feature, '\nmean:', get_mean_of_feature(df2, feature))

#         # categorical feature

#         elif df1[feature].dtype == 'object': # object, a string

#             print('\nCategorical feature(' + str(df1_description), ')\tFeature name:', feature, '\nDistribution:\n', get_categorical_distribution_of_feature(df1,feature)[:10])

#             print('\nCategorical feature(' + str(df2_description), ')\tFeature name:', feature, '\nDistribution:\n', get_categorical_distribution_of_feature(df2,feature)[:10])
# compare_dataframes(train_fraud, 'Train Fraud', train_not_fraud, 'Train Not Fraud')
# # Clear up RAM (10.3GB -> 8.6GB)

# del train_fraud, train_not_fraud
# compare_dataframes(train, 'Train set', test, 'Test set')
# plt.hist(train['TransactionDT'], label='train')

# plt.hist(test['TransactionDT'], label='test')

# plt.legend()

# plt.title('Distribution of TransactionDT dates')
# # could correct for time difference in later iteration, for now, just drop column

# train.drop(['TransactionDT'], axis=1)

# test.drop(['TransactionDT'], axis=1)



# print('dropped TransactionDT')
# drop_cols = [c for c in train.columns if (train[c].isnull().sum() /  num_train) > .80]



# # also dropping V107 (though float values and VESTA did not say it was categorical, it really looks categorical in {0,1})

# # it caused problems in making predictions, after further analysis, it seemed to have weak correlation with target variable

# drop_cols.append('V107')



# print('Dropping', len(drop_cols), 'columns.')

# print('Including...', drop_cols[0:10])
# train = train.drop(drop_cols, axis=1)

# test = test.drop(drop_cols, axis=1)
# def replace_nans(df):

#     """

#     Replaces missing values (NaNs) with the mean (if numerical) and with most

#     common category (if categorical)

    

#     Args:

#         df (pd.DataFrame)

        

#     Returns:

#         df (pd.DataFrame): transformed dataframe

#     """

#     # NOTE: fillna did not work well here, recommend using replace

    

#     for feature in df.columns:

#         # replace categorical variable with most frequent

#         if df[feature].dtype == 'object':

#             df[feature] = df[feature].replace(np.nan, df[feature].value_counts().index[0])

        

#         # replace NaN in numerical columns with mean (could try median)

#         else:

#             df[feature] = df[feature].replace(np.nan, df[feature].mean()) # most common category

            

#     return df
# # train = replace_nans(train)

# # test = replace_nans(test)



# # fill in -999 where vars have NaNs

# train = train.replace(np.nan, nan_replace)

# test = test.replace(np.nan, nan_replace)
# train.head() # nice and pretty
# train.isnull().sum().sum() # 0
# test.isnull().sum().sum() # 0
# %%time

# from sklearn.preprocessing import LabelEncoder



# # When trying to encode variabels, receive the following ValueError: y contains previously unseen labels: [nan, nan, nan,... , nan]

# for feature in train.columns:

#     if train[feature].dtype == 'object' or test[feature].dtype == 'object':

#         le = LabelEncoder()

#         le.fit(list(train[feature].values) + list(test[feature].values))

#         train[feature] = le.transform(list(train[feature].values))

#         test[feature] = le.transform(list(test[feature].values))
# # Don't see any strings here, looks like the encoding worked

# train.head()
# def normalize(df):

#     """

#     Normalize numerical variables

    

#     Args:

#         df (pd.DataFrame): dataframe to be normalized

        

#     Returns:

#         df (pd.Dataframe): dataframe where each column has mean 0

#     """

#     for feature in df.columns:

#         if df[feature].dtype != 'object': # if it is numerical

#             mu = df[feature].mean()

#             sd = df[feature].std()

#             df[feature] = (df[feature] - mu) / sd

            

#             # verify mean is 0

#             mu_after = df[feature].mean()

#             #print(feature, mu_after) # checks out

            

#     return df

            
# train = normalize(train)

# test = normalize(test)
# print(train.memory_usage().sum() / 1024**3, 'GB')

# print(test.memory_usage().sum() / 1024**3, 'GB')
# def reduce_mem_usage(df, verbose=True):

#     numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

#     start_mem = df.memory_usage().sum() / 1024**2    

#     for col in df.columns:

#         col_type = df[col].dtypes

#         if col_type in numerics:

#             c_min = df[col].min()

#             c_max = df[col].max()

#             if str(col_type)[:3] == 'int':

#                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

#                     df[col] = df[col].astype(np.int8)

#                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

#                     df[col] = df[col].astype(np.int16)

#                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

#                     df[col] = df[col].astype(np.int32)

#                 elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

#                     df[col] = df[col].astype(np.int64)  

#             else:

#                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

#                     df[col] = df[col].astype(np.float16)

#                 elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

#                     df[col] = df[col].astype(np.float32)

#                 else:

#                     df[col] = df[col].astype(np.float64)    

#     end_mem = df.memory_usage().sum() / 1024**2

#     if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

#     return df
# train = reduce_mem_usage(train)

# test = reduce_mem_usage(test)
# import xgboost as xgb

# from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
# y = y_train.astype('category')
# # grid of parameters, use GridSearch to find best combination

# n_estimators = [400, 550, 700]

# gamma = [.5, 1, 3]

# max_depth = [6, 8, 10]
# import time



# start = time.time()



# # try all combinations of parameters

# for n_est in n_estimators:

#     for md in max_depth: 

#         for g in gamma:               

#                 # train/test split, hopefully with a large dataset this is sufficient to estimate roc auc

#                 X_train, X_test, y_train, y_valid = train_test_split(train, y, test_size=.3, random_state=RANDOM_SEED, shuffle=True)

                

#                 # fit

#                 clf = xgb.XGBClassifier(n_estimators=n_est,

#                                         gamma=g,

#                                         max_depth=md,

#                                         missing=nan_replace,

#                                         subsample=.8,

#                                         colsample_bytree=.8,

#                                         scale_pos_weight=20, # to correct for class imbalance

#                                         random_state=RANDOM_SEED,

#                                         tree_method='gpu_hist')

                

#                 # fit with these parameters

#                 clf.fit(X_train, y_train)

#                 del X_train, y_train

                

#                 # predict on test/ estimate roc_auc, pick model with

#                 y_pred = clf.predict_proba(X_test)

#                 del X_test, clf

                

#                 print(roc_auc_score(y_valid, y_pred[:,1]), 'with parameters n_estimators={}, max_depth={}, gamma={},'.format(n_est, md, g))

#                 del y_valid, y_pred

                

#                 now = time.time()

#                 print('ELAPSED TIME:', now-start, 'seconds')

                

#                 # print(train.memory_usage().sum() / 1024**3, 'GB')

#                 # print(y.memory_usage() / 1024**3, 'GB\n')

                

#                 # train = reduce_mem_usage(train)

                

#                 # give RAM time to clear

#                 time.sleep(10)

                
# %%time



# # define xgboost classifier

# clf = xgb.XGBClassifier(n_estimators=400,

#                             gamma=1,

#                             max_depth=6,

#                             missing=nan_replace,

#                             subsample=.8,

#                             colsample_bytree=.8,

#                             scale_pos_weight=20, # to correct for class imbalance

#                             random_state=RANDOM_SEED,

#                             tree_method='gpu_hist')

    

# # fit classifier

# clf.fit(train, y_train)

# del train, y_train
# sample_submission = pd.read_csv(os.path.join(input_path, 'sample_submission.csv'), index_col='TransactionID')



# y_pred = clf.predict_proba(test)

# del clf, test



# sample_submission['isFraud'] = y_pred[:,1]

# del y_pred

# sample_submission.to_csv('xgboost_with_tuning2.csv')