import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import scipy as sp

import statistics 

from pandas import DataFrame, Series

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm

from sklearn.preprocessing import quantile_transform

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import LabelEncoder 

from datetime import datetime



from sklearn.tree import ExtraTreeClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble.weight_boosting import AdaBoostClassifier

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.ensemble.bagging import BaggingClassifier

from sklearn.ensemble.forest import ExtraTreesClassifier

from sklearn.ensemble.forest import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model.stochastic_gradient import SGDClassifier

import lightgbm as lgb

from lightgbm import LGBMClassifier



import seaborn as sns

from sklearn.decomposition import TruncatedSVD



import category_encoders as ce

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve



from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix
df_train = pd.read_csv('/kaggle/input/homework-for-students3/train.csv', index_col=0, parse_dates=['issue_d'])

df_test =pd.read_csv('/kaggle/input/homework-for-students3/test.csv', index_col=0, parse_dates=['issue_d'])



year = df_train.issue_d.dt.year

df_train  = df_train[year >= 2014]

columns = df_train.columns



y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis = 1)

X_test  = df_test



del df_train

del df_test
drop_col = ['issue_d','earliest_cr_line']

X_train = X_train.drop(columns=drop_col)

X_test  = X_test.drop(columns=drop_col)
def capping(series, min_threshold, max_threshold):

    series_filtered = series.copy()

    index_outlier_up = [series_filtered  >= max_threshold]

    index_outlier_low = [series_filtered <= min_threshold]

    series_filtered.iloc[index_outlier_up] = max_threshold

    series_filtered.iloc[index_outlier_low] = min_threshold

    return series_filtered
X_all = pd.concat([X_train,X_test],axis =0)



X_all['Asset'] = X_all['tot_cur_bal'] + X_all['annual_inc']*2



X_train = X_all.iloc[:X_train.shape[0],:]

X_test = X_all.iloc[X_train.shape[0]:,:]
col = 'loan_amnt'

# cap

max_threshold = 40000

X_train[col] = capping(X_train[col],0, max_threshold)

X_test[col]= capping(X_test[col],0, max_threshold)



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col = 'installment'

# cap

max_threshold = 1600

X_train[col] = capping(X_train[col],0, max_threshold)

X_test[col]= capping(X_test[col],0, max_threshold)



# apply log

X_train[col] = X_train[col].apply(np.log1p)

X_test[col] = X_test[col].apply(np.log1p)



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col = 'grade'

mapping_dict = {

    "grade": {

        "A": 0,"B": 1,"C": 2,"D": 3,"E": 4,"F": 5,"G": 6

        }

}



X_train = X_train.replace(mapping_dict)

X_test = X_test.replace(mapping_dict)

mapping_col = ['grade']

X_train[mapping_col] = X_train[mapping_col].fillna(-1)

X_test[mapping_col] = X_test[mapping_col].fillna(-1)

X_train[mapping_col] = X_train[mapping_col].astype(int)

X_test[mapping_col] = X_test[mapping_col].astype(int)





# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col = 'sub_grade'



mapping_dict = {

    "sub_grade": {

        "A1": 0,"A2": 1,"A3": 2,"A4": 3,"A5": 4,

        "B1": 5,"B2": 6,"B3": 7,"B4": 8,"B5": 9,

        "C1": 10,"C2": 11,"C3": 12,"C4": 13,"C5": 14,

        "D1": 15,"D2": 16,"D3": 17,"D4": 18,"D5": 19,

        "E1": 20,"E2": 21,"E3": 22,"E4": 23,"E5": 24,

        "F1": 25,"F2": 26,"F3": 27,"F4": 28,"F5": 29,

        "G1": 30,"G2": 31,"G3": 32,"G4": 33,"G5": 34

        }

}

X_train = X_train.replace(mapping_dict)

X_test = X_test.replace(mapping_dict)

mapping_col = ['sub_grade']

X_train[mapping_col] = X_train[mapping_col].fillna(-1)

X_test[mapping_col] = X_test[mapping_col].fillna(-1)

X_train[mapping_col] = X_train[mapping_col].astype(int)

X_test[mapping_col] = X_test[mapping_col].astype(int)



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()



col    = 'emp_length'

target = 'loan_condition'



X_temp = pd.concat([X_train, y_train], axis=1)

# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

X_test[col] = X_test[col].map(summary) 

    

enc_train = Series(np.zeros(len(X_train)), index=X_train.index)

    

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)



X_train[col] = enc_train



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col    = 'home_ownership'

target = 'loan_condition'



X_temp = pd.concat([X_train, y_train], axis=1)

# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

X_test[col] = X_test[col].map(summary) 

    

enc_train = Series(np.zeros(len(X_train)), index=X_train.index)

    

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)



X_train[col] = enc_train



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col = 'annual_inc'

# cap

max_threshold = 250000

X_train[col] = capping(X_train[col],0, max_threshold)

X_test[col]= capping(X_test[col],0, max_threshold)



# apply log

X_train[col] = X_train[col].apply(np.log1p)

X_test[col] = X_test[col].apply(np.log1p)



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col    = 'purpose'

target = 'loan_condition'



X_temp = pd.concat([X_train, y_train], axis=1)

# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

X_test[col] = X_test[col].map(summary) 

    

enc_train = Series(np.zeros(len(X_train)), index=X_train.index)

    

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)



X_train[col] = enc_train



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col    = 'title'

target = 'loan_condition'



X_temp = pd.concat([X_train, y_train], axis=1)

# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

X_test[col] = X_test[col].map(summary) 

    

enc_train = Series(np.zeros(len(X_train)), index=X_train.index)

    

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)



X_train[col] = enc_train



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col    = 'zip_code'

target = 'loan_condition'



X_train[col] = X_train[col].str[0:2]

X_test[col] = X_test[col].str[0:2]



X_temp = pd.concat([X_train, y_train], axis=1)

# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

X_test[col] = X_test[col].map(summary) 

    

enc_train = Series(np.zeros(len(X_train)), index=X_train.index)

    

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)



X_train[col] = enc_train



# cap

max_threshold = 0.3

min_threshold = 0.1

X_train[col] = capping(X_train[col], min_threshold , max_threshold)

X_test[col]= capping(X_test[col], min_threshold , max_threshold)





# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()

col    = 'addr_state'

target = 'loan_condition'



X_temp = pd.concat([X_train, y_train], axis=1)

# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

X_test[col] = X_test[col].map(summary) 

    

enc_train = Series(np.zeros(len(X_train)), index=X_train.index)

    

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)



X_train[col] = enc_train



# cap

max_threshold = 0.25

min_threshold = 0.12

X_train[col] = capping(X_train[col], min_threshold , max_threshold)

X_test[col]= capping(X_test[col], min_threshold , max_threshold)



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col = 'dti'

# cap

max_threshold = 40

X_train[col] = capping(X_train[col],0, max_threshold)

X_test[col]= capping(X_test[col],0, max_threshold)



# apply log

#X_train[col] = X_train[col].apply(np.log1p)

#X_test[col] = X_test[col].apply(np.log1p)



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col = 'delinq_2yrs'

# cap

max_threshold = 3

X_train[col] = capping(X_train[col],0, max_threshold)

X_test[col]= capping(X_test[col],0, max_threshold)







# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col = 'inq_last_6mths'

# cap

max_threshold = 6

X_train[col] = capping(X_train[col],0, max_threshold)

X_test[col]= capping(X_test[col],0, max_threshold)



# apply log

#X_train[col] = X_train[col].apply(np.log1p)

#X_test[col] = X_test[col].apply(np.log1p)



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()



#1つだけあるtestのmissingを0で補完

X_test[col].fillna(0, inplace=True)
col = 'mths_since_last_delinq'

# cap

max_threshold = 80

X_train[col] = capping(X_train[col],0, max_threshold)

X_test[col]= capping(X_test[col],0, max_threshold)



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col = 'mths_since_last_record'

# cap

max_threshold = 130

X_train[col] = capping(X_train[col],0, max_threshold)

X_test[col]= capping(X_test[col],0, max_threshold)



# apply log

#X_train[col] = X_train[col].apply(np.log1p)

#X_test[col] = X_test[col].apply(np.log1p)



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col = 'open_acc'

# cap

max_threshold = 40

X_train[col] = capping(X_train[col],0, max_threshold)

X_test[col]= capping(X_test[col],0, max_threshold)



# apply log

X_train[col] = X_train[col].apply(np.log1p)

X_test[col] = X_test[col].apply(np.log1p)



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col = 'pub_rec'

# cap

max_threshold = 1

X_train[col] = capping(X_train[col],0, max_threshold)

X_test[col]= capping(X_test[col],0, max_threshold)



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col = 'revol_bal'

# cap

max_threshold = 60000

X_train[col] = capping(X_train[col],0, max_threshold)

X_test[col]= capping(X_test[col],0, max_threshold)



# apply log

X_train[col] = X_train[col].apply(np.log1p)

X_test[col] = X_test[col].apply(np.log1p)



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col = 'revol_util'

# cap

max_threshold = 100

X_train[col] = capping(X_train[col],0, max_threshold)

X_test[col]= capping(X_test[col],0, max_threshold)



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col = 'total_acc'

# cap

max_threshold = 75

X_train[col] = capping(X_train[col],0, max_threshold)

X_test[col]= capping(X_test[col],0, max_threshold)



# apply log

X_train[col] = X_train[col].apply(np.log1p)

X_test[col] = X_test[col].apply(np.log1p)



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col = 'initial_list_status'



ordinalcat_feature = ['initial_list_status']

oe = OrdinalEncoder(cols=ordinalcat_feature)



X_train = oe.fit_transform(X_train)

X_test = oe.transform(X_test)



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col = 'collections_12_mths_ex_med'

# cap

max_threshold = 1

X_train[col] = capping(X_train[col],0, max_threshold)

X_test[col]= capping(X_test[col],0, max_threshold)



# apply log

#X_train[col] = X_train[col].apply(np.log1p)

#X_test[col] = X_test[col].apply(np.log1p)



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col = 'mths_since_last_major_derog'

# cap

max_threshold = 100

X_train[col] = capping(X_train[col],0, max_threshold)

X_test[col]= capping(X_test[col],0, max_threshold)



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col = 'application_type'



ordinalcat_feature = ['application_type']

oe = OrdinalEncoder(cols=ordinalcat_feature)



X_train = oe.fit_transform(X_train)

X_test = oe.transform(X_test)



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
# delete later

col = acc_now_delinq



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col = 'tot_coll_amt'

# cap

max_threshold = 1

X_train[col] = capping(X_train[col],0, max_threshold)

X_test[col]= capping(X_test[col],0, max_threshold)



# apply log

#X_train[col] = X_train[col].apply(np.log1p)

#X_test[col] = X_test[col].apply(np.log1p)



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col = 'tot_cur_bal'

# cap

max_threshold = 800000

X_train[col] = capping(X_train[col],0, max_threshold)

X_test[col]= capping(X_test[col],0, max_threshold)



# apply log

X_train[col] = X_train[col].apply(np.log1p)

X_test[col] = X_test[col].apply(np.log1p)



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col = 'Asset'

# cap

max_threshold = 1000000

X_train[col] = capping(X_train[col],0, max_threshold)

X_test[col]= capping(X_test[col],0, max_threshold)



# apply log

X_train[col] = X_train[col].apply(np.log1p)

X_test[col] = X_test[col].apply(np.log1p)



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
col    = 'emp_title'

col4   = 'emp_title4'

target = 'loan_condition'



X_train[col4] = X_train[col].str[0:4]

X_test[col4] = X_test[col].str[0:4]



X_temp = pd.concat([X_train, y_train], axis=1)

# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col4])[target].mean()

X_test[col4] = X_test[col4].map(summary) 

    

enc_train = Series(np.zeros(len(X_train)), index=X_train.index)

    

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col4])[target].mean()

    enc_train.iloc[val_ix] = X_val[col4].map(summary)



X_train[col4] = enc_train



# cap

max_threshold = 0.5

min_threshold = 0

X_train[col4] = capping(X_train[col4],min_threshold, max_threshold)

X_test[col4]= capping(X_test[col4],min_threshold, max_threshold)



# check histgram

plt.figure(figsize=[7,7])

X_train[col4].hist(density=True, alpha=0.5, bins=20)

X_test[col4].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col4)

plt.ylabel('density')

plt.show()



#-------------------------------------------------------------

col    = 'emp_title'

target = 'loan_condition'



X_temp = pd.concat([X_train, y_train], axis=1)

# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

X_test[col] = X_test[col].map(summary) 

    

enc_train = Series(np.zeros(len(X_train)), index=X_train.index)

    

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)



X_train[col] = enc_train



# cap

max_threshold = 0.5

min_threshold = 0

X_train[col] = capping(X_train[col],min_threshold, max_threshold)

X_test[col]= capping(X_test[col],min_threshold, max_threshold)



# check histgram

plt.figure(figsize=[7,7])

X_train[col].hist(density=True, alpha=0.5, bins=20)

X_test[col].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(col)

plt.ylabel('density')

plt.show()
drop_col = ['title','acc_now_delinq','grade','installment']

X_train = X_train.drop(columns=drop_col)

X_test  = X_test.drop(columns=drop_col)
X_train1 = X_train

X_train2 = X_train

X_train3 = X_train

X_test1  = X_test

X_test2  = X_test

X_test3  = X_test
drop_col = ['emp_title']

X_train1 = X_train1.drop(columns=drop_col)

X_test1  = X_test1.drop(columns=drop_col)
drop_col = ['emp_title']

X_train2 = X_train2.drop(columns=drop_col)

X_test2  = X_test2.drop(columns=drop_col)
ratio_PCA = 0.95

from sklearn.decomposition import PCA



sequence_col = ['sub_grade'

,'loan_amnt'

,'annual_inc'

,'dti'

,'Asset']



X_train_sequence = X_train2[sequence_col]

X_test_sequence = X_test2[sequence_col]



scaler = StandardScaler()

scaler.fit(X_train_sequence)

X_train_sequence[sequence_col] = scaler.transform(X_train_sequence[sequence_col])

X_test_sequence[sequence_col] = scaler.transform(X_test_sequence[sequence_col])



X_train_sequence.fillna(X_train_sequence.median(), inplace=True)

X_test_sequence.fillna(X_test_sequence.median(), inplace=True)



pca = PCA(n_components=X_train_sequence.shape[1])

#モデルのパラメータをfitして取得しPCAオブジェクトへ格納

pca.fit(X_train_sequence)

sorted_variance = sorted(pca.explained_variance_ratio_,reverse=True)

cum_sum = np.cumsum(sorted_variance)

ratio = cum_sum / np.sum(sorted_variance)

col_num = len(ratio) - len(ratio[ratio>ratio_PCA]) +1

if col_num >= X_train_sequence.shape[1]:

    col_num = X_train_sequence.shape[1]-1

sorted_variance = sorted(pca.explained_variance_ratio_,reverse=True)

 # 累積寄与率が90%以上になるように



np.set_printoptions(suppress=True)

# SVD

svd = TruncatedSVD(n_components=col_num, n_iter=7, random_state=42)

svd.fit(X_train_sequence)



X_train_sequence = svd.transform(X_train_sequence)

X_test_sequence  = svd.transform(X_test_sequence)



column_name = []

for c in range(0,col_num):

    column_name.append('PCA' + str(c))



X_train_sequence = pd.DataFrame(X_train_sequence,columns = column_name)

X_test_sequence = pd.DataFrame(X_test_sequence,columns = column_name)

X_train_sequence.index = X_train.index

X_test_sequence.index = X_test.index



drop_col = sequence_col

X_train2 = X_train2.drop(columns=drop_col)

X_test2  = X_test2.drop(columns=drop_col)



X_train2 = pd.concat([X_train2, X_train_sequence], axis=1)

X_test2 = pd.concat([X_test2, X_test_sequence], axis=1)
drop_col = ['emp_title4']

X_train3 = X_train3.drop(columns=drop_col)

X_test3  = X_test3.drop(columns=drop_col)
# create model and prediction 

clf = LGBMClassifier(boosting_type = 'gbdt',class_weight='balanced')

clf.fit(X_train1, y_train, eval_metric='auc')

y_pred1 = clf.predict_proba(X_test1)[:,1]



clf = LGBMClassifier(boosting_type = 'gbdt',class_weight='balanced')

clf.fit(X_train2, y_train, eval_metric='auc')

y_pred2 = clf.predict_proba(X_test2)[:,1]



clf = LGBMClassifier(boosting_type = 'gbdt',class_weight='balanced')

clf.fit(X_train3, y_train, eval_metric='auc')

y_pred3 = clf.predict_proba(X_test3)[:,1]



y_pred  = (y_pred1+y_pred2+y_pred3)/3 
Y_predAll = pd.DataFrame()

Y_predAll['y_pred1'] = y_pred1

Y_predAll['y_pred2'] = y_pred2

Y_predAll['y_pred3'] = y_pred3



mask = np.zeros_like(Y_predAll.corr(method ='spearman'))

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(15, 10))

    ax = sns.heatmap(Y_predAll.corr(method ='spearman'), mask=mask, vmax=1, vmin=0.8, square=True,linewidths=.5,xticklabels=1, yticklabels=1)

# memo: 結構相関高いのが不安...
submission = pd.read_csv('/kaggle/input/homework-for-students3/sample_submission.csv', index_col=0)

submission.loan_condition = y_pred

submission.to_csv('submission.csv')