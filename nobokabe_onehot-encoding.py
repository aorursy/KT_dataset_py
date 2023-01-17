# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import GradientBoostingClassifier

from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm_notebook as tqdm

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import PowerTransformer

import category_encoders as ce

from sklearn.feature_extraction.text import TfidfVectorizer

#import nltk

#from nltk.corpus import stopwords

#from keras.preprocessing import sequence

#from keras.preprocessing.text import Tokenizer





import gc

import os
# ----------

# Load Data

# ----------

def load_data():

    print('------------------------')

    print('exec load_data() ... ')

    # Input data files are available in the "../input/" directory.

    # read files

    df_train = pd.read_csv('../input/train.csv', index_col=0)

    df_test = pd.read_csv('../input/test.csv', index_col=0)



    state_latlong = pd.read_csv('../input/statelatlong.csv')

    state_gdp = pd.read_csv('../input/US_GDP_by_State.csv')



    # save train & test index

    train_idx = df_train.index

    test_idx = df_test.index



    # nearest 7 states

    st_len = len(state_latlong)

    nearest = []

    for i in range(st_len):

        i_distance = []

        for j in range(st_len):

            if i == j:

                i_distance.append(float('inf'))

            else:

                i_distance.append(np.square(state_latlong.Longitude.values[i] - state_latlong.Longitude.values[j]) + np.square(state_latlong.Latitude.values[i] - state_latlong.Latitude.values[j]))

        nearest.append(np.argsort(i_distance)[::1][:7:])

    

    states = state_latlong.State.values

    state_latlong = pd.concat([state_latlong, pd.DataFrame(states[nearest], index=state_latlong.index, columns=['n1','n2','n3','n4','n5','n6','n7'])], axis=1)

    

    del st_len

    del nearest



    # merge train & test data

    df = pd.concat([df_train,df_test],sort=True)

    del df_train

    del df_test



    # inner join {train,test} & state_latlong

    df = df.reset_index().merge(state_latlong.rename(columns={"State":"addr_state"})).set_index(df.index.names)



    # inner join above & state_gdp

    columns = state_gdp.columns.drop(['State','year'])

    new_gdp = pd.DataFrame()

    for year, sdf in state_gdp.groupby('year'):

        tmp_gdp = sdf.reset_index(drop=True)

        new_gdp['State'] = tmp_gdp.State

        new_gdp[columns + str(year)] = tmp_gdp[columns]



    # calc difference (=annual growth)

    new_gdp[columns+'2013'] = new_gdp[columns+'2014'].values - new_gdp[columns+'2013'].values

    new_gdp[columns+'2014'] = new_gdp[columns+'2015'].values - new_gdp[columns+'2014'].values



    df = df.reset_index().merge(new_gdp.rename(columns={"State":"City"})).set_index(df.index.names)

    df.drop('City', axis=1, inplace=True)



    del columns

    del tmp_gdp

    del new_gdp

    del state_latlong

    del state_gdp

    

    return df, train_idx, test_idx, states





def sprit_X_y(df, train_idx, test_idx):

    print('------------------------')

    print('exec sprit_X_y() ... ')



    # Split to "train & test data(X)" , "label data(y_train)"

    y_train = df.loan_condition.drop(test_idx)

    X = df.drop('loan_condition', axis=1)



    # drop useless parameter

    #X.drop('issue_d', axis=1, inplace=True)

    

    return X, y_train





# -------------------

# Create New Feature 

# -------------------

def create_new_f(X):

    print('------------------------')

    print('exec create_nef_f() ... ')

    # impute NaN

    #X['annual_inc'] = X.annual_inc[X.annual_inc.isnull()].fillna(0)



    # create new feature: ratio(loan_amnt / annual_inc)

    X['loan_per_inc'] = X.loan_amnt / (X.annual_inc + 100)



    # create new feature: NaN indicate flag

    for col in X.columns:

        if X[col].isnull().any():

            X['nanflg_'+col] = X[col].isnull().astype(int)



    # create new feature: count NaN columns

    X['nan_cnt'] = X.isnull().sum(axis=1)



    # split year and month

    X['issue_d'] = pd.to_datetime(X.issue_d, format="%b-%Y")



    X['issue_d_year'] = X['issue_d'].dt.year

    X['issue_d_month'] = X['issue_d'].dt.month.astype(str)



    X['issue_d'] = X['issue_d'].astype(int)

    #X.drop('issue_d', axis=1, inplace=True)



    # split year and month

    X['earliest_cr_line'] = pd.to_datetime(X.earliest_cr_line, format="%b-%Y")



    X['earliest_cr_line_year'] = X['earliest_cr_line'].dt.year

    X['earliest_cr_line_month'] = X['earliest_cr_line'].dt.month.astype(str)



    X['earliest_cr_line'] = X['earliest_cr_line'].astype(int)



    #X.drop('earliest_cr_line', axis=1, inplace=True)



    X['relative_earliest_cr_line'] = X['issue_d'] - X['earliest_cr_line']

    #X.drop('issue_d', axis=1, inplace=True)



    # Categorize 51 states to 4 group  by longtitude.

    #X['cat_states'] = 1

    #X['cat_states'][X.Longitude >= -140] += 1

    #X['cat_states'][X.Longitude >= -97] += 1

    #X['cat_states'][X.Longitude >= -78] += 1

    #X['cat_states'] = X['cat_states'].astype(str)

    

    return X





# ------------------

# Numerical Feature 

# ------------------

def proc_num_f(X):

    print('------------------------')

    print('exec proc_num_f() ... ')



    # calc log1p

    X.loan_amnt = X.loan_amnt.apply(np.log1p)

    X.annual_inc = X.annual_inc.apply(np.log1p)



    # zipcode char to num

    X.zip_code = X.zip_code.str[0:3].astype(int)



    # char to num

    X.emp_length.fillna('-1', inplace=True)

    X.emp_length = X.emp_length.str.strip().str.replace("[a-zA-Z]","").str.replace("10\+","15").str.replace("< 1", "0").astype(int)



    # columns of numeric(int or float)

    numerics = []

    for col in X.columns:

        if X[col].dtype == 'float64' or X[col].dtype == 'int64':

            numerics.append(col)

            print(col, X[col].nunique())



    return X, numerics



# ---------------------

# Standarize/Normarize 

# ---------------------

def std_num_f(X, numerics):

    print('------------------------')

    print('exec std_num_f() ... ')

    # NaN processing

    X[numerics] = X[numerics].fillna(-999)



    std = StandardScaler()

    X[numerics] = pd.DataFrame(std.fit_transform(X[numerics].astype(float)), index=X.index, columns=X[numerics].columns)

    del std



    return X





# --------------------

# Categorical Feature 

# --------------------

def proc_cat_f(X, test_idx):

    print('------------------------')

    print('exec proc_cat_f() ... ')

    # subgrade use only second char (A4 -> 4, E2 -> 2)

    X.sub_grade = X.sub_grade.str[1:]



    for col in X.columns:

        if X[col].dtype == 'object':

            print(col,X[col].nunique())



    # only use 'title' data in test

    #X['title'] = X['title'].fillna('#')

    #titles = X['title'].drop(index=train_idx).unique()

    #X.title[~X.title.str.match('^' + '$|^'.join(titles) + '$')] = '#'

    #X.drop(['title'], axis=1, inplace=True)



    # target encoding

    #tmp_X_train = X.drop(index=test_idx).reset_index()

    #tmp_y_train = y_train.reset_index()

    #tmp_X = X.reset_index()

    #list_cols = ['addr_state']

    #te = ce.TargetEncoder(cols=list_cols, handle_unknown='ignore')

    #te.fit(tmp_X_train,tmp_y_train)

    #X = te.transform(tmp_X).set_index('ID')

    

    #del tmp_X_train

    #del tmp_y_train

    #del tmp_X



    # one hot encoding

    oh_list = ['grade', 'sub_grade', 'home_ownership', 'purpose', 'addr_state', 'initial_list_status', 'application_type', 'earliest_cr_line_month', 'issue_d_month']

    #oh_list = ['grade', 'sub_grade', 'home_ownership', 'purpose', 'addr_state', 'initial_list_status', 'application_type', 'earliest_cr_line_month']



    X = pd.concat([X.drop(columns=oh_list, axis=1), pd.get_dummies(X[oh_list])], axis=1)



    # nearest states

    #for st in states:

    #    X['addr_state_' + st][(X[['n1','n2','n3']] == st).any(axis=1)] = 0.1

    #    X['addr_state_' + st][(X[['n4','n5','n6','n7']] == st).any(axis=1)] = 0.1



    X.drop(['n1','n2','n3','n4','n5','n6','n7'], axis=1, inplace=True)



    # drop nan value column as for one hot encoding 

    #for col in X.columns:

    #    if '_nan' in col:

    #        X.drop(col, axis=1, inplace=True)



    # columns of string(=object)

    cats = []

    for col in X.columns:

        if X[col].dtype == 'object':

            cats.append(col)



    cats.remove('emp_title')

    cats.remove('title')



    # NaN processing

    X[cats] = X[cats].fillna('#')



    # ordinal encoding

    #oe = ce.OrdinalEncoder(cols=cats, return_df=False)

    #X_train[cats] = oe.fit_transform(X_train[cats])

    #X_test[cats] = oe.transform(X_test[cats])

    

    return X





# -------------

# Text Feature 

# -------------

def proc_txt_f(X, train_idx, test_idx):

    print('------------------------')

    print('exec proc_txt_f() ... ')

    # NaN processing

    X[['emp_title', 'title']] = X[['emp_title', 'title']].fillna('a')



    # TFIDF

    tfidf = TfidfVectorizer(strip_accents='ascii', stop_words='english', max_features=300)

    tfidf.fit(X.emp_title[train_idx])

    TXT = tfidf.transform(X.emp_title)



    #tfidf2 = TfidfVectorizer(strip_accents='ascii', stop_words='english', max_features=100)

    #tfidf2.fit(X.title[train_idx])

    #TXT2 = tfidf2.transform(X.title)



    X_train = pd.concat([X, pd.DataFrame(TXT.todense(), index=X.index)], axis=1)



    del tfidf

    #del tfidf2

    del TXT

    #del TXT2



    X.drop(['emp_title', 'title'], axis=1, inplace=True)

    

    return X





###############################################

# another version. transform train, test each other

###############################################

def proc_txt_f_old(X_train, X_test):

    print('------------------------')

    print('exec proc_txt_f_old() ... ')

    # NaN processing

    X_train[['emp_title', 'title']] = X_train[['emp_title', 'title']].fillna('a')

    X_test[['emp_title', 'title']] = X_test[['emp_title', 'title']].fillna('a')



    # TFIDF

    tfidf = TfidfVectorizer(strip_accents='ascii', stop_words='english', max_features=300, ngram_range=(1,2), lowercase=True)

    TXT_train = tfidf.fit_transform(X_train.emp_title)

    TXT_test = tfidf.transform(X_test.emp_title)



    #tfidf2 = TfidfVectorizer(strip_accents='ascii', stop_words='english', max_features=100)

    #TXT_train2 = tfidf2.fit_transform(X_train.title)

    #TXT_test2 = tfidf2.transform(X_test.title)



    #X_train = pd.concat([X_train, pd.DataFrame(np.hstack([TXT_train.todense(), TXT_train2.todense()]), index=X_train.index)], axis=1)

    #X_test = pd.concat([X_test, pd.DataFrame(np.hstack([TXT_test.todense(), TXT_test2.todense()]), index=X_test.index)], axis=1)

    X_train = pd.concat([X_train, pd.DataFrame(TXT_train.todense(), index=X_train.index)], axis=1)

    X_test = pd.concat([X_test, pd.DataFrame(TXT_test.todense(), index=X_test.index)], axis=1)





    del TXT_train

    del TXT_test

    #del TXT_train2

    #del TXT_test2



    X_train.drop(['emp_title','title'], axis=1, inplace=True)

    X_test.drop(['emp_title','title'], axis=1, inplace=True)

    

    return X_train, X_test





##########

# drop only

##########

def non_proc_txt_f(X):

    print('------------------------')

    print('exec non_proc_txt_f() ... ')

    X.drop(['emp_title','title'], axis=1, inplace=True)

    return X





# -----------

# Data prepare

# -----------

def split_train_test(X, train_idx, test_idx):

    print('------------------------')

    print('exec split_train_test() ... ')

    X_train = X.drop(index=test_idx)

    X_test = X.drop(index=train_idx)

    del X

    #del df

    

    return X_train, X_test





#---------

# Train & Test

#---------

def train_test(X_train, X_test, y_train):

    print('------------------------')

    print('exec train_test() ... ')

    scores = []

    total_score = 0

    y_tests = np.zeros(len(X_test.index))



    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

    epochs = [1238, 1134, 1463, 1168, 1368] # magic number



    # training phase

    for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

        X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

        X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]



        #break

        print('Magic num: %d' % epochs[i])

        clf = LGBMClassifier(boosting_type='dart', class_weight=None, colsample_bytree=0.7, importance_type='split', learning_rate=0.1, max_depth=-1, min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0, n_estimators=epochs[i], n_jobs=-1, num_leaves=31, objective=None, random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

        #clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7, importance_type='split', learning_rate=0.05, max_depth=-1, min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0, n_estimators=2500, n_jobs=-1, num_leaves=31, objective=None, random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

        clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val,y_val)])

        y_pred = clf.predict_proba(X_val)[:,1]

        score = roc_auc_score(y_val, y_pred)

        scores.append(score)

        y_tests += clf.predict_proba(X_test)[:,1]



    for i in range(0, len(scores)):

        print('CV Score of Fold_%d is %f' % (i, scores[i]))

        total_score += scores[i]

    print('Avg CV Socore: %f' % (total_score/len(scores)))



    y_test = pd.DataFrame(y_tests/len(scores), index=X_test.index)



    return y_test



# --------------

# Submit 

# --------------

def test_submit(X_test, y_test):

    print('------------------------')

    print('exec test_submit() ... ')



    # assemble result

    df_submission = pd.DataFrame(index=X_test.index)

    df_submission['loan_condition'] = y_test

    print(df_submission)



    # write file

    df_submission.to_csv('submission.csv')
%%time

##############

# main routine

##############

# load data

df, train_idx, test_idx, states = load_data()

X, y_train = sprit_X_y(df, train_idx, test_idx)



# numeric features

X = create_new_f(X)

X, numerics = proc_num_f(X)



# categorical features

X = proc_cat_f(X, test_idx)



# text features

#X = proc_txt_f(X, train_idx, test_idx)

#gc.collect()



# text features(drop)!!!!!!

#X = non_proc_txt_f(X)



# split data

X_train, X_test = split_train_test(X, train_idx, test_idx)

gc.collect()



# text features (old version)

X_train, X_test = proc_txt_f_old(X_train, X_test)

gc.collect()



# train

y_test = train_test(X_train, X_test, y_train)



# submit

test_submit(X_test, y_test)


