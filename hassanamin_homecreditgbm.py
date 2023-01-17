# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
# Modeling
import lightgbm as lgb

# Splitting data
from sklearn.model_selection import train_test_split
# Performance
from sklearn.metrics import roc_auc_score
# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder
# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import NMF
from sklearn.preprocessing import OneHotEncoder

N_FOLDS = 5
MAX_EVALS = 5

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('../input/application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('../input/application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('../input/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('../input/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

def load_data(feature_reduction,numeric_feature):
    train_features = pd.read_csv('../input/application_train.csv')
    target_features = pd.read_csv('../input/application_test.csv')
    bureau=bureau_and_balance()
    # Sample 16000 rows (10000 for training, 6000 for testing)
    if feature_reduction==True:
         train_features = train_features.sample(n = 16000, random_state = 42)
    
    # Only numeric features
    if numeric_feature==True:
        train_features = train_features.select_dtypes('number')
        target_features= target_features.select_dtypes('number')
    #Histogram analysis of training data
    data_analysis(train_features)
    # Extract the labels
    train_y_labels = np.array(train_features['TARGET'].astype(np.int32)).reshape((-1, ))
    # Extract the Target ids
    target_ids = target_features['SK_ID_CURR']
    # Combine Datasets
    df=combine_datasets(train_features,target_features)
    print("Bureau df shape:", bureau.shape)
    df = df.join(bureau, how='left', on='SK_ID_CURR')
    # Drop
    train_features = train_features.drop(columns = ['TARGET', 'SK_ID_CURR'])
    target_features= target_features.drop(columns = ['SK_ID_CURR'])
    # Encoding Features
    train_sz=train_features.shape
    train_features,target_features=data_encoding(df,train_sz)
    # Feature reduction to most relevant features
    train_red,target_red=feature_reduction_pca(train_features,train_y_labels,target_features,2,10)
    # Split into training and testing data
    train_X, test_X, train_y, test_y = train_test_split(train_red, train_y_labels, test_size = 0.2, random_state = 50)
    
    print("Training features shape train_X.shape,train_red.shape: \n", train_X.shape,train_red.shape)
    print("Testing features shape target_features.shape, target_red.shape: \n ", target_features.shape, target_red.shape)
    return train_X,test_X,train_y,test_y,target_red,target_ids
def combine_datasets(train_app,target_app):
    # Combine Datasets
    df = train_app.append(target_app).reset_index()
    return df
def data_analysis(data):
    # Histogram of target
    data['TARGET'].astype(int).plot.hist()
    # Number of each type of column
    print("Number of each type of column # ",data.dtypes.value_counts())
def data_encoding(df,train_sz):
    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0
    # Combine Datasets
    #df = train.append(target).reset_index()
    # Iterate through the columns
    for col in df:
        if df[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(df[col].unique())) <= 2:
                # Train on the training data
                le.fit(df[col])
                # Transform both training and testing data
                df[col] = le.transform(df[col])
                #target[col] = le.transform(target[col])
            
                # Keep track of how many columns were label encoded
                le_count += 1
            
    print('%d columns were label encoded.' % le_count)
    # one-hot encoding of categorical variables
    df = pd.get_dummies(df)
    # Preprocessing
    df=preprocessing(df)
    #target = pd.get_dummies(target)
    #train_sz=train.shape
    print("Train Size during encoding : \n ",train_sz[0],train_sz[1])
    train=df.iloc[:train_sz[0]]
    target=df.iloc[train_sz[0]:]
    #target = target.drop(columns = ['TARGET'])
    
    return train,target
    
def feature_reduction_pca(train_X,train_y,target_X,red_tech,n_c):
    if red_tech==1:
        reducer=PCA(n_components=n_c)
    elif red_tech==2:
        reducer=LDA(n_components=10)
    
    
    reducer.fit(train_X, train_y)
    train_reduced_samples = pd.DataFrame(reducer.transform(train_X))
    target_reduced_samples = pd.DataFrame(reducer.transform(target_X))
    
    train_X_reduced=train_reduced_samples.values
    target_X_reduced=target_reduced_samples.values
    print("Train Head after feature reduction : \n",train_reduced_samples.head())
    
    return train_X_reduced,target_X_reduced
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
def preprocessing(df):
    all_features = list(df.columns)
   # all_features.remove('TARGET')
   # all_features.remove('SK_ID_CURR')
    scaler=RobustScaler()
    # fill na
    for feature in all_features:
        df[feature] = df[feature].fillna(df[feature].mean())
    
    # scaling
    df[all_features] = pd.DataFrame(scaler.fit_transform(df[all_features]))
    
    return df

def lgb_preprocessing(train_X,test_X,train_y,test_y):
    # Create a training and testing dataset
    train_set = lgb.Dataset(data = train_X, label = train_y)
    test_set = lgb.Dataset(data = test_X, label = test_y)
    return train_set,test_set
def lgb_crossvalidation(train_set):
    # Get default hyperparameters
    #model = lgb.LGBMClassifier()
    model = lgb.LGBMClassifier(
            boosting_type='gbdt',
            objective='binary',
            n_estimators=1000,
            learning_rate=0.1,
            num_leaves=31,
            feature_fraction=0.8,
            subsample=0.8,
            max_depth=8,
            reg_alpha=1,
            reg_lambda=1,
            min_child_weight=40,
            random_state=2018,
            nthread=-1)
    
    default_params = model.get_params()

    # Remove the number of estimators because we set this to 10000 in the cv call
    del default_params['n_estimators']

    # Cross validation with early stopping
    cv_results = lgb.cv(default_params, train_set, num_boost_round = 10000, early_stopping_rounds = 100, 
                    metrics = 'auc', nfold = N_FOLDS, seed = 42)
    # Printing results
    print('The maximum validation ROC AUC was: {:.5f} with a standard deviation of {:.5f}.'.format(cv_results['auc-mean'][-1], cv_results['auc-stdv'][-1]))
    print('The optimal number of boosting rounds (estimators) was {}.'.format(len(cv_results['auc-mean'])))
    
    #Training using optimal parameters
    model.n_estimators = len(cv_results['auc-mean'])
    # Train and make predicions with model
    model.fit(train_X, train_y)
    
    return model,cv_results
def lgb_testing(model,test_X,test_y):
    preds = model.predict_proba(test_X)[:, 1]
    baseline_auc = roc_auc_score(test_y, preds)

    print('The baseline model scores {:.5f} ROC AUC on the test set.'.format(baseline_auc))
    return preds
def result_submission(target_ids,preds):
    submission = pd.DataFrame({'SK_ID_CURR': target_ids, 'TARGET': preds})
    submission.to_csv('submission_simple_features_random.csv', index = False)
train_X,test_X,train_y,test_y,target,target_ids=load_data(False,False)
train_set,test_set=lgb_preprocessing(train_X,test_X,train_y,test_y)
model,cv_results=lgb_crossvalidation(train_set)
test_preds=lgb_testing(model,test_X,test_y)
application_train_test()
# Predictions on the target data
preds = model.predict_proba(target)[:, 1]
result_submission(target_ids,preds)
