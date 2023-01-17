# Date wrangling

import datetime



# Data wrangling

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

import gc



# Package to track the optimizing of parameters 

import time



# Status tracker

from tqdm import tqdm



# xgboost library

import xgboost as xgb

from xgboost import plot_importance



# gbm library

import lightgbm as lgb



# One hot encoding 

from sklearn import preprocessing



# Dimensionality reduction

from sklearn.decomposition import PCA



# Hyper parameter optimization

from sklearn.model_selection import KFold, TimeSeriesSplit, StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.metrics import make_scorer

from hyperopt import fmin, tpe, hp, space_eval



#TODO move analyzing and ploting to a class 

analyze_data = False

plot_data = False
# Reading the training data

train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
# Seeing the columns of the data

print("Identity columns:", train_identity.columns)

print("Transaction columns:", train_transaction.columns)



# Shapes

print("Identity shape:", train_identity.shape)

print("Transaction shape", train_transaction.shape)
# Merging the two data sets together

d = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

print("Shape of merged data", d.shape)
# Reading the test set

test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')

test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')



# Merging to one data frame

d_test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

print(f'Test data shape: {d_test.shape}')
# Freeing up memory

del test_identity, test_transaction, train_identity, train_transaction

gc.collect()



# Convergin inf to NA

d = d.replace([np.inf, -np.inf], np.nan)

d_test = d_test.replace([np.inf, -np.inf], np.nan)
print(d.head(10))
# Balance of data

print('Percent of fraud transactions:', d[d['isFraud']==1].shape[0] * 100 /d.shape[0])

print('Percent of good transactions:', d[d['isFraud']==0].shape[0] * 100 /d.shape[0])
if analyze_data:

    features = d.columns

    feature_info = {}

    row_count = d.shape[0]



    for feature in tqdm(features):

        NA_count = d[d[feature].isna()].shape[0]

        NA_share = round(NA_count * 100/row_count, 4)

        unique_values_count = len(d[feature].unique())

        coltype = d[feature].dtype.kind



        feature_info.update({

            feature: {

                'NA_count': NA_count,

                'NA_share': NA_share,

                'unique_values_count': unique_values_count,

                'coltype': coltype

            }

        })

    

    feature_df = pd.DataFrame({

        'NA_share': [feature_info.get(x).get('NA_share') for x in feature_info],

        'NA_count': [feature_info.get(x).get('NA_count') for x in feature_info],

        'unique_values_count': [feature_info.get(x).get('unique_values_count') for x in feature_info],

        'coltype': [feature_info.get(x).get('coltype') for x in feature_info],

        'feature': [x for x in feature_info.keys()]

    }).sort_values('NA_share')

    print(feature_df)
if analyze_data:

    na_shares = np.array([feature_info.get(x).get('NA_share') for x in feature_info])

    labels, counts = np.unique(na_shares, return_counts=True)

    count_df = pd.DataFrame({

        'na_share': labels, 

        'count': counts

    }).sort_values('na_share', ascending=False)

    print(count_df.head(20))

    plt.hist(na_shares)

    plt.show()
if analyze_data:

    # Getting the columns for each of the data type

    numeric_cols = [x for x in feature_info if feature_info.get(x).get('coltype')=='f']

    numeric_cols += [x for x in feature_info if feature_info.get(x).get('coltype')=='i']



    categorical_cols = [x for x in feature_info if (feature_info.get(x).get('coltype')=='O')]



    # Removing unwanted features

    numeric_cols = [x for x in numeric_cols if x not in ['isFraud', 'TransactionID']]



    # Getting the datatypes

    height = [len(numeric_cols), len(categorical_cols)]

    bars = ('numeric', 'categorical')

    y_pos = np.arange(len(bars))



    # Drawing the distribution

    plt.bar(y_pos, height)

    plt.xticks(y_pos, bars)

    plt.show()
if analyze_data and plot_data:

    for i, float_col in enumerate(numeric_cols):

        plt.figure(figsize=(16, 8))

        sns.violinplot(x='isFraud', y=float_col, data=d, hue='isFraud').set_title(float_col)

        plt.show()
# Defining a feature engineering class

class featureEngineer:

    

    def __init__(self):

        self.email_map = {

        'gmail': 'google', 

        'att.net': 'att', 

        'twc.com': 'spectrum', 

        'scranton.edu': 'other', 

        'optonline.net': 'other', 

        'hotmail.co.uk': 'microsoft',

        'comcast.net': 'other', 

        'yahoo.com.mx': 'yahoo', 

        'yahoo.fr': 'yahoo',

        'yahoo.es': 'yahoo', 

        'charter.net': 'spectrum', 

        'live.com': 'microsoft', 

        'aim.com': 'aol', 

        'hotmail.de': 'microsoft', 

        'centurylink.net': 'centurylink',

        'gmail.com': 'google', 

        'me.com': 'apple', 

        'earthlink.net': 'other', 

        'gmx.de': 'other',

        'web.de': 'other', 

        'cfl.rr.com': 'other', 

        'hotmail.com': 'microsoft', 

        'protonmail.com': 'other', 

        'hotmail.fr': 'microsoft', 

        'windstream.net': 'other', 

        'outlook.es': 'microsoft', 

        'yahoo.co.jp': 'yahoo', 

        'yahoo.de': 'yahoo',

        'servicios-ta.com': 'other', 

        'netzero.net': 'other', 

        'suddenlink.net': 'other',

        'roadrunner.com': 'other', 

        'sc.rr.com': 'other', 

        'live.fr': 'microsoft',

        'verizon.net': 'yahoo', 

        'msn.com': 'microsoft', 

        'q.com': 'centurylink', 

        'prodigy.net.mx': 'att', 

        'frontier.com': 'yahoo', 

        'anonymous.com': 'other', 

        'rocketmail.com': 'yahoo', 

        'sbcglobal.net': 'att', 

        'frontiernet.net': 'yahoo', 

        'ymail.com': 'yahoo', 

        'outlook.com': 'microsoft', 

        'mail.com': 'other', 

        'bellsouth.net': 'other', 

        'embarqmail.com': 'centurylink', 

        'cableone.net': 'other', 

        'hotmail.es': 'microsoft', 

        'mac.com': 'apple', 

        'yahoo.co.uk': 'yahoo',

        'netzero.com': 'other', 

        'yahoo.com': 'yahoo', 

        'live.com.mx': 'microsoft', 

        'ptd.net': 'other', 

        'cox.net': 'other',

        'aol.com': 'aol', 

        'juno.com': 'other', 

        'icloud.com': 'apple'

        }



        self.minmaxcols = ['TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'D1', 'D15', 'dist1']

        

        self.logcols = ['TransactionAmt']



    def feature_engineer(self, X):

        """

        A function that transforms features in a given dataset by certain rules

        """

        for col in X.columns:



            if col in self.logcols:

                X[col + '_log'] = np.log(X[col])

            

            if col in self.minmaxcols:

                normalizer = preprocessing.MinMaxScaler()

                X[col + '_scaled'] = normalizer.fit_transform(X[col].values.reshape(-1, 1)) 



            if col in ['P_emaildomain', 'R_emaildomain']:

                X[col + '_provider'] = X[col].map(self.email_map)

                X[col + '_suffix'] = X[col].map(lambda x: str(x).split('.')[-1])



            if col == 'TransactionDT':

                START_DATE = '2017-12-01'

                startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")

                X["Date"] = X[col].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))

                X['Weekdays'] = X['Date'].dt.dayofweek

                X['Hours'] = X['Date'].dt.hour

                X['Days'] = X['Date'].dt.day

                X['Weekdays_Hours'] = X['Weekdays'] + X['Hours']



            if col == 'id_30':

                X[col + '_cleaned'] = X[col]

                for ops in ['windows', 'ios', 'mac', 'android', 'linux']:

                    X.loc[X[col].str.lower().str.contains(ops, na=False), col + '_cleaned'] = ops



            if col == 'id_31':

                X[col + '_cleaned'] = np.nan

                for ops in ['chrome', 'firefox', 'edge', 'ie', 

                            'android', 'opera', 'safari', 'samsung', 

                            'google', 'blackberry']:

                    X.loc[X[col].str.lower().str.contains(ops, na=False), col + '_cleaned'] = ops

                # Creating a column to check if the device is a mobile

                X.loc[X[col].str.lower().str.contains('mobile', na=False), col + '_cleaned_mobile'] = "1"

        

        # Crude aggregations

        X['P_emaildomain__addr1'] = X['P_emaildomain'] + '__' + X['addr1'].astype(str)

        X['card1__card2'] = X['card1'].astype(str) + '__' + X['card2'].astype(str)

        X['card1__addr1'] = X['card1'].astype(str) + '__' + X['addr1'].astype(str)

        X['card2__addr1'] = X['card2'].astype(str) + '__' + X['addr1'].astype(str)

        X['card12__addr1'] = X['card1__card2'] + '__' + X['addr1'].astype(str)    



        X['TransactionAmt_to_mean_card1'] = X['TransactionAmt'] / X.groupby(['card1'])['TransactionAmt'].transform('mean')

        X['TransactionAmt_to_mean_card2'] = X['TransactionAmt'] / X.groupby(['card2'])['TransactionAmt'].transform('mean')

        X['TransactionAmt_to_mean_card3'] = X['TransactionAmt'] / X.groupby(['card3'])['TransactionAmt'].transform('mean')

        X['TransactionAmt_to_mean_card5'] = X['TransactionAmt'] / X.groupby(['card5'])['TransactionAmt'].transform('mean')

        

        X['TransactionAmt_to_std_card1'] = X['TransactionAmt'] / X.groupby(['card1'])['TransactionAmt'].transform('std')

        X['TransactionAmt_to_std_card2'] = X['TransactionAmt'] / X.groupby(['card2'])['TransactionAmt'].transform('std')

        X['TransactionAmt_to_std_card3'] = X['TransactionAmt'] / X.groupby(['card3'])['TransactionAmt'].transform('std')

        X['TransactionAmt_to_std_card5'] = X['TransactionAmt'] / X.groupby(['card5'])['TransactionAmt'].transform('std')

        

        return X
# Defining a class to preproces data 

class preprocesingEngineer:

    """

    Class that preprocess data before modeling

    """

    def __init__(self):

        self.numeric_cols = [

            'TransactionAmt_log', 'card1', 'card2', 'card3', 'card5', 

            'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',

            'D1', 'Hours', 'dist1', 'D15', 

            'TransactionAmt_to_mean_card1', 'TransactionAmt_to_mean_card2', 'TransactionAmt_to_mean_card3', 'TransactionAmt_to_mean_card5', 

            'TransactionAmt_to_std_card1', 'TransactionAmt_to_std_card2', 'TransactionAmt_to_std_card3', 'TransactionAmt_to_std_card5',

            'id_02'

        ]

        

        self.categorical_cols = [

            'ProductCD', 

            'P_emaildomain_provider', 'P_emaildomain_suffix', 

            'R_emaildomain_provider', 'R_emaildomain_suffix',

            'id_30_cleaned', 'id_31_cleaned',

            'addr1', 'addr2', 

            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',

            'Weekdays', 'Days', 'Weekdays_Hours',

            "id_12", "id_13","id_14","id_15","id_16","id_17","id_18","id_19","id_20","id_21","id_22","id_23","id_24",

            "id_25","id_26","id_27","id_28","id_29","id_30","id_32","id_34", 'id_36', "id_37", "id_38",

            'P_emaildomain__addr1', 'card1__card2', 'card1__addr1', 'card2__addr1',

            'card12__addr1',

        ]

        

        self.target_col = 'isFraud'

        

    def to_string(self, X):

        """

        A method to convert categorical columns to strings

        """

        for feature in self.categorical_cols:

            X[feature] = [str(x) for x in X[feature].values]

            

        return X    

    

    def encode_labels(self, X, X_test=pd.DataFrame({})):

        """

        A method to label encode categorical columns

        """

        for feature in self.categorical_cols:

            # Initiating the label encoder

            lbl = preprocessing.LabelEncoder()

            

            # Getting unique column values

            features = list(set(X[feature].values))

            if not X_test.empty:

                features += list(set(X[feature].values))

            

            # Fitting the label encoder    

            lbl.fit(features)

            

            # Transforming the original feature

            X[feature] = lbl.transform(list(X[feature].values))

            

        return X      

    

    def leave_final_cols(self, X, leave_target=False):

        """

        A method to leave only the columns which will be used in modeling

        """

        cols = self.numeric_cols + self.categorical_cols

        

        if leave_target:

            return X[self.target_col].values, X[cols]

        else:

            return X[cols]            
class modelingEngineer:

    """

    Class for creating and using boosting models

    """

    

    def __init__(self):

        self.NFOLDS = 5

        

        self.top_ft = 50

        

        self.lgb_parameters={

            'objective': 'binary',

            'metric': 'auc',

            'num_leaves': 600,

            'min_child_weight': 0.03,

            'feature_fraction': 0.27,

            'bagging_fraction': 0.31,

            'min_data_in_leaf': 106,

            'n_jobs': -1,

            'learning_rate': 0.005,

            'max_depth': -1,

            'tree_learner': 'serial',

            'colsample_bytree': 0.5,

            'subsample_freq': 1,

            'subsample': 0.7,

            'max_bin': 300,

            'verbose': -1,

            'early_stopping_rounds': 200,

        }

        

    def fit_model(self, X_train, Y_train, X_test):

        """

        Fits the model using cross-validation and early stopping

        """

        folds = KFold(n_splits=self.NFOLDS)

        splits = folds.split(X_train, Y_train)



        feature_importances = pd.DataFrame()

        feature_importances['feature'] = X_train.columns

        

        y_preds = np.zeros(X_test.shape[0])

        

        for fold_n, (train_index, valid_index) in enumerate(splits):

            x_train, x_valid = X_train.iloc[train_index], X_train.iloc[valid_index]

            y_train, y_valid = Y_train[train_index], Y_train[valid_index]



            dtrain = lgb.Dataset(x_train, label=y_train)

            dvalid = lgb.Dataset(x_valid, label=y_valid)



            clf = lgb.train(self.lgb_parameters, dtrain, 10000, valid_sets = [dvalid], verbose_eval=100)

            

            feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()



            # Predicting the true values of Y

            y_preds += clf.predict(X_test) / self.NFOLDS

            

            del x_train, x_valid, y_train, y_valid

            gc.collect()

       

        return y_preds, feature_importances

    

    def plot_feature_importance(self, feature_importances):

        feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(self.NFOLDS)]].mean(axis=1)

        sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(self.top_ft), x='average', y='feature')

        plt.show()
# Defining the global pipeline parameters

optimize_params = True

N_evals = 10

use_gpu = False



class pipeline(featureEngineer, preprocesingEngineer, modelingEngineer):

    

    def __init__(self):

        self.d_train = d

        self.d_test = d_test

        

    def pipeline(self):

        """

        Pipeline that creates new features, preproces and models data

        """

        # Creating new features

        fe = featureEngineer()

        X_train = fe.feature_engineer(self.d_train)

        X_test = fe.feature_engineer(self.d_test)

        

        # Preprocesing data

        pre = preprocesingEngineer()

        X_train = pre.to_string(X_train)

        X_test = pre.to_string(X_test)



        X_train = pre.encode_labels(X_train, X_test)

        X_test = pre.encode_labels(X_test, X_train)



        Y_train, X_train = pre.leave_final_cols(X_train, leave_target=True)

        X_test = pre.leave_final_cols(X_test)

        

        # Modeling data 

        me = modelingEngineer()



        # Fitting the model and getting predictions

        y_preds, feature_importances = me.fit_model(X_train, Y_train, X_test)

        

        # Ploting the most important features

        fig, ax = plt.subplots(figsize=(15, 20))

        me.plot_feature_importance(feature_importances)

    

        # Forecasting the test set and creating file for submission

        d_upload = pd.DataFrame({

            'TransactionID': self.d_test['TransactionID'],

            'isFraud': y_preds})

        

        return d_upload
# Fitting the pipeline

pipe = pipeline()

d_upload = pipe.pipeline()



# Saving for uplaod

d_upload.to_csv('submission.csv', index=False)