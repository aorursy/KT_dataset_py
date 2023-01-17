!pip install --no-warn-conflicts -U -q scikit-learn==0.22.1 autoimpute impyute fancyimpute
from fancyimpute import NuclearNormMinimization, SoftImpute, BiScaler, IterativeSVD, NuclearNormMinimization

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer

from sklearn.linear_model import LogisticRegression, RidgeClassifier

from autoimpute.imputations import SingleImputer, MultipleImputer

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import SimpleImputer, IterativeImputer

from sklearn.model_selection import StratifiedKFold

import time, os, warnings, random, string, re, gc

from sklearn.feature_selection import RFE, RFECV

from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import roc_auc_score

from IPython.display import display

from scipy.stats import rankdata

from autoimpute.visuals import *

import matplotlib.pyplot as plt

import category_encoders as ce

import plotly_express as px

import impyute as impy 

import seaborn as sns

import pandas as pd 

import scipy as sp

import numpy as np



sns.set_style('whitegrid')

warnings.filterwarnings('ignore')

warnings.simplefilter('ignore')

pd.set_option('display.max_columns', 1000)

pd.set_option('display.max_rows', 500)

SEED = 2020

SPLITS = 25
def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()
base =  pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

baseTe =  pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')

baseTe['target'] = -1

score = dict()
pd.DataFrame(base.isna().sum(axis=1).describe(), columns=['Value']).T
pd.DataFrame(base.isna().sum(axis=0)/len(base), columns=['missing percent']).sort_values('missing percent', ascending=False).T
plot_md_locations(base)
plot_nullility_dendogram(base)
def null_analysis(df):

  '''

  desc: get nulls for each column in counts & percentages

  arg: dataframe

  return: dataframe

  '''

  null_cnt = df.isnull().sum() # calculate null counts

  null_cnt = null_cnt[null_cnt!=0] # remove non-null cols

  null_percent = null_cnt / len(df) * 100 # calculate null percentages

  null_table = pd.concat([pd.DataFrame(null_cnt), pd.DataFrame(null_percent)], axis=1)

  null_table.columns = ['counts', 'percentage']

  null_table.sort_values('counts', ascending=False, inplace=True)

  return null_table
null_table = null_analysis(base)

px.bar(null_table.reset_index(), x='index', y='percentage', text='counts', height=500)
score = dict()
base.nom_1.unique()
def SiavashMapping(df):

    

    bin_3_mapping = {'T':1 , 'F':0}

    bin_4_mapping = {'Y':1 , 'N':0}

    nom_0_mapping = {'Red' : 0, 'Blue' : 1, 'Green' : 2}

    nom_1_mapping = {'Trapezoid' : 0, 'Star' : 1, 'Circle': 2, 'Triangle' : 3, 'Polygon' : 4, 'Square': 5}

    nom_2_mapping = {'Hamster' : 0 , 'Axolotl' : 1, 'Lion' : 2, 'Dog' : 3, 'Cat' : 4, 'Snake' : 5}

    nom_3_mapping = {'Russia' : 0, 'Canada' : 1, 'Finland' : 2, 'Costa Rica' : 3, 'China' : 4, 'India' : 5}

    nom_4_mapping = {'Bassoon' : 0, 'Theremin' : 1, 'Oboe' : 2, 'Piano' : 3}

    nom_5_mapping = dict(zip((df.nom_5.dropna().unique()), range(len((df.nom_5.dropna().unique())))))

    nom_6_mapping = dict(zip((df.nom_6.dropna().unique()), range(len((df.nom_6.dropna().unique())))))

    nom_7_mapping = dict(zip((df.nom_7.dropna().unique()), range(len((df.nom_7.dropna().unique())))))

    nom_8_mapping = dict(zip((df.nom_8.dropna().unique()), range(len((df.nom_8.dropna().unique())))))

    nom_9_mapping = dict(zip((df.nom_9.dropna().unique()), range(len((df.nom_9.dropna().unique())))))

    ord_1_mapping = {'Novice' : 0, 'Contributor' : 1, 'Expert' : 2, 'Master': 3, 'Grandmaster': 4}

    ord_2_mapping = { 'Freezing': 0, 'Cold': 1, 'Warm' : 2, 'Hot': 3, 'Boiling Hot' : 4, 'Lava Hot' : 5}

    ord_3_mapping = {'a':0, 'b':1, 'c':2 ,'d':3 ,'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 'j':9, 'k':10, 'l':11, 'm':12, 'n':13, 'o':14}

    ord_4_mapping = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'J':9, 'K':10,'L':11,'M':12,

                     'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,'Z':25}

    sorted_ord_5 = sorted(df.ord_5.dropna().unique())

    ord_5_mapping = dict(zip(sorted_ord_5, range(len(sorted_ord_5))))

    df['bin_3'] = df.loc[df.bin_3.notnull(), 'bin_3'].map(bin_3_mapping)

    df['bin_4'] = df.loc[df.bin_4.notnull(), 'bin_4'].map(bin_4_mapping)

    df['nom_0'] = df.loc[df.nom_0.notnull(), 'nom_0'].map(nom_0_mapping)

    df['nom_1'] = df.loc[df.nom_1.notnull(), 'nom_1'].map(nom_1_mapping)

    df['nom_2'] = df.loc[df.nom_2.notnull(), 'nom_2'].map(nom_2_mapping)

    df['nom_3'] = df.loc[df.nom_3.notnull(), 'nom_3'].map(nom_3_mapping)

    df['nom_4'] = df.loc[df.nom_4.notnull(), 'nom_4'].map(nom_4_mapping)

    df['nom_5'] = df.loc[df.nom_5.notnull(), 'nom_5'].map(nom_5_mapping)

    df['nom_6'] = df.loc[df.nom_6.notnull(), 'nom_6'].map(nom_6_mapping)

    df['nom_7'] = df.loc[df.nom_7.notnull(), 'nom_7'].map(nom_7_mapping)

    df['nom_8'] = df.loc[df.nom_8.notnull(), 'nom_8'].map(nom_8_mapping)

    df['nom_9'] = df.loc[df.nom_9.notnull(), 'nom_9'].map(nom_9_mapping)

    df['ord_1'] = df.loc[df.ord_1.notnull(), 'ord_1'].map(ord_1_mapping)

    df['ord_2'] = df.loc[df.ord_2.notnull(), 'ord_2'].map(ord_2_mapping)

    df['ord_3'] = df.loc[df.ord_3.notnull(), 'ord_3'].map(ord_3_mapping)

    df['ord_4'] = df.loc[df.ord_4.notnull(), 'ord_4'].map(ord_4_mapping)

    df['ord_5'] = df.loc[df.ord_5.notnull(), 'ord_5'].map(ord_5_mapping)

    

    return df
def AntMapping(df, ordinal):

    ord_maps = {

        'ord_0': {val: i for i, val in enumerate([1, 2, 3])},

        'ord_1': {

            val: i

            for i, val in enumerate(

                ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster']

            )

        },

        'ord_2': {

            val: i

            for i, val in enumerate(

                ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']

            )

        },

        **{col: {val: i for i, val in enumerate(sorted(df[col].dropna().unique()))} for col in ['ord_3', 'ord_4', 'ord_5', 'day', 'month']},

    }

    ord_cols = pd.concat([df[col].map(ord_map).fillna(max(ord_map.values())//2).astype('float32') for col, ord_map in ord_maps.items()], axis=1)

    ord_cols /= ord_cols.max()

    ord_sqr = 4*(ord_cols - 0.5)**2

    ord_cols_sqr = [feat+'_sqr' for feat in ordinal]

    df[ordinal] = ord_cols

    df[ord_cols_sqr] = ord_sqr

    return df
def CountEncoding(df, cols, df_test=None):

    for col in cols:

        frequencies = df[col].value_counts().reset_index()

        df_values = df[[col]].merge(frequencies, how='left', left_on=col, right_on='index').iloc[:,-1].values

        df[col+'_counts'] = df_values

        if df_test is not None:

            df_test_values = df_test[[col]].merge(frequencies, how='left', left_on=col, right_on='index').fillna(1).iloc[:,-1].values

            df_test[col+'_counts'] = df_test_values

    count_cols = [col+'_counts' for col in cols]

    if df_test is not None:

        return df, df_test, count_cols

    else:

        return df, count_cols

    

def YurgensenMapping(df):

    

    def TLC(s):

        s = str(s)

        return (((ord(s[0]))-64)*52+((ord(s[1]))-64)-6)

    

    df['ord_0_ord_2'] = df['ord_0'].astype('str')+df['ord_2'].astype('str')

    df['ord_0'] = df['ord_0'].fillna(2.01)

    df.loc[df.ord_2=='Freezing',    'ord_2'] = 0

    df.loc[df.ord_2=='Cold',        'ord_2'] = 1

    df.loc[df.ord_2=='Warm',        'ord_2'] = 2

    df.loc[df.ord_2=='Hot',         'ord_2'] = 3

    df.loc[df.ord_2=='Boiling Hot', 'ord_2'] = 4

    df.loc[df.ord_2=='Lava Hot',    'ord_2'] = 5

    df['ord_2'] = df['ord_2'].fillna(2.37)

    df.loc[df.ord_1=='Novice',      'ord_1'] = 0

    df.loc[df.ord_1=='Contributor', 'ord_1'] = 1

    df.loc[df.ord_1=='Expert',      'ord_1'] = 2

    df.loc[df.ord_1=='Master',      'ord_1'] = 3

    df.loc[df.ord_1=='Grandmaster', 'ord_1'] = 4

    df['ord_1'] = df['ord_1'].fillna(1.86)

    df['ord_5'] = df.loc[df.ord_5.notnull(), 'ord_5'].apply(lambda x: TLC(x))

    df['ord_5'] = df['ord_5'].fillna('Zx').apply(lambda x: TLC(x))

    df['ord_5'] = df['ord_5'].rank()

    df['ord_3'] = df.loc[df.ord_3.notnull(), 'ord_3'].apply(lambda x: ord(str(x))-96)

    df['ord_3'] = df['ord_3'].fillna(8.44)

    df['ord_4'] = df.loc[df.ord_4.notnull(), 'ord_4'].apply(lambda x: ord(str(x))-64)

    df['ord_4'] = df['ord_4'].fillna(14.31)

    

    return df

    
def RidgeClf(train, test, ordinal, ohe, scaler, seed, splits, drop_idx=None, dimreducer=None):

       

    y_train = train['target'].values.copy()

    train_length = train.shape[0]

    test['target'] = -1

    data = pd.concat([train, test], axis=0).reset_index(drop=True)

    X_ohe = pd.get_dummies(data[ohe],columns=ohe,drop_first=True,dummy_na=True,sparse=True, dtype='int8').sparse.to_coo()

    if dimreducer is not None:

        X_ohe = sp.sparse.csr_matrix(dimreducer.fit_transform(X_ohe))

        gc.collect()

    if ordinal is not None:

        if scaler is not None:

            X_ord = scaler.fit_transform(data[ordinal])

        else: 

            X_ord = data[ordianl].values

        data_ = sp.sparse.hstack([X_ohe, X_ord]).tocsr()

    else:

        data_ = sp.sparse.hstack([X_ohe]).tocsr()

    

    train = data_[:train_length]

    test = data_[train_length:]

    model = RidgeClassifier(alpha=152.5)

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

    oof = np.zeros((train.shape[0],))

    y_pred = np.zeros((test.shape[0],))



    for tr_ind, val_ind in skf.split(train, y_train):

        if drop_idx is not None:

            idx = list(set(tr_ind)-set(drop_idx))

            X_tr, X_val = train[idx],  train[val_ind]

            y_tr, y_val = y_train[idx], y_train[val_ind]

        else:

            X_tr, X_val = train[tr_ind],  train[val_ind]

            y_tr, y_val = y_train[tr_ind], y_train[val_ind]

        train_set = {'X':X_tr, 'y':y_tr}

        val_set = {'X':X_val, 'y':y_val}

        model.fit(train_set['X'], train_set['y'])

        oof[val_ind] = model.decision_function(val_set['X'])

        y_pred += model.decision_function(test) / splits

    oof_auc_score = roc_auc_score(y_train, oof)

    oof = rankdata(oof)/len(oof)

    y_pred = rankdata(y_pred)/len(y_pred)

    return oof, y_pred, oof_auc_score



def LogRegClf(train, test, ordinal, ohe, scaler, seed, splits, drop_idx=None, dimreducer=None):

    params = { 

        'fit_intercept' : True,

        'random_state': SEED,   

        'penalty' : 'l2',

        'verbose' : 0,   

        'solver' : 'lbfgs',     

        'max_iter' : 1000,

        'n_jobs' : 4,

        'C' : 0.05,

            }

    y_train = train['target'].values.copy()

    train_length = train.shape[0]

    test['target'] = -1

    data = pd.concat([train, test], axis=0).reset_index(drop=True)

    X_ohe = pd.get_dummies(data[ohe],columns=ohe,drop_first=True,dummy_na=True,sparse=True, dtype='int8').sparse.to_coo()

    if dimreducer is not None:

        X_ohe = sp.sparse.csr_matrix(dimreducer.fit_transform(X_ohe))

        gc.collect()

    if ordinal is not None:

        if scaler is not None:

            X_ord = scaler.fit_transform(data[ordinal])

        else:

            X_ord = data[ordinal].values

        data_ = sp.sparse.hstack([X_ohe, X_ord]).tocsr()

    else:

        data_ = sp.sparse.hstack([X_ohe]).tocsr()

    

    train = data_[:train_length]

    test = data_[train_length:]

    model = LogisticRegression(**params)

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

    oof = np.zeros((train.shape[0],))

    y_pred = np.zeros((test.shape[0],))



    for tr_ind, val_ind in skf.split(train, y_train):

        if drop_idx is not None:

            idx = list(set(tr_ind)-set(drop_idx))

            X_tr, X_val = train[idx],  train[val_ind]

            y_tr, y_val = y_train[idx], y_train[val_ind]

        else:

            X_tr, X_val = train[tr_ind],  train[val_ind]

            y_tr, y_val = y_train[tr_ind], y_train[val_ind]

        train_set = {'X':X_tr, 'y':y_tr}

        val_set = {'X':X_val, 'y':y_val}

        model.fit(train_set['X'], train_set['y'])

        oof[val_ind] = model.predict_proba(val_set['X'])[:, 1]

        y_pred += model.predict_proba(test)[:, 1] / splits

    oof_auc_score = roc_auc_score(y_train, oof)

    oof = rankdata(oof)/len(oof)

    y_pred = rankdata(y_pred)/len(y_pred)

    return oof, y_pred, oof_auc_score



train = base.copy()

test = baseTe.copy()

features = [feat for feat in train.columns if feat not in ['target','id']]

ohe = [feat for feat in features if feat not in ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']]

ordinal = [feat for feat in features if feat not in ohe]

train[features] = SiavashMapping(train[features])

test[features]  = SiavashMapping(test[features])

imp = SimpleImputer(strategy='constant')

train[features] = imp.fit_transform(train[features])

test[features]  = imp.transform(test[features])

train[features] = train[features].astype(np.int16)

test[features]  = test[features].astype(np.int16)

scaler = RobustScaler(quantile_range=(10.0, 90.0))

oof1, pred1, score['Constant'] = RidgeClf(train=train, test=test, ordinal=ordinal, ohe=ohe, scaler=scaler, seed=SEED, splits=SPLITS, drop_idx=None)

print(f'AUC score for Constant Imputation : {score["Constant"]}')
train = base.copy()

test  = baseTe.copy()

features =  [feat for feat in train.columns if feat not in ['target','id']]

ohe =       [feat for feat in features if feat not in ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']]

ordinal =   [feat for feat in features if feat not in ohe]

train[features] = SiavashMapping(train[features])

test[features]  = SiavashMapping(test[features])

imp =             SimpleImputer(strategy='constant')

train[ordinal] =  imp.fit_transform(train[ordinal])

test[ordinal]  =  imp.transform(test[ordinal])

train[ordinal] =  train[ordinal].astype(np.int16)

test[ordinal]  =  test[ordinal].astype(np.int16)

scaler =          RobustScaler(quantile_range=(10.0, 90.0))

oof2, pred2, score['Constant-OrdinalOnly'] = LogRegClf(train=train, test=test, ordinal=ordinal, ohe=ohe, scaler=scaler, seed=SEED, splits=SPLITS, drop_idx=None)

print(f'AUC score for Constant Imputation of ordinal columns: {score["Constant-OrdinalOnly"]}')
train = base.copy()

test  = baseTe.copy()

features =  [feat for feat in train.columns if feat not in ['target','id']]

ohe =       [feat for feat in features if feat not in ['ord_3', 'ord_4', 'ord_5']]

ordinal =   [feat for feat in features if feat not in ohe]

train[features] = SiavashMapping(train[features])

test[features]  = SiavashMapping(test[features])

imp =             SimpleImputer(strategy='constant')

train[ordinal] =  imp.fit_transform(train[ordinal])

test[ordinal]  =  imp.transform(test[ordinal])

train[ordinal] =  train[ordinal].astype(np.int16)

test[ordinal]  =  test[ordinal].astype(np.int16)

scaler =          RobustScaler(quantile_range=(10.0, 90.0))

oof3, pred3, score['Constant-Ord4Only'] = LogRegClf(train=train, test=test, ordinal=ordinal, ohe=ohe, scaler=scaler, seed=SEED, splits=SPLITS, drop_idx=None)

print(f'AUC score for Constant Imputation of ord_4 column: {score["Constant-Ord4Only"]}')
train = base.copy()

test  = baseTe.copy()

features =  [feat for feat in train.columns if feat not in ['target','id']]

ohe =  features

ordinal = None

train[features] = SiavashMapping(train[features])

test[features]  = SiavashMapping(test[features])

scaler =          RobustScaler(quantile_range=(10.0, 90.0))

oof4, pred4, score['CompleteOHE'] = LogRegClf(train=train, test=test, ordinal=ordinal, ohe=ohe, scaler=scaler, seed=SEED, splits=SPLITS, drop_idx=None)

print(f'AUC score for No imputation just OHE: {score["CompleteOHE"]}')
train = base.copy()

test  = baseTe.copy()

features =  [feat for feat in train.columns if feat not in ['target','id']]

ohe =  features

ordinal = None

train[features] = SiavashMapping(train[features])

test[features]  = SiavashMapping(test[features])

imp =             SimpleImputer(strategy='constant')

train[features] = imp.fit_transform(train[features])

test[features]  = imp.transform(test[features])

train[features] = train[features].astype(np.int16)

test[features]  = test[features].astype(np.int16)

scaler =          RobustScaler(quantile_range=(10.0, 90.0))

oof5, pred5, score['Constant-CompleteOHE'] = LogRegClf(train=train, test=test, ordinal=ordinal, ohe=ohe, scaler=scaler, seed=SEED, splits=SPLITS, drop_idx=None)

print(f'AUC score for Constant Imputation with complete OHE: {score["Constant-CompleteOHE"]}')
train = base.copy()

test  = baseTe.copy()

features =  [feat for feat in train.columns if feat not in ['target','id']]

ohe =  features

ordinal = None

train[features] = SiavashMapping(train[features])

test[features]  = SiavashMapping(test[features])

imp =             SingleImputer(strategy='mode')

train[features] = imp.fit_transform(train[features])

test[features]  = imp.transform(test[features])

train[features] = train[features].astype(np.int16)

test[features]  = test[features].astype(np.int16)

scaler =          RobustScaler(quantile_range=(10.0, 90.0))

oof6, pred6, score['Mode-CompleteOHE'] = LogRegClf(train=train, test=test, ordinal=ordinal, ohe=ohe, scaler=scaler, seed=SEED, splits=SPLITS, drop_idx=None)

print(f'AUC score for Mode Imputation with complete OHE: {score["Mode-CompleteOHE"]}')
train = base.copy()

test  = baseTe.copy()

features =  [feat for feat in train.columns if feat not in ['target','id']]

ohe =       [feat for feat in features if feat not in ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']]

ordinal =   [feat for feat in features if feat not in ohe]

train[features] = SiavashMapping(train[features])

test[features]  = SiavashMapping(test[features])

imp =             SingleImputer(strategy='mean')

train[ordinal] =  imp.fit_transform(train[ordinal])

test[ordinal]  =  imp.transform(test[ordinal])

train[ordinal] =  train[ordinal].astype(np.int16)

test[ordinal]  =  test[ordinal].astype(np.int16)

scaler =          RobustScaler(quantile_range=(10.0, 90.0))

oof7, pred7, score['Mean-OHE'] = LogRegClf(train=train, test=test, ordinal=ordinal, ohe=ohe, scaler=scaler, seed=SEED, splits=SPLITS, drop_idx=None)

print(f'AUC score for Mean Imputation with OHE on non-ordinal columns: {score["Mean-OHE"]}')
train = base.copy()

test  = baseTe.copy()

train = YurgensenMapping(train)

test  = YurgensenMapping(test)

features = [feat for feat in train.columns if feat not in ['target','id']]

ohe =      [feat for feat in features if feat not in ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']]

ordinal =  [feat for feat in features if feat not in ohe]

scaler  =  RobustScaler(quantile_range=(10.0, 90.0))

oof8, pred8, score['Yurgensen'] = LogRegClf(train=train, test=test, ordinal=ordinal, ohe=ohe, scaler=scaler, seed=SEED, splits=SPLITS, drop_idx=None)

print(f'AUC score for Yurgensen ordinal-Mapping : {score["Yurgensen"]}')
train = base.copy()

test  = baseTe.copy()

features = [feat for feat in train.columns if feat not in ['target','id']]

ohe =      [feat for feat in features if feat not in ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']]

ordinal =  [feat for feat in features if feat not in ohe] + ['day', 'month']

train  = AntMapping(train, ordinal)

test   = AntMapping(test , ordinal)

scaler = None

oof9, pred9, score['Ant'] = LogRegClf(train=train, test=test, ordinal=ordinal, ohe=ohe, scaler=scaler, seed=SEED, splits=SPLITS, drop_idx=None)

print(f'AUC score for Max Imputation of ordinals with OHE on non-ordinal columns: {score["Ant"]}')
train = base.copy()

test  = baseTe.copy()

features = [feat for feat in train.columns if feat not in ['target','id']]

ohe =      [feat for feat in features if feat not in ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']]

ordinal =  [feat for feat in features if feat not in ohe] + ['day', 'month']

train  = AntMapping(train, ordinal)

test   = AntMapping(test , ordinal)

train, test, count_cols = CountEncoding(train, ordinal, test)

ordinal += count_cols

scaler  = RobustScaler(quantile_range=(10.0, 90.0))

oof10, pred10, score['Ant-CE'] = LogRegClf(train=train, test=test, ordinal=ordinal, ohe=ohe, scaler=scaler, seed=SEED, splits=SPLITS, drop_idx=None)

print(f'AUC score for Max Imputation of ordinals with OHE on non-ordinal columns with CountEncoding of Ordinals: {score["Ant-CE"]}')
train = base.copy()

test  = baseTe.copy()

features =  [feat for feat in train.columns if feat not in ['target','id']]

ohe =       [feat for feat in features if feat not in ['ord_0', 'ord_1', 'ord_2']]

ordinal =   [feat for feat in features if feat not in ohe]

train[features] = SiavashMapping(train[features])

test[features]  = SiavashMapping(test[features])

imp =             SimpleImputer(strategy='median')

train[ordinal] =  imp.fit_transform(train[ordinal])

test[ordinal]  =  imp.transform(test[ordinal])

train[ordinal] =  train[ordinal].astype(np.int16)

test[ordinal]  =  test[ordinal].astype(np.int16)

scaler =          RobustScaler(quantile_range=(10.0, 90.0))

oof11, pred11, score['median-OrdPartial'] = LogRegClf(train=train, test=test, ordinal=ordinal, ohe=ohe, scaler=scaler, seed=SEED, splits=SPLITS, drop_idx=None)

print(f'AUC score for median Imputation of ord_0, ord_1 and ord_2 columns: {score["median-OrdPartial"]}')
train = base.copy()

test  = baseTe.copy()

features =  [feat for feat in train.columns if feat not in ['target','id']]

ohe =       [feat for feat in features if feat not in ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']]

ordinal =   [feat for feat in features if feat not in ohe]

train[features] = SiavashMapping(train[features])

test[features]  = SiavashMapping(test[features])

imp =             SingleImputer(strategy='norm')

train[ordinal] =  imp.fit_transform(train[ordinal])

test[ordinal]  =  imp.transform(test[ordinal])

train[ordinal] =  train[ordinal].astype(np.int16)

test[ordinal]  =  test[ordinal].astype(np.int16)

scaler =          RobustScaler(quantile_range=(10.0, 90.0))

oof12, pred12, score['Norm'] = LogRegClf(train=train, test=test, ordinal=ordinal, ohe=ohe, scaler=scaler, seed=SEED, splits=SPLITS, drop_idx=None)

print(f'AUC score for Norm Imputation of ordinals: {score["Norm"]}')
train = base.copy()

test  = baseTe.copy()

features =  [feat for feat in train.columns if feat not in ['target','id']]

ohe =       [feat for feat in features if feat not in ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']]

ordinal =   [feat for feat in features if feat not in ohe]

train[features] = SiavashMapping(train[features])

test[features]  = SiavashMapping(test[features])

imp =             SingleImputer(strategy='locf')

train[ordinal] =  imp.fit_transform(train[ordinal])

test[ordinal]  =  imp.transform(test[ordinal])

train[ordinal] =  train[ordinal].astype(np.int16)

test[ordinal]  =  test[ordinal].astype(np.int16)

scaler =          RobustScaler(quantile_range=(10.0, 90.0))

oof13, pred13, score['LOCF'] = LogRegClf(train=train, test=test, ordinal=ordinal, ohe=ohe, scaler=scaler, seed=SEED, splits=SPLITS, drop_idx=None)

print(f'AUC score for LOCF Imputation of ordinals: {score["LOCF"]}')
train = base.copy()

test  = baseTe.copy()

features =  [feat for feat in train.columns if feat not in ['target','id']]

ohe =       [feat for feat in features if feat not in ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']]

ordinal =   [feat for feat in features if feat not in ohe]

train[features] = SiavashMapping(train[features])

test[features]  = SiavashMapping(test[features])

imp =             SingleImputer(strategy='default univariate')

train[ordinal] =  imp.fit_transform(train[ordinal])

test[ordinal]  =  imp.transform(test[ordinal])

train[ordinal] =  train[ordinal].astype(np.int16)

test[ordinal]  =  test[ordinal].astype(np.int16)

scaler =          RobustScaler(quantile_range=(10.0, 90.0))

oof14, pred14, score['DefUnivariate'] = LogRegClf(train=train, test=test, ordinal=ordinal, ohe=ohe, scaler=scaler, seed=SEED, splits=SPLITS, drop_idx=None)

print(f'AUC score for Default Univariate Imputation: {score["DefUnivariate"]}')
train = base.copy()

test  = baseTe.copy()

features =  [feat for feat in train.columns if feat not in ['target','id']]

ohe = features

ordinal = None

train[features] = SiavashMapping(train[features])

test[features]  = SiavashMapping(test[features])

imp =             SingleImputer(strategy='interpolate')

train[features] = imp.fit_transform(train[features])

test[features]  = imp.transform(test[features])

train[features] = train[features].astype(np.int16)

test[features]  = test[features].astype(np.int16)

scaler =          RobustScaler(quantile_range=(10.0, 90.0))

oof15, pred15, score['Interpolate-OHE'] = LogRegClf(train=train, test=test, ordinal=ordinal, ohe=ohe, scaler=scaler, seed=SEED, splits=SPLITS, drop_idx=None)

print(f'Average AUC score for Interpolate Imputation with OHE om all columns: {score["Interpolate-OHE"]}')
train = base.copy()

test  = baseTe.copy()

features =  [feat for feat in train.columns if feat not in ['target','id']]

ohe = features

ordinal = None

train[features] = SiavashMapping(train[features])

test[features]  = SiavashMapping(test[features])

imp =             SoftImpute(max_iters=100, verbose=False)

train[features] = imp.fit_transform(train[features])

test[features]  = imp.fit_transform(test[features])

train[features] = train[features].astype(np.int16)

test[features]  = test[features].astype(np.int16)

scaler =          RobustScaler(quantile_range=(10.0, 90.0))

oof16, pred16, score['SoftImpute-OHE'] = LogRegClf(train=train, test=test, ordinal=ordinal, ohe=ohe, scaler=scaler, seed=SEED, splits=SPLITS, drop_idx=None)

print(f'AUC score for SoftImpute Imputation : {score["SoftImpute-OHE"]}')
train = base.copy()

test  = baseTe.copy()

features =  [feat for feat in train.columns if feat not in ['target','id']]

ohe =       [feat for feat in features if feat not in ['ord_0', 'ord_1', 'ord_2']]

ordinal =   [feat for feat in features if feat not in ohe]

train[features] = SiavashMapping(train[features])

test[features]  = SiavashMapping(test[features])

imp =             SingleImputer(strategy='least squares')

train[ordinal] =  imp.fit_transform(train[ordinal])

test[ordinal]  =  imp.transform(test[ordinal])

train[ordinal] =  train[ordinal].astype(np.int16)

test[ordinal]  =  test[ordinal].astype(np.int16)

scaler =          RobustScaler(quantile_range=(10.0, 90.0))

oof17, pred17, score['LQ-PartialOHE'] = LogRegClf(train=train, test=test, ordinal=ordinal, ohe=ohe, scaler=scaler, seed=SEED, splits=SPLITS, drop_idx=None)

print(f'AUC score for LQ Imputation with Partial OHE : {score["LQ-PartialOHE"]}')
train = base.copy()

test  = baseTe.copy()

features =  [feat for feat in train.columns if feat not in ['target','id']]

ohe =       [feat for feat in features if feat not in ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']]

ordinal =   [feat for feat in features if feat not in ohe]

train[features] = SiavashMapping(train[features])

test[features]  = SiavashMapping(test[features])

imp =             SingleImputer(strategy='stochastic')

train[ordinal] =  imp.fit_transform(train[ordinal])

test[ordinal]  =  imp.transform(test[ordinal])

train[ordinal] =  train[ordinal].astype(np.int16)

test[ordinal]  =  test[ordinal].astype(np.int16)

scaler =          RobustScaler(quantile_range=(10.0, 90.0))

oof18, pred18, score['stochastic'] = LogRegClf(train=train, test=test, ordinal=ordinal, ohe=ohe, scaler=scaler, seed=SEED, splits=SPLITS, drop_idx=None)

print(f'AUC score for stochastic Imputation : {score["stochastic"]}')
train = base.copy()

test  = baseTe.copy()

features =  [feat for feat in train.columns if feat not in ['target','id']]

ohe =       [feat for feat in features if feat not in ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']]

ordinal =   [feat for feat in features if feat not in ohe]

train[features] = SiavashMapping(train[features])

test[features]  = SiavashMapping(test[features])

imp =             IterativeImputer(max_iter=500, initial_strategy='most_frequent', random_state=SEED)

train[features] =  imp.fit_transform(train[features])

test[features]  =  imp.transform(test[features])

train[features] =  train[features].astype(np.int16)

test[features]  =  test[features].astype(np.int16)

scaler =          RobustScaler(quantile_range=(10.0, 90.0))

oof19, pred19, score['Iterative'] = LogRegClf(train=train, test=test, ordinal=ordinal, ohe=ohe, scaler=scaler, seed=SEED, splits=SPLITS, drop_idx=None)

print(f'AUC score for Iterative Imputation : {score["Iterative"]}')
train = base.copy()

test  = baseTe.copy()

drop_idx =  base[(base.isna().sum(axis=1)>3)].index.values

drop_idx =  [i for i in drop_idx]

features =  [feat for feat in train.columns if feat not in ['target','id']]

ohe =       [feat for feat in features if feat not in ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']]

ordinal =   [feat for feat in features if feat not in ohe]

train[features] = SiavashMapping(train[features])

test[features]  = SiavashMapping(test[features])

imp =             IterativeImputer(max_iter=500, initial_strategy='most_frequent', random_state=SEED)

train[features] =  imp.fit_transform(train[features])

test[features]  =  imp.transform(test[features])

train[features] =  train[features].astype(np.int16)

test[features]  =  test[features].astype(np.int16)

scaler =          RobustScaler(quantile_range=(10.0, 90.0))

oof20, pred20, score['IterativeWithDrop'] = LogRegClf(train=train, test=test, ordinal=ordinal, ohe=ohe, scaler=scaler, seed=SEED, splits=SPLITS, drop_idx=drop_idx)

print(f'AUC score for Iterative Imputation With threshold Drop : {score["IterativeWithDrop"]}')
train = base.copy()

test  = baseTe.copy()

drop_idx =  base[(base.isna().sum(axis=1)>3)].index.values

drop_idx =  [i for i in drop_idx]

features =  [feat for feat in train.columns if feat not in ['target','id']]

ohe =       [feat for feat in features if feat not in ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']]

ordinal =   [feat for feat in features if feat not in ohe]

train[features] = SiavashMapping(train[features])

test[features]  = SiavashMapping(test[features])

imp = IterativeImputer(max_iter=500, initial_strategy='most_frequent', random_state=SEED, add_indicator=True)

indicator_cols = [feat+'_ind' for feat in ordinal]

for col in indicator_cols:

    train[col] = 0

    test[col]  = 0

    train[col] = train[col].astype(np.uint8)

    test[col]  = test[col].astype(np.uint8)



train[ordinal+indicator_cols] =  imp.fit_transform(train[ordinal])

test[ordinal+indicator_cols]  =  imp.transform(test[ordinal])

train[ordinal] =  train[ordinal].astype(np.int16)

test[ordinal]  =  test[ordinal].astype(np.int16)

scaler =           RobustScaler(quantile_range=(10.0, 90.0))

ohe   += indicator_cols

oof21, pred21, score['IterativeWithIndicator'] = LogRegClf(train=train, test=test, ordinal=ordinal, ohe=ohe, scaler=scaler, seed=SEED, splits=SPLITS, drop_idx=drop_idx)

print(f'AUC score for Iterative Imputation With Indicator : {score["IterativeWithIndicator"]}')
train = base.copy()

test  = baseTe.copy()

train['nulls'] = train.isna().sum(axis=1)

test['nulls']  = test.isna().sum(axis=1)

features =  [feat for feat in train.columns if feat not in ['target','id']]

ohe =       [feat for feat in features if feat not in ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'nulls']]

ordinal =   [feat for feat in features if feat not in ohe]

train[features] = SiavashMapping(train[features])

test[features]  = SiavashMapping(test[features])

imp =             IterativeImputer(max_iter=500, initial_strategy='most_frequent', random_state=SEED)

train[ordinal] =  imp.fit_transform(train[ordinal])

test[ordinal]  =  imp.transform(test[ordinal])

train[ordinal] =  train[ordinal].astype(np.int16)

test[ordinal]  =  test[ordinal].astype(np.int16)

scaler =          RobustScaler(quantile_range=(10.0, 90.0))

oof22, pred22, score['IterativeWithSum'] = LogRegClf(train=train, test=test, ordinal=ordinal, ohe=ohe, scaler=scaler, seed=SEED, splits=SPLITS, drop_idx=None)

print(f'AUC score for Iterative Imputation with sum of missing values: {score["IterativeWithSum"]}')
scores = pd.DataFrame(score, index=['OOF-AUC']).T.sort_values(by='OOF-AUC', ascending=False)
scores
ax = scores.plot(kind='barh', title ='ROC-AUC Score For Different Imputation Techniques', figsize=(15, 10), legend=True, fontsize=12, alpha = 0.85, cmap = 'gist_gray')

ax.set_xlim((0.7,0.8))

plt.show()

def StackModels():

    

    idx = baseTe.id.values

    train_oofs = [oof1, oof2, oof3, oof4, oof5, oof6, oof7, oof8, oof9, oof10, oof11, oof12, oof13, oof14, oof15, oof16, oof17, oof18, oof19]

    test_oofs  = [pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8, pred9, pred10, pred11, pred12, pred13, pred14, pred15, pred16, pred17, pred18, pred19]

    X_train = pd.concat([pd.DataFrame(file) for file in train_oofs], axis=1)

    X_test = pd.concat([pd.DataFrame(file) for file in test_oofs], axis=1)

    X_train.columns = ['y_' + str(i) for i in range(len(train_oofs))]

    X_test.columns = ['y_' + str(i) for i in range(len(train_oofs))]

    X_train = pd.concat([X_train, base[['target']]], axis=1)

    X_test = pd.concat([X_test, baseTe[['target']]], axis=1)

    for i, (oof, pred) in enumerate(zip(train_oofs, test_oofs)):

        train_oofs[i] = rankdata(oof)/len(oof)

        test_oofs[i] = rankdata(pred)/len(pred)

    for f in X_train.columns:

        X_train[f] = X_train[f].astype('float32')

        X_test[f] = X_test[f].astype('float32')  

    features = np.array([f for f in X_train.columns if f not in ['target']])

    target = ['target']

    oof_pred_final = np.zeros((len(base), ))

    y_pred_final = np.zeros((len(baseTe),))

    skf = StratifiedKFold(n_splits=SPLITS, shuffle=True, random_state=SEED)

    model = RidgeClassifier()

    selector = RFECV(model, step=1, cv = skf, scoring='roc_auc', verbose=0, n_jobs=4)

    selector.fit(X_train[features], X_train[target])

    selected_features = [i for i, y in enumerate(selector.ranking_) if y == 1]

    selected_features = features[selected_features]

    for fold, (tr_ind, val_ind) in enumerate(skf.split(X_train, X_train[target])):

        x_tr, x_val = X_train[selected_features].iloc[tr_ind], X_train[selected_features].iloc[val_ind]

        y_tr, y_val = X_train[target].iloc[tr_ind], X_train[target].iloc[val_ind]

        train_set = {'X':x_tr, 'y':y_tr}

        val_set = {'X':x_val, 'y':y_val}

        model = RidgeClassifier()

        model.fit(train_set['X'],train_set['y'])

        fold_pred = model.decision_function(val_set['X'])

        oof_pred_final[val_ind] = fold_pred

        y_pred_final += model.decision_function(X_test[selected_features]) / (SPLITS)

    oof_auc_score = roc_auc_score(base[target], oof_pred_final)

    print(f'OOF Stack ROC-AUC Score is: {oof_auc_score:.7f}')

    y_pred_final = rankdata(y_pred_final)/len(y_pred_final)

    np.save('oof_pred_final.npy',oof_pred_final)

    np.save('y_pred_final.npy', y_pred_final)

    print('*'* 36)

    print('        OOF files saved!')

    print('*'* 36)

    submission = pd.DataFrame.from_dict({

        'id': idx,

        'target': y_pred_final

        })

    submission.to_csv('submission.csv', index=False)

    print('*'* 36)

    print('     Submission file saved!')

    print('*'* 36)

    

    return

StackModels()