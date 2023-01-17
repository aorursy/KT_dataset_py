# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



# Get version python/keras/tensorflow/sklearn

from platform import python_version

import sklearn



# Folder manipulation

import os



# Garbage collector

import gc



# Linear algebra and data processing

import numpy as np

import pandas as pd

from pandas import datetime



# Visualisation of picture and graph

import matplotlib.pyplot as plt

import seaborn as sns 



# Sklearn importation

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate, KFold

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, MinMaxScaler

from sklearn.metrics import roc_auc_score, classification_report, roc_curve

from sklearn.base import clone

from sklearn import base
print(os.listdir("../input"))

print("Python version : " + python_version())

print("Sklearn version : " + sklearn.__version__)
MAIN_DIR = "../input/cat-in-the-dat/"



TRAIN_DIR = MAIN_DIR + "train.csv"

TEST_DIR = MAIN_DIR + "test.csv"



BINS_FEAT = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']

NAMES_FEAT = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

ORDINALS_FEAT = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']

OTHERS_FEAT = ['day', 'month']



# Set graph font size

sns.set(font_scale=1.2)
def load_data():

    df_train = pd.read_csv(TRAIN_DIR)

    df_test = pd.read_csv(TEST_DIR)

    return df_train, df_test
data_train_raw, data_test_raw = load_data()
print(f"Shape training data : {data_train_raw.shape}")

print(f"Shape test data : {data_test_raw.shape}")
data_train_raw.head()
data_train_raw.isna().sum()
data_train_raw.info()
# Reduce memory used on RAM

def pre_processing(data_train, data_test):

    df_train = data_train.copy()

    df_test = data_test.copy()

    

    dtypes = {'bin_0': 'int8',

              'bin_1': 'int8',

              'bin_2': 'int8',

              'ord_0': 'int8',

              'day': 'int8',

              'month': 'int8'}

    

    df_train = df_train.astype(dtypes)

    df_train['target'] = df_train['target'].astype('int8')

    

    df_test = df_test.astype(dtypes)

    

    return df_train, df_test
data_train_pre, data_test_pre = pre_processing(data_train_raw, data_test_raw)
def count_plot(data_train, data_test,

               feats,

               title="",

               n_cols=3,

               figsize=(20, 10)):

    

    df_train = data_train.copy()

    df_test = data_test.copy()

    

    df_train['dataset'] = 'train'

    df_test['dataset'] = 'test'

    

    df_train_test = pd.concat([df_train, df_test], axis=0)

    

    n_axes = int(np.ceil(len(feats)/n_cols))

    n_axes_last = len(feats)%3

    

    if(n_axes > 1):

        fig, axes = plt.subplots(n_axes, n_cols, figsize=figsize)

        # Delete useless ax

        for ax in axes[-1,n_axes_last:]:

            fig.delaxes(ax)

    else:

        fig, axes = plt.subplots(n_axes, len(feats), figsize=figsize)

    

    for ax, feat in zip(axes.ravel()[0:len(feats)], feats):

        sns.countplot(x=feat, hue="dataset", data=df_train_test, ax=ax)

        

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=None)

    fig.suptitle(title, fontsize=20)
count_plot(data_train_raw, data_test_raw, 

           feats=BINS_FEAT, 

           title="Binary features")
data_train_pre[NAMES_FEAT].head()
count_plot(data_train_pre, data_test_pre, 

           feats=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'],

           title="Names features",

           figsize=(20, 10))
data_train_pre[ORDINALS_FEAT].head()
count_plot(data_train_pre, data_test_pre, 

           feats=['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4'],

           title="Ordinals features",

           figsize=(20, 10))
count_plot(data_train_pre, data_test_pre, 

           feats=OTHERS_FEAT,

           title="Others features",

           n_cols=2,

           figsize=(13, 4))
fig, ax = plt.subplots(1, 1, figsize=(5, 4))

sns.countplot(x='target', data=data_train_pre, ax=ax)

fig.suptitle('Target feature', fontsize=20)
class KFoldTargetEncoderTrain(base.BaseEstimator,

                               base.TransformerMixin):

    def __init__(self,colnames,targetName,

                  n_fold=5, verbosity=True,

                  discardOriginal_col=False):

        self.colnames = colnames

        self.targetName = targetName

        self.n_fold = n_fold

        self.verbosity = verbosity

        self.discardOriginal_col = discardOriginal_col

        

    def fit(self, X, y=None):

        return self

    

    def transform(self,X):

        assert(type(self.targetName) == str)

        assert(type(self.colnames) == str)

        assert(self.colnames in X.columns)

        assert(self.targetName in X.columns)

        mean_of_target = X[self.targetName].mean()

        kf = KFold(n_splits = self.n_fold,

                   shuffle = False, random_state=2019)

        col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'

        X[col_mean_name] = np.nan

        for tr_ind, val_ind in kf.split(X):

            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]

            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(X_tr.groupby(self.colnames)

                                     [self.targetName].mean())

            X[col_mean_name].fillna(mean_of_target, inplace = True)

        if self.verbosity:

            encoded_feature = X[col_mean_name].values

            print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,self.targetName,                    

                   np.corrcoef(X[self.targetName].values,

                               encoded_feature)[0][1]))

        if self.discardOriginal_col:

            X = X.drop(self.targetName, axis=1)

        return X

    

    

class KFoldTargetEncoderTest(base.BaseEstimator,

                             base.TransformerMixin):

    def __init__(self,train,colNames,encodedName):

        

        self.train = train

        self.colNames = colNames

        self.encodedName = encodedName

        

    def fit(self, X, y=None):

        return self

    

    def transform(self,X):

        mean =  self.train[[self.colNames,

                self.encodedName]].groupby(

                                self.colNames).mean().reset_index() 

        

        dd = {}

        for index, row in mean.iterrows():

            dd[row[self.colNames]] = row[self.encodedName]

        X[self.encodedName] = X[self.colNames]

        X = X.replace({self.encodedName: dd})

        return X
def feature_engineering(data_train, data_test):

    df_train = data_train.copy()

    df_test = data_test.copy()

    

    df_traintest = pd.concat([df_train, df_test])

    

    print("# BINS FEATURES")

    

    print("\t# bin_3")

    df_traintest.loc[df_train['bin_3'] == 'T', 'bin_3'] = 1

    df_traintest.loc[df_train['bin_3'] == 'F', 'bin_3'] = 0

    

    print("\t# bin_4")

    df_traintest.loc[df_train['bin_4'] == 'Y', 'bin_4'] = 1

    df_traintest.loc[df_train['bin_4'] == 'N', 'bin_4'] = 0

    

    print("# ORDINALS FEATURES")

    

    print("\t# Label encoding")

    

    ord_1_map = {

        'Grandmaster':'4',

        'Master':'3',

        'Expert':'2',

        'Contributor':'1',

        'Novice':'0'

    }

    df_traintest['ord_1'] = df_traintest['ord_1'].map(ord_1_map)

    

    ord_2_map = {

        'Lava Hot': '5',

        'Boiling Hot':'4',

        'Hot':'3',

        'Warm':'2',

        'Cold':'1',

        'Freezing':'0'

    }

    df_traintest['ord_2'] = df_traintest['ord_2'].map(ord_2_map)

    

    df_traintest['ord_5a'] = df_traintest['ord_5'].apply(lambda x : list(x)[0])

    df_traintest['ord_5b'] = df_traintest['ord_5'].apply(lambda x : list(x)[1])

    df_traintest['ord_5'] = df_traintest.drop(['ord_5'], axis=1)

    

    for feat in ['ord_3', 'ord_4', 'ord_5a', 'ord_5b']:

        dict_ord = dict()

        values = df_traintest[feat].apply(lambda x : list(x)[0]).value_counts().index.sort_values().values



        for value, label in zip(values, range(values.shape[0])):

            dict_ord[value] = label

            

        df_traintest[feat] = df_traintest[feat].map(dict_ord)

        

    feat_to_encode = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5a', 'ord_5b']

    df_traintest[feat_to_encode] = MinMaxScaler().fit_transform(df_traintest[feat_to_encode])

    

    print("\t# Target encoding")



    # Split data for encoding

    df_train_enc = df_traintest.iloc[:df_train.shape[0]]

    df_test_enc = df_traintest.iloc[df_train.shape[0]:]

    

    for feat in feat_to_encode:

        print(f"\t\t# {feat}")

        

        # For train

        targetc = KFoldTargetEncoderTrain(feat,'target',n_fold=5)

        df_train_enc = targetc.fit_transform(df_train_enc)

        

        # For test

        test_targetc = KFoldTargetEncoderTest(df_train_enc,

                                       feat,

                                       feat+'_Kfold_Target_Enc')

        df_test_enc = test_targetc.fit_transform(df_test_enc)

        

    df_traintest = pd.concat([df_train_enc, df_test_enc], axis=0)

    

    print("# NAMESS FEATS and OTHERS_FEAT")



    df_traintest = pd.get_dummies(df_traintest,

                                  columns=NAMES_FEAT+OTHERS_FEAT,

                                  sparse=True,

                                  drop_first=True)

    

    df_train = df_traintest.iloc[:df_train.shape[0], :]

    df_test = df_traintest.iloc[df_train.shape[0]:, :]

    

    df_test = df_test.drop(['target'], axis=1)

    

    print("# DROP FEATURES")

    

    df_traintest = df_traintest.drop(['bin_0', 'ord_5b'], axis=1)

    

    gc.collect()

    

    return df_train, df_test
%%time

data_train_raw, data_test_raw = load_data()

data_train_pre, data_test_pre = pre_processing(data_train_raw, data_test_raw)

data_train, data_test = feature_engineering(data_train_pre, data_test_pre)
print(f"Shape training data : {data_train.shape}")

print(f"Shape test data : {data_test.shape}")
def get_matrices(data_train, data_test):

    X = data_train.drop(['target', 'id'], axis=1)

    X_test = data_test.drop(['id'], axis=1)

    

    # Transform to sparse matrice

    X = X.astype(float).to_sparse().to_coo().tocsr()

    X_test = X_test.astype(float).to_sparse().to_coo().tocsr()

    

    y = data_train['target']

    return X, y, X_test
def plot_roc_auc_folds(y_true_folds, y_pred_folds, n_cols=3, figsize=(15, 10), title=""):

    

    def plot_roc_auc(y_true, y_pred, ax, i):

        fpr, tpr, thresholds = roc_curve(y_true, y_pred,

                                         drop_intermediate=False)

        auc_score = roc_auc_score(y_true, y_pred)

        

        ax.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

        ax.plot([0, 1], [0, 1], 'k--')



        ax.set_xlim([0.0, 1.0])

        ax.set_ylim([0.0, 1.05])

        ax.set_xlabel('FPR or [1 - TPR]')

        ax.set_ylabel('TPR')

        ax.set_title(f'ROC fold {i}')

        ax.legend(loc="lower right")

        return ax

    

    n_axes = int(np.ceil(len(y_true_folds)/n_cols))

    n_axes_last = len(y_true_folds)%3

    

    if(n_axes > 1):

        fig, axes = plt.subplots(n_axes, n_cols, figsize=figsize)

    else:

        fig, axes = plt.subplots(n_axes, len(y_true_folds), figsize=figsize)

    

    for y_true, y_pred, ax, fold in zip(y_true_folds, 

                                        y_pred_folds,

                                        axes.ravel()[0:len(y_true_folds)],

                                        range(len(y_true_folds))):

        plot_roc_auc(y_true, y_pred, ax=ax, i=fold)

        

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.3)

    fig.suptitle(title, fontsize=20)
def print_classification_report_folds(y_true_folds, y_pred_folds):

    print("Detailed classification report:\n")

    for y_true, y_pred, fold in zip(y_true_folds, 

                              y_pred_folds, 

                              range(len(y_true_folds))):

        print(f"FOLDS : {fold}")

        print(classification_report(y_true=y_true, y_pred=y_pred))
def train_eval(X, y, n_folds=6):

    

    kf = KFold(n_splits=n_folds, shuffle=True)

    print(f"Numer of folds is {kf.get_n_splits(X)}")

    

    y_true_folds = []

    y_pred_folds = []

    

    fold = kf.get_n_splits(X)

    

    for train_index, test_index in kf.split(X):

        print(f"FOLD : {fold}")

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

    

        model = LogisticRegression(n_jobs=-1, solver='lbfgs', max_iter=3000, C=0.1)

        model.fit(X_train, y_train)

        

        y_true, y_pred = y_test.values, model.predict(X_test)

        

        y_true_folds.append(y_true)

        y_pred_folds.append(y_pred)

        

        fold -= 1

    

    return model, y_true_folds, y_pred_folds
%%time

X, y, X_test = get_matrices(data_train, data_test)

model, y_true_folds, y_pred_folds = train_eval(X, y)
plot_roc_auc_folds(y_true_folds, y_pred_folds, title="ROC and ROC-AUC for each folds")
print_classification_report_folds(y_true_folds, y_pred_folds)
%%time

# Retrain the model on all the data

model = clone(model, safe=True)

model.fit(X, y)
def plot_pred_ratio(y_pred):

    df = pd.DataFrame(y_pred, columns=['y_pred'])

    

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    sns.countplot(df['y_pred'], ax=ax)

    ax.set_title("Prediction repartition")
y_pred = model.predict(X_test)

plot_pred_ratio(y_pred)
def submission(y_pred):

    df_test = pd.read_csv(TEST_DIR)

    sub = pd.DataFrame({"id": df_test["id"], "target": y_pred})

    sub.to_csv("submission.csv", index=False)
y_pred = model.predict_proba(X_test)[:,1]

submission(y_pred)