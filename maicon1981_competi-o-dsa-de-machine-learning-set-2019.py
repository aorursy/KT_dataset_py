# Import libraries to store data

import pandas as pd

import numpy as np



# Import libraries to visualize data

import matplotlib.pyplot as plt

import seaborn as sns



# Import libraries to process data

from tsfresh import extract_relevant_features, extract_features, select_features

from tsfresh.utilities.dataframe_functions import impute

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import StratifiedKFold



# Import libraries to classify data and score results

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from mlxtend.classifier import StackingCVClassifier

from sklearn.metrics import confusion_matrix



# Import libraries used in functions and for feedback

import os

import gc

import logging

import itertools

import warnings
# Settings

path = os.getcwd()

gc.enable()

%matplotlib inline

warnings.filterwarnings("ignore")
# Kaggle kernel: IS_LOCAL = False

IS_LOCAL = False

if(IS_LOCAL):

    PATH='../input/'

else:

    PATH='../input/competicao-dsa-machine-learning-sep-2019/'
print(os.listdir(PATH))
# Logger

def get_logger():

    FORMAT = '[%(levelname)s] %(asctime)s: %(name)s: %(message)s'

    logging.basicConfig(format=FORMAT)

    logger = logging.getLogger('Main')

    logger.setLevel(logging.DEBUG)

    return logger



logger = get_logger()
logger.info('Start load data')
# Read in data into a dataframe

X_train = pd.read_csv(os.path.join(PATH, 'X_treino.csv'))

y_train = pd.read_csv(os.path.join(PATH, 'y_treino.csv'))

X_test = pd.read_csv(os.path.join(PATH, 'X_teste.csv'))
logger.info('Start exploratory data analysis')
# Show dataframe columns

print(X_train.columns)
# Display top of dataframe

X_train.head()
# Display the shape of dataframe

X_train.shape
# See the column data types and non-missing values

X_train.info()
# Unique values by features

X_train.nunique(dropna=False, axis=0)
# Missing values by features

X_train.isnull().sum(axis=0)
# Statistics of numerical features

X_train.describe().T
# Boxplots for each column

X_train.plot(kind='box', subplots=True, layout=(4,3), figsize=(14,10))
# Show dataframe columns

print(X_test.columns)
# Display top of dataframe

X_test.head()
# Display the shape of dataframe

X_test.shape
# See the column data types and non-missing values

X_test.info()
# Unique values by features

X_test.nunique(dropna=False, axis=0)
# Missing values by features

X_test.isnull().sum(axis=0)
# Statistics of numerical features

X_test.describe().T
# Boxplots for each column

X_test.plot(kind='box', subplots=True, layout=(4,3), figsize=(14,10))
# Show dataframe columns

print(y_train.columns)
# Display top of dataframe

y_train.head()
# Display the shape of dataframe

y_train.shape
# See the column data types and non-missing values

y_train.info()
# Unique values by features

y_train.nunique(dropna=False, axis=0)
# Missing values by features

y_train.isnull().sum(axis=0)
# Statistics of numerical features

y_train.describe().T
# Boxplots for each column

y_train.plot(kind='box', subplots=True, layout=(1,2), figsize=(5,4))
# Levels distribution of categorical

y_train.groupby('surface').size().sort_values(ascending=False)
# Distribution of target feature (surface)

f, ax = plt.subplots(1,1, figsize=(16,4))

total = float(len(y_train))

g = sns.countplot(y_train['surface'], order = y_train['surface'].value_counts().index, color='steelblue')

g.set_title("Number and percentage of surface")

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(100*height/total),

            ha="center") 

plt.show()
# Distribution of group_id

f, ax = plt.subplots(1,1, figsize=(18,8))

total = float(len(y_train))

g = sns.countplot(y_train['group_id'], order = y_train['group_id'].value_counts().index, color='steelblue')

g.set_title("Number and percentage of group_id")

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.1f}%'.format(100*height/total),

            ha="center", rotation='90') 

plt.show()
# Density plots of features

def plot_feature_distribution(df1, df2, label1, label2, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(2, 5 ,figsize=(16,8))



    for feature in features:

        i += 1

        plt.subplot(2, 5, i)

        sns.kdeplot(df1[feature], bw=0.5, label=label1)

        sns.kdeplot(df2[feature], bw=0.5, label=label2)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=8)

        plt.tick_params(axis='y', which='major', labelsize=8)

    plt.show()
features = X_train.columns.values[3:]

plot_feature_distribution(X_train, X_test, 'train', 'test', features)
def plot_feature_class_distribution(classes, tt, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(5, 2, figsize=(16,24))



    for feature in features:

        i += 1

        plt.subplot(5, 2, i)

        for clas in classes:

            ttc = tt[tt['surface']==clas]

            sns.kdeplot(ttc[feature], bw=0.5, label=clas)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=8)

        plt.tick_params(axis='y', which='major', labelsize=8)

    plt.show()
classes = (y_train['surface'].value_counts()).index

tt = X_train.merge(y_train, on='series_id', how='inner')

plot_feature_class_distribution(classes, tt, features)
# Target feature - surface and group_id distribution

fig, ax = plt.subplots(1,1,figsize=(24,6))

tmp = pd.DataFrame(y_train.groupby(['group_id', 'surface'])['series_id'].count().reset_index())

m = tmp.pivot(index='surface', columns='group_id', values='series_id')

s = sns.heatmap(m, linewidths=.1, linecolor='black', annot=True, cmap="YlGnBu")

s.set_title('Number of surface category per group_id', size=16)

plt.show()
# Correlation map for train dataset

corr = X_train.corr()

_ , ax = plt.subplots( figsize =( 12 , 10 ) )

cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = {'fontsize' : 12 })
# Correlation map for test dataset

corr = X_test.corr()

_ , ax = plt.subplots( figsize =( 12 , 10 ) )

cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = {'fontsize' : 12 })
logger.info('Start feature engineering')
logger.info('Start processing missing values')
# Function to calculate missing values by column

def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
missing_values_table(X_train)
missing_values_table(X_test)
missing_values_table(y_train)
logger.info('Start Euler factors and additional features')
# https://stackoverflow.com/questions/53033620/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr?rq=1

def quaternion_to_euler(x, y, z, w):

    import math

    t0 = +2.0 * (w * x + y * z)

    t1 = +1.0 - 2.0 * (x * x + y * y)

    X = math.atan2(t0, t1)



    t2 = +2.0 * (w * y - z * x)

    t2 = +1.0 if t2 > +1.0 else t2

    t2 = -1.0 if t2 < -1.0 else t2

    Y = math.asin(t2)



    t3 = +2.0 * (w * z + x * y)

    t4 = +1.0 - 2.0 * (y * y + z * z)

    Z = math.atan2(t3, t4)



    return X, Y, Z
# Calculate euler factors and several additional features starting from the original features

def perform_euler_factors_calculation(df):

    df['total_angular_velocity'] = np.sqrt(np.square(df['angular_velocity_X']) + np.square(df['angular_velocity_Y']) + np.square(df['angular_velocity_Z']))

    df['total_linear_acceleration'] = np.sqrt(np.square(df['linear_acceleration_X']) + np.square(df['linear_acceleration_Y']) + np.square(df['linear_acceleration_Z']))

    df['total_xyz'] = np.sqrt(np.square(df['orientation_X']) + np.square(df['orientation_Y']) +

                              np.square(df['orientation_Z']))

    df['acc_vs_vel'] = df['total_linear_acceleration'] / df['total_angular_velocity']

    

    x, y, z, w = df['orientation_X'].tolist(), df['orientation_Y'].tolist(), df['orientation_Z'].tolist(), df['orientation_W'].tolist()

    nx, ny, nz = [], [], []

    for i in range(len(x)):

        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])

        nx.append(xx)

        ny.append(yy)

        nz.append(zz)

    

    df['euler_x'] = nx

    df['euler_y'] = ny

    df['euler_z'] = nz

    

    df['total_angle'] = np.sqrt(np.square(df['euler_x']) + np.square(df['euler_y']) + np.square(df['euler_z']))

    df['angle_vs_acc'] = df['total_angle'] / df['total_linear_acceleration']

    df['angle_vs_vel'] = df['total_angle'] / df['total_angular_velocity']

    return df
X_train = perform_euler_factors_calculation(X_train)
X_test = perform_euler_factors_calculation(X_test)
X_train.shape
X_test.shape
features = X_train.columns.values[13:]

plot_feature_distribution(X_train, X_test, 'train', 'test', features)
classes = (y_train['surface'].value_counts()).index

tt = X_train.merge(y_train, on='series_id', how='inner')

plot_feature_class_distribution(classes, tt, features)
logger.info('Start aggregated feature extraction')
orientations = ['orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W']
angular_velocity = ['angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z']
linear_acceleration = ['linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z']
params = {'abs_energy':None,

          'absolute_sum_of_changes':None,

          'agg_autocorrelation':[{'f_agg':'var','maxlag':32}],

          'change_quantiles':[{'ql':0.25,'qh':0.75,'isabs':True, 'f_agg':'mean'},

                             {'ql':0.25,'qh':0.75,'isabs':True, 'f_agg':'std'}],

          'cid_ce':[{'normalize':True},{'normalize':False}],

          'fft_aggregated':[{'aggtype': 'centroid'},

                            {'aggtype': 'variance'},

                            {'aggtype': 'skew'},

                            {'aggtype': 'kurtosis'}],

          'c3': [{'lag': 1}, {'lag': 2}, {'lag': 3}],

          'standard_deviation': None,

          'variance': None,

          'skewness': None,

          'kurtosis': None,

          'maximum': None,

          'minimum': None,

          'sample_entropy':None,

          'mean_abs_change':None,

          'sum_values':None,

          'quantile': [{'q': 0.1},

                       {'q': 0.2},

                       {'q': 0.3},

                       {'q': 0.4},

                       {'q': 0.6},

                       {'q': 0.7},

                       {'q': 0.8},

                       {'q': 0.9}],

          'large_standard_deviation': [{'r': 0.25},{'r':0.35}],

          'fft_coefficient': [{'coeff': 0, 'attr': 'real'},

                              {'coeff': 1, 'attr': 'real'},

                              {'coeff': 2, 'attr': 'real'},

                              {'coeff': 3, 'attr': 'real'},

                              {'coeff': 4, 'attr': 'real'},

                              {'coeff': 5, 'attr': 'real'},

                              {'coeff': 6, 'attr': 'real'},

                              {'coeff': 7, 'attr': 'real'},

                              {'coeff': 8, 'attr': 'real'},

                              {'coeff': 9, 'attr': 'real'},

                              {'coeff': 10, 'attr': 'real'},

                              {'coeff': 11, 'attr': 'real'},

                              {'coeff': 12, 'attr': 'real'},

                              {'coeff': 13, 'attr': 'real'},

                              {'coeff': 14, 'attr': 'real'},

                              {'coeff': 15, 'attr': 'real'},

                              {'coeff': 16, 'attr': 'real'},

                              {'coeff': 17, 'attr': 'real'},

                              {'coeff': 18, 'attr': 'real'},

                              {'coeff': 19, 'attr': 'real'},

                              {'coeff': 20, 'attr': 'real'},

                              {'coeff': 21, 'attr': 'real'},

                              {'coeff': 22, 'attr': 'real'},

                              {'coeff': 23, 'attr': 'real'},

                              {'coeff': 24, 'attr': 'real'},

                              {'coeff': 25, 'attr': 'real'},

                              {'coeff': 26, 'attr': 'real'},

                              {'coeff': 27, 'attr': 'real'},

                              {'coeff': 28, 'attr': 'real'},

                              {'coeff': 29, 'attr': 'real'},

                              {'coeff': 30, 'attr': 'real'},

                              {'coeff': 31, 'attr': 'real'},

                              {'coeff': 32, 'attr': 'real'},

                              {'coeff': 33, 'attr': 'real'},

                              {'coeff': 34, 'attr': 'real'},

                              {'coeff': 35, 'attr': 'real'},

                              {'coeff': 36, 'attr': 'real'},

                              {'coeff': 37, 'attr': 'real'},

                              {'coeff': 38, 'attr': 'real'},

                              {'coeff': 39, 'attr': 'real'},

                              {'coeff': 40, 'attr': 'real'},

                              {'coeff': 41, 'attr': 'real'},

                              {'coeff': 42, 'attr': 'real'},

                              {'coeff': 43, 'attr': 'real'},

                              {'coeff': 44, 'attr': 'real'},

                              {'coeff': 45, 'attr': 'real'},

                              {'coeff': 46, 'attr': 'real'},

                              {'coeff': 47, 'attr': 'real'},

                              {'coeff': 48, 'attr': 'real'},

                              {'coeff': 49, 'attr': 'real'},

                              {'coeff': 50, 'attr': 'real'},

                              {'coeff': 51, 'attr': 'real'},

                              {'coeff': 52, 'attr': 'real'},

                              {'coeff': 53, 'attr': 'real'},

                              {'coeff': 54, 'attr': 'real'},

                              {'coeff': 55, 'attr': 'real'},

                              {'coeff': 56, 'attr': 'real'},

                              {'coeff': 57, 'attr': 'real'},

                              {'coeff': 58, 'attr': 'real'},

                              {'coeff': 59, 'attr': 'real'},

                              {'coeff': 60, 'attr': 'real'},

                              {'coeff': 61, 'attr': 'real'},

                              {'coeff': 62, 'attr': 'real'},

                              {'coeff': 63, 'attr': 'real'},

                              {'coeff': 64, 'attr': 'real'}],

         }
# The package TSFRESH was developed to automate feature extraction and selection from time series data.

# https://tsfresh.readthedocs.io/en/latest/text/introduction.html

tsfresh_train = extract_features(X_train.drop(['row_id'], axis=1),

                                 column_id='series_id',

                                 column_sort='measurement_number',

                                 default_fc_parameters=params)

impute(tsfresh_train)
relevant_train_features = set()

for label in y_train['surface'].unique():

    y_train_binary = (y_train['surface'].values == label).astype(int)

    X_train_filtered = select_features(tsfresh_train, y_train_binary, fdr_level=0.382)

    print("Number of relevant features for class {}: {}/{}".format(

                    label, X_train_filtered.shape[1], tsfresh_train.shape[1]))

    relevant_train_features = relevant_train_features.union(set(X_train_filtered.columns))
tsfresh_test = extract_features(X_test.drop(['row_id'], axis=1),

                                column_id='series_id', 

                                column_sort='measurement_number',

                                default_fc_parameters=params)

impute(tsfresh_test)
len(relevant_train_features)
tsfresh_train = tsfresh_train[list(relevant_train_features)]

tsfresh_test = tsfresh_test[list(relevant_train_features)]
tsfresh_train.shape
tsfresh_test.shape
logger.info('Prepare models')
# LabelEncoder

le = LabelEncoder()

y_train['surface'] = le.fit_transform(y_train['surface'])
y_train['surface'].head()
# Create the scaler object

scaler = StandardScaler()

# Fit on the training data

scaler.fit(tsfresh_train)

# Transform both the training and testing data

tsfresh_train = scaler.transform(tsfresh_train)

tsfresh_test = scaler.transform(tsfresh_test)
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
logger.info('Start running model')
RFC = RandomForestClassifier(n_estimators=500, n_jobs=-1)
sub_preds_rf = np.zeros((tsfresh_test.shape[0], 9))

oof_preds_rf = np.zeros((tsfresh_train.shape[0]))

score = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(tsfresh_train, y_train['surface'])):

    clf = RFC

    clf.fit(tsfresh_train[trn_idx], y_train['surface'][trn_idx])

    oof_preds_rf[val_idx] = clf.predict(tsfresh_train[val_idx])

    sub_preds_rf += clf.predict_proba(tsfresh_test) / folds.n_splits

    score += clf.score(tsfresh_train[val_idx], y_train['surface'][val_idx])

    print('Fold: {} score: {}'.format(fold_, clf.score(tsfresh_train[val_idx], y_train['surface'][val_idx])))

print('Avg Accuracy', score / folds.n_splits)
logger.info('Prepare confusion matrix')
def plot_confusion_matrix(actual, predicted, classes, title='Confusion Matrix'):

    conf_matrix = confusion_matrix(actual, predicted)

    

    plt.figure(figsize=(8, 8))

    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)

    plt.title(title, size=12)

    plt.colorbar(fraction=0.05, pad=0.05)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90)

    plt.yticks(tick_marks, classes)



    thresh = conf_matrix.max() / 2.

    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):

        plt.text(j, i, format(conf_matrix[i, j], 'd'),

        horizontalalignment="center", color="white" if conf_matrix[i, j] > thresh else "black")



    plt.ylabel('True Label')

    plt.xlabel('Predicted Label')

    plt.grid(False)

    plt.tight_layout()
plot_confusion_matrix(y_train['surface'], oof_preds_rf, le.classes_)
logger.info('Start running model')
param_grid = {

    'max_depth': [3, 4, 5, 6],  # the maximum depth of each tree

    'min_child_weight': np.linspace(0.8, 1.2, 4),

    'gamma': np.linspace(0, 0.2, 4),

}
XGB = xgb.sklearn.XGBClassifier(learning_rate = 0.025,

                                objective = 'multi:softmax',

                                n_estimators = 150,

                                max_depth = 5,

                                min_child_weight = 1.2,

                                subsample=0.8,

                                colsample_bytree = 0.8,

                                gamma = 0.066,

                                n_jobs = -1,

                                silent = True,

                                seed = 42)
sub_preds_xgb = np.zeros((tsfresh_test.shape[0], 9))

oof_preds_xgb = np.zeros((tsfresh_train.shape[0]))

score = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(tsfresh_train, y_train['surface'])):

    clf =  XGB

    clf.fit(tsfresh_train[trn_idx], y_train['surface'][trn_idx])

    oof_preds_xgb[val_idx] = clf.predict(tsfresh_train[val_idx])

    sub_preds_xgb += clf.predict_proba(tsfresh_test) / folds.n_splits

    score += clf.score(tsfresh_train[val_idx], y_train['surface'][val_idx])

    print('Fold: {} score: {}'.format(fold_, clf.score(tsfresh_train[val_idx], y_train['surface'][val_idx])))

print('Avg Accuracy', score / folds.n_splits)
logger.info('Prepare confusion matrix')
plot_confusion_matrix(y_train['surface'], oof_preds_xgb, le.classes_)
logger.info('Start running model')
SCLF = StackingCVClassifier(classifiers=[XGB, RFC],

                            meta_classifier=RFC,

                            use_features_in_secondary=True,

                            n_jobs=-1,

                            random_state=42)
sub_preds_sclf = np.zeros((tsfresh_test.shape[0], 9))

oof_preds_sclf = np.zeros((tsfresh_train.shape[0]))

score = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(tsfresh_train, y_train['surface'])):

    clf =  SCLF

    clf.fit(tsfresh_train[trn_idx], y_train['surface'][trn_idx])

    oof_preds_sclf[val_idx] = clf.predict(tsfresh_train[val_idx])

    sub_preds_sclf += clf.predict_proba(tsfresh_test) / folds.n_splits

    score += clf.score(tsfresh_train[val_idx], y_train['surface'][val_idx])

    print('Fold: {} score: {}'.format(fold_, clf.score(tsfresh_train[val_idx], y_train['surface'][val_idx])))

print('Avg Accuracy', score / folds.n_splits)
logger.info('Prepare confusion matrix')
plot_confusion_matrix(y_train['surface'], oof_preds_sclf, le.classes_)
logger.info("Prepare submission")
submission = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
submission['surface'] = le.inverse_transform(sub_preds_sclf.argmax(axis=1))
submission.to_csv('submission.csv', index=False)
submission.head(10)