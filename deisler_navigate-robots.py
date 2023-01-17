# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input")

     )



# Any results you write to the current directory are saved as output.
import os

import time

import numpy as np

import pandas as pd

import seaborn as sns

from seaborn import countplot,lineplot, barplot

import matplotlib.pyplot as plt

from matplotlib import rcParams

%matplotlib inline



from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.model_selection import LeaveOneGroupOut

from sklearn.model_selection import GroupKFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder, StandardScaler

le = preprocessing.LabelEncoder()



from scipy import stats

from scipy.stats import norm

from scipy.stats import randint as sp_randint



from numba import jit

import itertools



from bayes_opt import BayesianOptimization

import lightgbm as lgb

import xgboost as xgb

import catboost as cb



import warnings

warnings.filterwarnings('ignore')

import gc

gc.enable()
tr = pd.read_csv('../input/X_train.csv')

te = pd.read_csv('../input/X_test.csv')

target = pd.read_csv('../input/y_train.csv')

ss = pd.read_csv('../input/sample_submission.csv')

data = pd.read_csv('../input/X_train.csv')
tr.head()
tr.shape, te.shape
len(tr.measurement_number.value_counts())
print(len(tr.series_id.value_counts()))

print(128*3810,tr.shape)
len(te.measurement_number.value_counts())
len(te.series_id.value_counts())
target.head()
len(target.series_id.value_counts())
target.surface.value_counts()
sns.set(style='darkgrid')

sns.countplot(y = 'surface',

              data = target,

              order = target['surface'].value_counts().index)

plt.show()
data.describe()
te.describe()
target.describe()
def missing_data(data):

    totalt = data.isnull().sum().sort_values(ascending=False)

    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

    return pd.concat([totalt, percent], axis=1, keys=['Total', 'Percent'])
train_missing_data = missing_data(tr)

print ("Missing Data at Training")

train_missing_data.tail()
test_missing_data = missing_data(te)

print ("Missing Data at Training")

test_missing_data.tail()
target.groupby('group_id').surface.nunique().max()
target['group_id'].nunique()
group27 = target[ target.group_id == 27]

print(group27)
target.group_id.value_counts()
plt.figure(figsize=(23,5)) 

sns.set(style="darkgrid")

countplot(x="group_id", data=target, order = target['group_id'].value_counts().index)

plt.show()
train_group = pd.merge(tr,target, on = 'series_id', how = 'left')

train_group[:5]
serie1 = tr.head(128)

serie1.head()
serie1.describe()
plt.figure(figsize=(26, 16))

for i, col in enumerate(serie1.columns[3:]):

    plt.subplot(3, 4, i + 1)

    plt.plot(serie1[col])

    plt.title(col)
del serie1

gc.collect()
f,ax = plt.subplots(figsize=(10, 8))

sns.heatmap(tr.iloc[:,3:].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
f,ax = plt.subplots(figsize=(10, 8))

sns.heatmap(te.iloc[:,3:].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
def plot_feature_distribution(df1, df2, label1, label2, features,a=2,b=5):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(a,b,figsize=(17,9))



    for feature in features:

        i += 1

        plt.subplot(a,b,i)

        sns.kdeplot(df1[feature], bw=0.5,label=label1)

        sns.kdeplot(df2[feature], bw=0.5,label=label2)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=8)

        plt.tick_params(axis='y', which='major', labelsize=8)

    plt.show();
features = data.columns.values[3:]

plot_feature_distribution(data, te, 'train', 'test', features)
def plot_feature_class_distribution(classes,tt, features,a=5,b=2):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(a,b,figsize=(16,24))



    for feature in features:

        i += 1

        plt.subplot(a,b,i)

        for clas in classes:

            ttc = tt[tt['surface']==clas]

            sns.kdeplot(ttc[feature], bw=0.5,label=clas)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=8)

        plt.tick_params(axis='y', which='major', labelsize=8)

    plt.show();
classes = (target['surface'].value_counts()).index

aux = data.merge(target, on='series_id', how='inner')

plot_feature_class_distribution(classes, aux, features)
plt.figure(figsize=(26, 26))

for i,col in enumerate(aux.columns[3:13]):

    ax = plt.subplot(5,2,i+1)

    ax = plt.title(col)

    for surface in classes:

        surface_feature = aux[aux['surface'] == surface]

        sns.kdeplot(surface_feature[col], label = surface)
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
def fe_step0 (actual):

    

    # https://www.mathworks.com/help/aeroblks/quaternionnorm.html

    # https://www.mathworks.com/help/aeroblks/quaternionmodulus.html

    # https://www.mathworks.com/help/aeroblks/quaternionnormalize.html

        

    actual['norm_quat'] = (actual['orientation_X']**2 + actual['orientation_Y']**2 + actual['orientation_Z']**2 + actual['orientation_W']**2)

    actual['mod_quat'] = (actual['norm_quat'])**0.5

    actual['norm_X'] = actual['orientation_X'] / actual['mod_quat']

    actual['norm_Y'] = actual['orientation_Y'] / actual['mod_quat']

    actual['norm_Z'] = actual['orientation_Z'] / actual['mod_quat']

    actual['norm_W'] = actual['orientation_W'] / actual['mod_quat']

    

    return actual
data = fe_step0(data)

test = fe_step0(te)

print(data.shape)

data.head()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(18, 5))



ax1.set_title('quaternion X')

sns.kdeplot(data['norm_X'], ax=ax1, label="train")

sns.kdeplot(test['norm_X'], ax=ax1, label="test")



ax2.set_title('quaternion Y')

sns.kdeplot(data['norm_Y'], ax=ax2, label="train")

sns.kdeplot(test['norm_Y'], ax=ax2, label="test")



ax3.set_title('quaternion Z')

sns.kdeplot(data['norm_Z'], ax=ax3, label="train")

sns.kdeplot(test['norm_Z'], ax=ax3, label="test")



ax4.set_title('quaternion W')

sns.kdeplot(data['norm_W'], ax=ax4, label="train")

sns.kdeplot(test['norm_W'], ax=ax4, label="test")



plt.show()
def fe_step1 (actual):

    """Quaternions to Euler Angles"""

    

    x, y, z, w = actual['norm_X'].tolist(), actual['norm_Y'].tolist(), actual['norm_Z'].tolist(), actual['norm_W'].tolist()

    nx, ny, nz = [], [], []

    for i in range(len(x)):

        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])

        nx.append(xx)

        ny.append(yy)

        nz.append(zz)

    

    actual['euler_x'] = nx

    actual['euler_y'] = ny

    actual['euler_z'] = nz

    return actual
data = fe_step1(data)

test = fe_step1(test)

print (data.shape)

data.head()
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 5))



ax1.set_title('Roll')

sns.kdeplot(data['euler_x'], ax=ax1, label="train")

sns.kdeplot(test['euler_x'], ax=ax1, label="test")



ax2.set_title('Pitch')

sns.kdeplot(data['euler_y'], ax=ax2, label="train")

sns.kdeplot(test['euler_y'], ax=ax2, label="test")



ax3.set_title('Yaw')

sns.kdeplot(data['euler_z'], ax=ax3, label="train")

sns.kdeplot(test['euler_z'], ax=ax3, label="test")



plt.show()
data.head()
def feat_eng(data):

    

    df = pd.DataFrame()

    data['totl_anglr_vel'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 + data['angular_velocity_Z']**2)** 0.5

    data['totl_linr_acc'] = (data['linear_acceleration_X']**2 + data['linear_acceleration_Y']**2 + data['linear_acceleration_Z']**2)**0.5

    data['totl_xyz'] = (data['orientation_X']**2 + data['orientation_Y']**2 + data['orientation_Z']**2)**0.5

    data['acc_vs_vel'] = data['totl_linr_acc'] / data['totl_anglr_vel']

    

    def mean_change_of_abs_change(x):

        return np.mean(np.diff(np.abs(np.diff(x))))

    

    for col in data.columns:

        if col in ['row_id','series_id','measurement_number']:

            continue

        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()

        df[col + '_median'] = data.groupby(['series_id'])[col].median()

        df[col + '_max'] = data.groupby(['series_id'])[col].max()

        df[col + '_min'] = data.groupby(['series_id'])[col].min()

        df[col + '_std'] = data.groupby(['series_id'])[col].std()

        df[col + '_range'] = df[col + '_max'] - df[col + '_min']

        df[col + '_maxtoMin'] = df[col + '_max'] / df[col + '_min']

        df[col + '_mean_abs_chg'] = data.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

        df[col + '_mean_change_of_abs_change'] = data.groupby('series_id')[col].apply(mean_change_of_abs_change)

        df[col + '_abs_max'] = data.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))

        df[col + '_abs_min'] = data.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))

        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max'])/2

    return df
%%time

data = feat_eng(data)

test = feat_eng(test)

print ("New features: ",data.shape)
data.head()
#https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas

corr_matrix = data.corr().abs()

raw_corr = data.corr()



sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

                 .stack()

                 .sort_values(ascending=False))

top_corr = pd.DataFrame(sol).reset_index()

top_corr.columns = ["var1", "var2", "abs corr"]

# with .abs() we lost the sign, and it's very important.

for x in range(len(top_corr)):

    var1 = top_corr.iloc[x]["var1"]

    var2 = top_corr.iloc[x]["var2"]

    corr = raw_corr[var1][var2]

    top_corr.at[x, "raw corr"] = corr
top_corr.head(15)
data.fillna(0,inplace=True)

test.fillna(0,inplace=True)

data.replace(-np.inf,0,inplace=True)

data.replace(np.inf,0,inplace=True)

test.replace(-np.inf,0,inplace=True)

test.replace(np.inf,0,inplace=True)
target.head()
target['surface'] = le.fit_transform(target['surface'])
target['surface'].value_counts()
target.head()
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=20)

predicted = np.zeros((test.shape[0],9))

measured= np.zeros((data.shape[0]))

score = 0
for times, (trn_idx, val_idx) in enumerate(folds.split(data.values,target['surface'].values)):

    model = RandomForestClassifier(n_estimators=500)

    #model = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=5, n_jobs=-1)

    model.fit(data.iloc[trn_idx],target['surface'][trn_idx])

    measured[val_idx] = model.predict(data.iloc[val_idx])

    predicted += model.predict_proba(test)/folds.n_splits

    score += model.score(data.iloc[val_idx],target['surface'][val_idx])

    print("Fold: {} score: {}".format(times,model.score(data.iloc[val_idx],target['surface'][val_idx])))



    importances = model.feature_importances_

    indices = np.argsort(importances)

    features = data.columns

    

    if model.score(data.iloc[val_idx],target['surface'][val_idx]) > 0.91000:

        hm = 30

        plt.figure(figsize=(7, 10))

        plt.title('Feature Importances')

        plt.barh(range(len(indices[:hm])), importances[indices][:hm], color='b', align='center')

        plt.yticks(range(len(indices[:hm])), [features[i] for i in indices])

        plt.xlabel('Relative Importance')

        plt.show()

    gc.collect()
print('Avg Accuracy', score / folds.n_splits)
confusion_matrix(measured,target['surface'])
def plot_confusion_matrix(truth, pred, classes, normalize=False, title=''):

    cm = confusion_matrix(truth, pred)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    

    plt.figure(figsize=(10, 10))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    plt.title('Confusion matrix', size=15)

    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.grid(False)

    plt.tight_layout()
plot_confusion_matrix(target['surface'], measured, le.classes_)
ss['surface'] = le.inverse_transform(predicted.argmax(axis=1))

ss.to_csv('submission.csv', index=False)

ss.head()
# best_sub = pd.read_csv('../input/robots-best-submission/mybest0.73.csv')

# best_sub.to_csv('best_submission.csv', index=False)

# best_sub.head(10)