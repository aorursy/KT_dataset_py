%%time

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import random

import xgboost as xgb



from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from xgboost import XGBRegressor

import lightgbm as lgb

from lightgbm import LGBMRegressor

from sklearn.metrics import accuracy_score

# from sklearn.cross_validation import StratifiedKFold

from sklearn.metrics import matthews_corrcoef, roc_auc_score

from catboost import CatBoostClassifier,CatBoostRegressor



from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder



from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor

from sklearn.feature_selection import VarianceThreshold



from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.model_selection import KFold

from sklearn.metrics import r2_score,mean_squared_error

from math import sqrt

from scipy import stats

from scipy.stats import norm, skew #for some statistics

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV,Ridge
from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import f1_score

from sklearn.metrics import fbeta_score

from sklearn.metrics import recall_score

from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers.normalization import BatchNormalization

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

from keras.models import load_model

from keras.initializers import glorot_normal, Zeros, Ones

import keras.backend as K

from keras.optimizers import RMSprop

import tensorflow as tf
from imblearn.over_sampling import SMOTE

from sklearn.utils import resample
from IPython.core.interactiveshell import InteractiveShell

from tqdm import tqdm_notebook

InteractiveShell.ast_node_interactivity = "all"
import os

os.listdir('../input/')
print(os.listdir("../input/"))

train = pd.read_csv('../input/Train.csv')

test = pd.read_csv('../input/Test.csv')
test.head()
train.head()
train.shape

test.shape
train.describe()

##feature num3-10 have almost identical std ,mean,unique values and distribution

##feature der1,2,3 also have almost identical std ,mean,unique values and distribution.
print('class count in numbers: ')

train['highValue'].value_counts()

print('percentage of class count : ')

train['highValue'].value_counts()/train.shape[0] * 100
##get duplicated rows

train["is_duplicate"]= train[list(set(train.columns) - set(['patent']))].duplicated()
##788 duplicated rows found

train.groupby(['is_duplicate'])['is_duplicate'].count()
train.drop(train[train['is_duplicate'] == True].index, inplace=True)

train.drop(['is_duplicate'],axis=1,inplace=True)
f,ax=plt.subplots(figsize=(12,12))

sns.heatmap(train[train.columns].corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)
train.dtypes
# Number of NaNs in each column

train.isnull().sum(axis=0)
train.nunique()
Y1=train['highValue']

# train1=train.drop(['employee_id','is_promoted'],axis=1)

train1=train.drop(['patent','highValue'],axis=1)

# train1=train.drop(['id','num7','num8','num9','num10','num11','cat10','cat13','target'],axis=1)

train1=train1

Y=Y1.values



test_id=test['patent']

# test1 = test.drop(['employee_id'],axis=1)

# test1 = test.drop(['id','num7','num8','num9','num10','num11','cat10','cat13'],axis=1)

test1 = test.drop(['patent','highValue'],axis=1)

test1=test1
## This is a generic function to plot the area under the curve (AUC) for a model

def plot_auc(y_test,y_pred):

    ## Calculates auc score

    fp_rate, tp_rate, treshold = roc_curve(y_test, y_pred)

    auc_score = auc(fp_rate, tp_rate)

    ## Creates a new figure and adds its parameters

    plt.figure()

    plt.title('ROC Curve')

    ## Plot the data - false positive rate and true positive rate

    plt.plot(fp_rate, tp_rate, 'b', label = 'AUC = %0.2f' % auc_score)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')
def evaluate_model(model, X_test_param, y_test_param):

    print("Model evaluation")

#     X_test_param, X_test, y_test_param, y_test = train_test_split(X_test_param, y_test_param, random_state=42,test_size=0.1)

#     model.fit(X_test_param, y_test_param,eval_set=[(X_test, y_test)],early_stopping_rounds=50,verbose=50)

    model.fit(X_test_param, y_test_param)

    y_pred = model.predict_proba(X_test_param)[:, 1]

    print("Accuracy: {:.5f}".format(model.score(X_test_param, y_test_param)))

    print("AUC: {:.5f}".format(roc_auc_score(y_test_param, y_pred)))

    print("\n#### Classification Report ####\n")

    

    thresholds = np.linspace(0.01, 0.99, 50)

    mcc = np.array([f1_score(y_test_param, y_pred>thr) for thr in thresholds])

    best_threshold = thresholds[mcc.argmax()]

    predictions = list(map(lambda x: 1 if x > best_threshold else 0,y_pred))

    print(classification_report(y_test_param, predictions, target_names=['0','1']))

    plot_auc(y_test_param, y_pred )
train1.shape

test1.shape

Y.shape
scaler = StandardScaler()

# scaler = QuantileTransformer(n_quantiles=10, random_state=0)

scaler.fit(train1)

# Apply transform to both the training set and the test set.

train2 = scaler.transform(train1)

test2 = scaler.transform(test1)
pca = PCA(n_components=2)

principalComponents = pca.fit_transform(train2)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])



finalDf = pd.concat([principalDf, train['highValue']], axis = 1)

finalDf.head(5)



fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 Component PCA', fontsize = 20)



targets = [0,1]

colors = ['r', 'g']

for target, color in zip(targets,colors):

    indicesToKeep = finalDf['highValue'] == target

    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']

               , finalDf.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

ax.legend(targets)

ax.grid()
clf = lgb.LGBMClassifier()

# clf = LogisticRegression()

# clf = lgb.LGBMClassifier(objective = 'binary',metric='binary_logloss',max_depth= 8, learning_rate=0.0941, n_estimators=197, num_leaves= 17, reg_alpha=3.4492 , reg_lambda= 0.0422)

evaluate_model(clf, train1.values, Y)
#create the cross validation fold for different boosting and linear model.

from sklearn.model_selection  import KFold,StratifiedKFold

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

SEED=42

clf = lgb.LGBMClassifier()

# clf = LogisticRegression()

st_train = train1.values

st_test = test1.values

# clf = lgb.LGBMClassifier(max_depth= 8, learning_rate=0.0941, n_estimators=197, num_leaves= 17, reg_alpha=3.4492 , reg_lambda= 0.0422) #lgb_pca

fold = 8

cv = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)

# cv = StratifiedKFold(Y, n_folds=fold,shuffle=True, random_state=30)

X_preds = np.zeros(st_train.shape[0])

preds = np.zeros(st_test.shape[0])

for i, (tr, ts) in enumerate(cv.split(st_train,Y)):

    print(ts.shape)

    mod = clf.fit(st_train[tr], Y[tr])

    X_preds[ts] = mod.predict_proba(st_train[ts])[:,1]

    preds += mod.predict_proba(st_test)[:,1]

    print("fold {}, ROC AUC: {:.3f}".format(i, roc_auc_score(Y[ts], X_preds[ts])))

score = roc_auc_score(Y, X_preds)

print(score)

preds1 = preds/fold
subfin = pd.DataFrame({'patent': test['patent'].values, 'highValue': preds1})

subfin=subfin.reindex(columns=["patent","highValue"])

subfin.to_csv('submission.csv', index=False)
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe

create_download_link(subfin)
thresholds = np.linspace(0.01, 0.99, 50)

mcc = np.array([matthews_corrcoef(Y, X_preds>thr) for thr in thresholds])

plt.plot(thresholds, mcc)

best_threshold = thresholds[mcc.argmax()]

print(mcc.max())

print(best_threshold)
prediction_rfc=list(range(len(preds1)))

for i in range(len(preds1)):

    prediction_rfc[i]=1 if preds1[i]>best_threshold else 0



sub = pd.DataFrame({'patent': test['patent'].values, 'highValue': prediction_rfc})

sub=sub.reindex(columns=["patent","highValue"])

filename = 'submission.csv'

sub.to_csv(filename, index=False)
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(sub)
lgb.plot_importance(clf,figsize=(20,10))
rf = RandomForestClassifier()

rf.fit(train1,Y)
plt.figure(figsize=(20,10))

plt.plot(rf.feature_importances_)



plt.xticks(np.arange(train1.shape[1]), train1.columns.tolist(), rotation=90);
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

import eli5

from eli5.sklearn import PermutationImportance

train_X, val_X, train_y, val_y = train_test_split(train1[:100000], Y[:100000], random_state=1)

first_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(train_X, train_y)



#Make a small change to the code below to use in this problem. 

perm = PermutationImportance(rf, random_state=1).fit(val_X, val_y)



#uncomment the following line to visualize your results

eli5.show_weights(perm, feature_names = val_X.columns.tolist())
#making copy

trn =  train.copy()

tst =  test.copy()
tst['is_train'] = 0

trn['is_train'] = 1 #1 for train
#combining test and train data

df_combine = pd.concat([trn, tst], axis=0, ignore_index=True)

#dropping 'target' column as it is not present in the test

df_combine = df_combine.drop(['highValue','patent'], axis=1)
y = df_combine['is_train'].values #labels

x = df_combine.drop('is_train', axis=1).values #covariates or our dependent variables
scaler = StandardScaler()

# scaler = QuantileTransformer(n_quantiles=10, random_state=0)

scaler.fit(x)

# Apply transform to both the training set and the test set.

train2 = scaler.transform(x)

# test2 = scaler.transform(test1)
pca = PCA(n_components=2)

principalComponents = pca.fit_transform(train2)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])



finalDf = pd.concat([principalDf, df_combine['is_train']], axis = 1)

finalDf.head(5)



fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 Component PCA', fontsize = 20)



targets = [0,1]

colors = ['r', 'g']

for target, color in zip(targets,colors):

    indicesToKeep = finalDf['is_train'] == target

    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']

               , finalDf.loc[indicesToKeep, 'principal component 2']

               , c = color

               , s = 50)

ax.legend(targets)

ax.grid()
df_combine.head()
df_train = train.copy()

df_test = test.copy()

train_target = train['highValue'].values

ntrain = df_train.shape[0]

ntest  = df_test.shape[0]
df_train.columns
from scipy.special import erfinv

def hot_encoder(df, columns):

    one_hot = {c: list(df[c].unique()) for c in columns}

    for c in one_hot:

        for val in one_hot[c]:

            df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)

    return df



def scale_feat(df):

    scaler = StandardScaler()

    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)



def rank_gauss(x):

    # x is numpy vector

    N = x.shape[0]

    temp = x.argsort()

    rank_x = temp.argsort() / N

    rank_x -= rank_x.mean()

    rank_x *= 2 # rank_x.max(), rank_x.min() should be in (-1, 1)

    efi_x = erfinv(rank_x) # np.sqrt(2)*erfinv(rank_x)

    efi_x -= efi_x.mean()

    return efi_x



def df_inputSwapNoise(df, p):

    ### feature with another value from the same column with probability p

    n = df.shape[0]

    idx = list(range(n))

    swap_n = round(n*p)

    for col in df.columns:

        arr = df[col].values

        col_vals = np.random.permutation(arr)

        swap_idx = np.random.choice(idx, size= swap_n)

        arr[swap_idx] = np.random.choice(col_vals, size = swap_n)

        df[col] = arr

    return df
print('Transforming data')

feature_cols = [c for c in df_train.columns if c not in ['patent','highValue']]

keep_cols    = [c for c in feature_cols]

#scale_cols = ['num18','num20','num21','num22','cat14']

#keep_cols    = [c for c in feature_cols if c not in scale_cols] 

#cat_cols     = [c for c in keep_cols if '_cat' in c]

##num18,num20,num21,num22,cat14



df_all = pd.concat([df_train[keep_cols], df_test[keep_cols]])

#df_all = scale_feat(df_all)

df_all_org = df_all.copy()

df_all_noise = df_inputSwapNoise(df_all, 0.15)

#df_all = hot_encoder(df_all, keep_cols)

data_all_org = df_all_org.values

data_all_noise = df_all_noise.values

cols = data_all_org.shape[1]

#print(df_all.columns)

print('Final data with {} columns'.format(cols))
for i in range(cols):

    u = np.unique(data_all_org[:,i])

    if u.shape[0] > 3:

        data_all_org[:,i] = rank_gauss(data_all_org[:,i])



for i in range(cols):

    u = np.unique(data_all_noise[:,i])

    if u.shape[0] > 3:

        data_all_noise[:,i] = rank_gauss(data_all_noise[:,i])



train_data_orig = data_all_org[0:ntrain,:]

test_data_orig  = data_all_org[ntrain:,:]

train_data_noise = data_all_noise[0:ntrain,:]

test_data_noise  = data_all_noise[ntrain:,:]

print(train_data_orig.shape)

print(test_data_orig.shape)

print(train_data_noise.shape)

print(test_data_noise.shape)
print('Original data')

all_data = np.vstack((train_data_orig, test_data_orig))

print('Noise data')

all_data_noise = np.vstack((train_data_noise, test_data_noise))
print('Creating neural net')

model = Sequential()

model.add(Dense(units=1500, input_dim = all_data.shape[1], kernel_initializer=glorot_normal()))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dense(units=1500, kernel_initializer=glorot_normal()))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dense(units=1500, kernel_initializer=glorot_normal()))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dense(all_data.shape[1])) 

model.add(Activation('linear'))



opt = keras.optimizers.Adam(lr=0.001)

model.compile(loss='mse', optimizer=opt)
print('Training neural net')

epochs = 10

chck = ModelCheckpoint('keras_dae.h5', monitor='loss', save_best_only=True)

cb = [ EarlyStopping(monitor='loss', patience=100, verbose=2, min_delta=0), chck ]

model.fit(all_data_noise, all_data, batch_size=128, verbose=1, epochs=epochs, callbacks=cb)



print('Applying neural net')

train_data_transform = model.predict(train_data_orig)

test_data_transform = model.predict(test_data_orig)

print(train_data_transform.shape)

print(test_data_transform.shape)
train_data_transform.shape

test_data_transform.shape
clf = lgb.LGBMClassifier()

# clf = lgb.LGBMClassifier(objective = 'binary',metric='binary_logloss',max_depth= 8, learning_rate=0.0941, n_estimators=197, num_leaves= 17, reg_alpha=3.4492 , reg_lambda= 0.0422)

evaluate_model(clf, train_data_transform, train_target)
#create the cross validation fold for different boosting and linear model.

from sklearn.model_selection  import KFold,StratifiedKFold

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

SEED=42

clf = lgb.LGBMClassifier()

# clf = LogisticRegression()

st_train = train_data_transform

st_test = test_data_transform

Y = train_target

fold = 5

cv = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)

X_preds = np.zeros(st_train.shape[0])

preds = np.zeros(st_test.shape[0])

for i, (tr, ts) in enumerate(cv.split(st_train,Y)):

    print(ts.shape)

    mod = clf.fit(st_train[tr], Y[tr])

    X_preds[ts] = mod.predict_proba(st_train[ts])[:,1]

    preds += mod.predict_proba(st_test)[:,1]

    print("fold {}, ROC AUC: {:.3f}".format(i, roc_auc_score(Y[ts], X_preds[ts])))

score = roc_auc_score(Y, X_preds)

print(score)

preds1 = preds/fold
#lightgbm bayesian optimization

from sklearn.model_selection import cross_val_score

from bayes_opt import BayesianOptimization



def xgboostcv(max_depth,learning_rate,n_estimators,num_leaves,reg_alpha,reg_lambda):

    return cross_val_score(lgb.LGBMClassifier(max_depth=int(max_depth),learning_rate=learning_rate,n_estimators=int(n_estimators),

                                             silent=True,nthread=-1,num_leaves=int(num_leaves),reg_alpha=reg_alpha,

                                           reg_lambda=reg_lambda),train1,Y,"roc_auc",cv=3).mean()



xgboostBO = BayesianOptimization(xgboostcv,{'max_depth': (4, 10),'learning_rate': (0.001, 0.1),'n_estimators': (10, 1000),

                                  'num_leaves': (4,30),'reg_alpha': (1, 5),'reg_lambda': (0, 0.1)})

xgboostBO.maximize()

print('-'*53)

print('Final Results')

print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])