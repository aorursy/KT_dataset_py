# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt 

from collections import Counter

from imblearn.over_sampling import SMOTE, ADASYN

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import RandomOverSampler

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler

from sklearn.utils import resample

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')

test = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')
## gender to binary integer

train["Gender"][train["Gender"] == "Male"] = 1

train["Gender"][train["Gender"] == "Female"] = 0

train["Gender"] = train["Gender"].astype(int)

## Vehicle Age from cat to integer

train["Vehicle_Age"][train["Vehicle_Age"] == "< 1 Year"] = 1

train["Vehicle_Age"][train["Vehicle_Age"] == "1-2 Year"] = 2

train["Vehicle_Age"][train["Vehicle_Age"] == "> 2 Years"] = 3

train["Vehicle_Age"] = train["Vehicle_Age"].astype(int)

## Vehicle Damage to binary integer

train["Vehicle_Damage"][train["Vehicle_Damage"] == "Yes"] = 1

train["Vehicle_Damage"][train["Vehicle_Damage"] == "No"] = 0

train["Vehicle_Damage"] = train["Vehicle_Damage"].astype(int)



train['Policy_Sales_Channel'] = train['Policy_Sales_Channel'].apply(lambda x: np.int(x))

train['Region_Code'] = train['Region_Code'].apply(lambda x: np.int(x))



train['Drive_exp'] = train['Age'] - train['Age'].min() ## new feature - drive experience + some new features

train['Low_exp'] = train['Drive_exp'].map(lambda s:1 if s<9 else 0)

train['High_exp'] = train['Drive_exp'].map(lambda s:1 if s>20 else 0)

train['Mid_exp'] = train['Drive_exp'].map(lambda s:1 if s<=20 & s>=9 else 0)

train = train.drop('Age', axis=1)

## some new features based on Annual_Premium, later we'll remove unnecessary

train['Annual_log'] = np.log(train.Annual_Premium + 0.01)

ss = StandardScaler() 

train['Annual_scaled'] = ss.fit_transform(train['Annual_Premium'].values.reshape(-1,1))



mm = MinMaxScaler() 

train['Annual_minmax'] = mm.fit_transform(train['Annual_Premium'].values.reshape(-1,1))

## new features based on frequency of 28th region and 152nd channel  in dataset

train['Region_Code_28'] = train['Region_Code'].map(lambda s:1 if s==28 else 0)

train['Policy_Sales_Channel_152'] = train['Policy_Sales_Channel'].map(lambda s:1 if s==152 else 0)





train['Annual_Premium_10'] = train['Annual_Premium'].map(lambda s:1 if s<=10000 else 0)



# the same for test dataset:

test["Gender"][test["Gender"] == "Male"] = 1

test["Gender"][test["Gender"] == "Female"] = 0

test["Gender"] = test["Gender"].astype(int)



test["Vehicle_Age"][test["Vehicle_Age"] == "< 1 Year"] = 1

test["Vehicle_Age"][test["Vehicle_Age"] == "1-2 Year"] = 2

test["Vehicle_Age"][test["Vehicle_Age"] == "> 2 Years"] = 3

test["Vehicle_Age"] = test["Vehicle_Age"].astype(int)



test["Vehicle_Damage"][test["Vehicle_Damage"] == "Yes"] = 1

test["Vehicle_Damage"][test["Vehicle_Damage"] == "No"] = 0

test["Vehicle_Damage"] = test["Vehicle_Damage"].astype(int)



test['Policy_Sales_Channel'] = test['Policy_Sales_Channel'].apply(lambda x: np.int(x))

test['Region_Code'] = test['Region_Code'].apply(lambda x: np.int(x))



test['Drive_exp'] = test['Age'] - test['Age'].min() ## new feature - drive experience + some new features

test['Low_exp'] = test['Drive_exp'].map(lambda s:1 if s<9 else 0)

test['High_exp'] = test['Drive_exp'].map(lambda s:1 if s>20 else 0)

test['Mid_exp'] = test['Drive_exp'].map(lambda s:1 if s<=20 & s>=9 else 0)

test = test.drop('Age', axis=1)



test['Annual_log'] = np.log(test.Annual_Premium + 0.01)

ss = StandardScaler() 

test['Annual_scaled'] = ss.fit_transform(test['Annual_Premium'].values.reshape(-1,1))



mm = MinMaxScaler() 

test['Annual_minmax'] = mm.fit_transform(test['Annual_Premium'].values.reshape(-1,1))



test['Region_Code_28'] = test['Region_Code'].map(lambda s:1 if s==28 else 0)

test['Policy_Sales_Channel_152'] = test['Policy_Sales_Channel'].map(lambda s:1 if s==152 else 0)

test['Annual_Premium_10'] = test['Annual_Premium'].map(lambda s:1 if s<=10000 else 0)

train = train[:20000] ## let's take one part from dataset 

train
def basic_details(df):

    b = pd.DataFrame()

    b['Missing value, %'] = round(df.isnull().sum()/df.shape[0]*100)

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

basic_details(train)
train.describe().T
def basic_analysis(df1, df2):

    '''the function compares the average values of  2 dataframes'''

    b = pd.DataFrame()

    b['Response df_mean'] = round(df1.mean(),2)

    b['Not Response df_mean'] = round(df2.mean(),2)

    c = (b['Response df_mean']/b['Not Response df_mean'])

    if [c<=1]:

        b['Variation, %'] = round((1-((b['Response df_mean']/b['Not Response df_mean'])))*100)

    else:

        b['Variation, %'] = round(((b['Response df_mean']/b['Not Response df_mean'])-1)*100)

        

    b['Influence'] = np.where(abs(b['Variation, %']) <= 9, "feature's effect on the target is not defined", 

                              "feature value affects the target")



    return b
response = train.drop(train[train['Response'] != 1].index)

not_response = train.drop(train[train['Response'] != 0].index)

basic_analysis(response,not_response)
sns.countplot(train.Response)

train['Response'].value_counts(normalize=True)
## distribution of cat features

   

cat_features = train[[ 'Vehicle_Damage', 'Previously_Insured', 'Gender','Vehicle_Damage', 'Vehicle_Age', 'Driving_License']].columns

for i in cat_features:

    sns.barplot(x="Response",y=i,data=train)

    plt.title(i+" by "+"Response")

    plt.show()
## distribution and checking for outliers in numeric features

import matplotlib.pyplot as plt

import seaborn as sns

features = train[['Drive_exp', 'Policy_Sales_Channel', 'Vintage', 'Region_Code']].columns



for i in features:

    sns.boxplot(x="Response",y=i,data=train)

    plt.title(i+" by "+"Response")

    plt.show()
## PPS matrix of correlation of non-linear relations between features

%pip install ppscore # installing ppscore, library used to check non-linear relationships between our variables

import ppscore as pps # importing ppscore

matrix_pps = pps.matrix(train.drop('id', axis=1))[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')

matrix_pps = matrix_pps.apply(lambda x: round(x, 2))

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(matrix_pps, vmin=0, vmax=1, cmap="icefire", linewidths=0.75, annot=True)
def spearman(frame, features):

    spr = pd.DataFrame()

    spr['feature'] = features

    spr['spearman'] = [frame[f].corr(frame['Response'], 'spearman') for f in features]

    spr = spr.sort_values('spearman')

    plt.figure(figsize=(6, 0.25*len(features)))

    f, ax = plt.subplots(figsize=(12, 9))

    sns.barplot(data=spr, y='feature', x='spearman', orient='h')

    

features = train.drop(['Response', 'id'], axis = 1).columns

spearman(train, features)
corrMatrix = train.corr()

f, ax = plt.subplots(figsize=(20, 10))

sns.heatmap(corrMatrix, annot = True)

plt.show()
# Let's explore the Annual Premium by Response and see the distribuition of Amount transactions

fig , axs = plt.subplots(nrows = 1 , ncols = 4 , figsize = (16,4))



sns.boxplot(x ="Response",y="Annual_Premium",data=train, ax = axs[0])

axs[0].set_title("Response vs Annual_Premium")



sns.boxplot(x ="Response",y="Annual_log",data=train, ax = axs[1])

axs[1].set_title("Response vs Log Annual_Premium")



sns.boxplot(x ="Response",y="Annual_scaled",data=train, ax = axs[2])

axs[2].set_title("Response vs Scaled Annual_Premium")



sns.boxplot(x ="Response",y="Annual_minmax",data=train, ax = axs[3])

axs[3].set_title("Response vs Min-Max Annual_Premium")



plt.show()
## let's drop some features from dataset

train = train.drop(['Annual_Premium', 'Annual_minmax', 'Annual_log'], axis=1)

test = test.drop(['Annual_Premium', 'Annual_minmax', 'Annual_log'], axis=1)

train = train.drop('Driving_License', axis=1)

test = test.drop('Driving_License', axis=1)
def reduce_memory_usage(df):

    """ The function will reduce memory of dataframe

    Note: Apply this function after removing missing value"""

    intial_memory = df.memory_usage().sum()/1024**2

    print('Intial memory usage:',intial_memory,'MB')

    for col in df.columns:

        mn = df[col].min()

        mx = df[col].max()

        if df[col].dtype != object:            

            if df[col].dtype == int:

                if mn >=0:

                    if mx < np.iinfo(np.uint8).max:

                        df[col] = df[col].astype(np.uint8)

                    elif mx < np.iinfo(np.uint16).max:

                        df[col] = df[col].astype(np.uint16)

                    elif mx < np.iinfo(np.uint32).max:

                        df[col] = df[col].astype(np.uint32)

                    elif mx < np.iinfo(np.uint64).max:

                        df[col] = df[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        df[col] = df[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        df[col] = df[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        df[col] = df[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        df[col] = df[col].astype(np.int64)

            if df[col].dtype == float:

                df[col] =df[col].astype(np.float32)

    

    red_memory = df.memory_usage().sum()/1024**2

    print('Memory usage after complition: ',red_memory,'MB')

    

reduce_memory_usage(train)
X = train.drop(['Response', 'id'], axis=1)

y = train.Response

test_id = test.id.values



sm = SMOTE(random_state=42)

X_sm, y_sm = sm.fit_sample(X, y)



rus = RandomUnderSampler(random_state=42)

X_rus, y_rus = rus.fit_resample(X, y)



ros = RandomOverSampler(random_state=42)

X_ros, y_ros= ros.fit_resample(X, y)



adasyn = ADASYN(random_state=42)

X_ad, y_ad = adasyn.fit_resample(X, y)



x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=10)

x_train, y_train= rus.fit_resample(x_train, y_train)







d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)





params = {

        'objective':'binary:hinge',

        'n_estimators': 800,

        'max_depth':2,

        'learning_rate':0.01,

        'eval_metric':'auc',

        'min_child_weight':4,

        'subsample':0.1,

        'colsample_bytree':0.6,

        'seed':29,

        'reg_lambda':2.5,

        'reg_alpha':7,

        'gamma':0.01,

        'scale_pos_weight':0,

        'nthread':-1

}



watchlist = [(d_train, 'train'), (d_valid, 'valid')]

nrounds=5000

model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=800, 

                           maximize=True, verbose_eval=10)
y_pred = model.predict(d_valid)

print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_valid, y_pred))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_valid, y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_valid, y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_valid, y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_valid, y_pred)))
confusion_matrix(y_valid, y_pred)
xgb.plot_importance(model)


from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_valid, y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
var = train.drop(['id', 'Response'], axis=1).columns.values



i = 0

t0 = train.loc[train['Response'] == 0]

t1 = train.loc[train['Response'] == 1]



sns.set_style('whitegrid')

plt.figure()

fig, ax = plt.subplots(3,5,figsize=(22,28))



for feature in var:

    i += 1

    plt.subplot(3,5,i)

    sns.kdeplot(t0[feature], bw=0.5,label="Class = 0")

    sns.kdeplot(t1[feature], bw=0.5,label="Class = 1")

    plt.xlabel(feature, fontsize=12)

    locs, labels = plt.xticks()

    plt.tick_params(axis='both', which='major', labelsize=12)

plt.show();
dmatrix_data = xgb.DMatrix(data=X_ros, label=y_ros)



cv_params = {

        'objective':'binary:hinge',

        'max_depth':13,

        'learning_rate':0.1,

        'eval_metric':'auc',

        'min_child_weight':1,

        'subsample':1,

        'colsample_bytree':0.6,

        'seed':29,

        'reg_lambda':2.79,

        'reg_alpha':7,

        'gamma':0.01,

        'scale_pos_weight':0,

        'nthread':-1

}

cross_val = xgb.cv(

    params=cv_params,

    dtrain=dmatrix_data, 

    nfold=5,

    num_boost_round=5000, 

    early_stopping_rounds=1000, 

    metrics='auc', 

    as_pandas=True, 

    seed=29)

print(cross_val.tail(1))

# from sklearn.model_selection import GridSearchCV



# clf = xgb.XGBClassifier()

# parameters = {

#      "eta"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

#      "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],

#      "min_child_weight" : [ 1, 3, 5, 7 ],

#      "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

#      "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]

#      }



# grid = GridSearchCV(clf,

#                     parameters, n_jobs=4,

#                     scoring="neg_log_loss",

#                     cv=3)



# grid.fit(x_train, y_train)
# ## model for prediction

# d_train = xgb.DMatrix(x_train, label=y_train)

# d_valid = xgb.DMatrix(x_valid, label=y_valid)

# X_test = test.drop('id', axis=1)

# d_test = xgb.DMatrix(X_test)

# params = {

#         'objective':'binary:logistic',

#         'n_estimators': 500,

#         'max_depth':12,

#         'learning_rate':0.1,

#         'eval_metric':'auc',

#         'min_child_weight':1,

#         'subsample':1,

#         'colsample_bytree':0.6,

#         'seed':29,

#         'reg_lambda':2.79,

#         'reg_alpha':7,

#         'gamma':0.01,

#         'scale_pos_weight':1,

#         'nthread':-1

# }



# watchlist = [(d_train, 'train'), (d_valid, 'valid')]

# nrounds=10000

# model_1 = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=800, 

#                            maximize=True, verbose_eval=10)
# sub = pd.DataFrame()

# sub['ID'] = test['id']

# sub['Response'] = model_1.predict(d_test)

# sub.to_csv('submission.csv', index=False)



# sub.head()