import pandas as pd 

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)





import gc

from datetime import datetime 

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from catboost import CatBoostClassifier

from sklearn import svm

import lightgbm as lgb

from lightgbm import LGBMClassifier

import xgboost as xgb



pd.set_option('display.max_columns', 100)





RFC_METRIC = 'gini'  #metric used for RandomForrestClassifier

NUM_ESTIMATORS = 100 #number of estimators used for RandomForrestClassifier

NO_JOBS = 4 #number of parallel jobs used for RandomForrestClassifier





#TRAIN/VALIDATION/TEST SPLIT

#VALIDATION

VALID_SIZE = 0.20 # simple validation using train_test_split

TEST_SIZE = 0.20 # test size using_train_test_split



#CROSS-VALIDATION

NUMBER_KFOLDS = 5 #number of KFolds for cross-validation







RANDOM_STATE = 2018



MAX_ROUNDS = 1000 #lgb iterations

EARLY_STOP = 50 #lgb early stop 

OPT_ROUNDS = 1000  #To be adjusted based on best validation rounds

VERBOSE_EVAL = 50 #Print out metric result



IS_LOCAL = False



import os



if(IS_LOCAL):

    PATH="/kaggle/input/creditcardfraud"

else:

    PATH="/kaggle/input/creditcardfraud"

print(os.listdir(PATH))
data_df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
print("Credit Card Fraud Detection data -  rows:",data_df.shape[0]," columns:", data_df.shape[1])
data_df.head()
data_df.describe()
total = data_df.isnull().sum().sort_values(ascending = False)

percent = (data_df.isnull().sum()/data_df.isnull().count()*100).sort_values(ascending = False)

pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()
temp = data_df["Class"].value_counts()

df = pd.DataFrame({'Class': temp.index,'values': temp.values})



trace = go.Bar(

    x = df['Class'],y = df['values'],

    name="Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)",

    marker=dict(color="yellow"),

    text=df['values']

)

data = [trace]

layout = dict(title = 'Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)',

          xaxis = dict(title = 'Class', showticklabels=True), 

          yaxis = dict(title = 'Number of transactions'),

          hovermode = 'closest',width=600

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='class')
class_0 = data_df.loc[data_df['Class'] == 0]["Time"]

class_1 = data_df.loc[data_df['Class'] == 1]["Time"]

#plt.figure(figsize = (15,5))

#plt.title('Credit Card Transactions Time Density Plot')

#sns.set_color_codes("pastel")

#sns.distplot(class_0,kde=True,bins=480)

#sns.distplot(class_1,kde=True,bins=480)

#plt.show()

hist_data = [class_0, class_1]

group_labels = ['Not Fraud', 'Fraud']



fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)

fig['layout'].update(title='Credit Card Transactions Time Density Plot', xaxis=dict(title='Time [s]'))

iplot(fig, filename='dist_only')
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,8))

s = sns.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=data_df, palette="PRGn",showfliers=True)

s = sns.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=data_df, palette="PRGn",showfliers=False)

plt.show();
tmp = data_df[['Amount','Class']].copy()

class_0 = tmp.loc[tmp['Class'] == 0]['Amount']

class_1 = tmp.loc[tmp['Class'] == 1]['Amount']

class_0.describe()
fraud = data_df.loc[data_df['Class'] == 1]



trace = go.Scatter(

    x = fraud['Time'],y = fraud['Amount'],

    name="Amount",

     marker=dict(

                color='rgb(238,23,11)',

                line=dict(

                    color='yellow',

                    width=1),

                opacity=0.5,

            ),

    text= fraud['Amount'],

    mode = "markers"

)

data = [trace]

layout = dict(title = 'Amount of fraudulent transactions',

          xaxis = dict(title = 'Time [s]', showticklabels=True), 

          yaxis = dict(title = 'Amount'),

          hovermode='closest'

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='fraud-amount')
plt.figure(figsize = (16,16))

plt.title('Credit Card Transactions features correlation plot (Pearson)')

corr = data_df.corr()

sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Greens")

plt.show()
s = sns.lmplot(x='V20', y='Amount',data=data_df, hue='Class', fit_reg=True,scatter_kws={'s':2})

s = sns.lmplot(x='V7', y='Amount',data=data_df, hue='Class', fit_reg=True,scatter_kws={'s':2})

plt.show()
s = sns.lmplot(x='V2', y='Amount',data=data_df, hue='Class', fit_reg=True,scatter_kws={'s':2})

s = sns.lmplot(x='V5', y='Amount',data=data_df, hue='Class', fit_reg=True,scatter_kws={'s':2})

plt.show()
var = data_df.columns.values



i = 0

t0 = data_df.loc[data_df['Class'] == 0]

t1 = data_df.loc[data_df['Class'] == 1]



sns.set_style('whitegrid')

plt.figure()

fig, ax = plt.subplots(8,4,figsize=(17,29))



for feature in var:

    i += 1

    plt.subplot(8,4,i)

    sns.kdeplot(t0[feature], bw=0.5,label="Class = 0")

    sns.kdeplot(t1[feature], bw=0.5,label="Class = 1")

    plt.xlabel(feature, fontsize=12)

    locs, labels = plt.xticks()

    plt.tick_params(axis='both', which='major', labelsize=13)

plt.show();
target = 'Class'

predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',\

       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',\

       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',\

       'Amount']
train_df, test_df = train_test_split(data_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True )

train_df, valid_df = train_test_split(train_df, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True )
clf = RandomForestClassifier(n_jobs=NO_JOBS, 

                             random_state=RANDOM_STATE,

                             criterion=RFC_METRIC,

                             n_estimators=NUM_ESTIMATORS,

                             verbose=False)
clf.fit(train_df[predictors], train_df[target].values)
preds = clf.predict(valid_df[predictors])
tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': clf.feature_importances_})

tmp = tmp.sort_values(by='Feature importance',ascending=False)

plt.figure(figsize = (10,7))

plt.title('Features importance',fontsize=16)

s = sns.barplot(x='Feature',y='Feature importance',data=tmp)

s.set_xticklabels(s.get_xticklabels(),rotation=90)

plt.show()  
cm = pd.crosstab(valid_df[target].values, preds, rownames=['Actual'], colnames=['Predicted'])

fig, (ax1) = plt.subplots(ncols=1, figsize=(6,6))

sns.heatmap(cm, 

            xticklabels=['Not Fraud', 'Fraud'],

            yticklabels=['Not Fraud', 'Fraud'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="DarkGreen", cmap="Greens")

plt.title('Confusion Matrix', fontsize=15)

plt.show()
roc_auc_score(valid_df[target].values, preds)
clf = AdaBoostClassifier(random_state=RANDOM_STATE,

                         algorithm='SAMME.R',

                         learning_rate=0.8,

                             n_estimators=NUM_ESTIMATORS)
clf.fit(train_df[predictors], train_df[target].values)
preds = clf.predict(valid_df[predictors])
tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': clf.feature_importances_})

tmp = tmp.sort_values(by='Feature importance',ascending=False)

plt.figure(figsize = (9,6))

plt.title('Features importance',fontsize=14)

s = sns.barplot(x='Feature',y='Feature importance',data=tmp)

s.set_xticklabels(s.get_xticklabels(),rotation=90)

plt.show() 
cm = pd.crosstab(valid_df[target].values, preds, rownames=['Actual'], colnames=['Predicted'])

fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(cm, 

            xticklabels=['Not Fraud', 'Fraud'],

            yticklabels=['Not Fraud', 'Fraud'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="Red", cmap="Reds")

plt.title('Confusion Matrix', fontsize=14)

plt.show()
roc_auc_score(valid_df[target].values, preds)
clf = CatBoostClassifier(iterations=500,

                             learning_rate=0.02,

                             depth=12,

                             eval_metric='AUC',

                             random_seed = RANDOM_STATE,

                             bagging_temperature = 0.2,

                             od_type='Iter',

                             metric_period = VERBOSE_EVAL,

                             od_wait=100)
clf.fit(train_df[predictors], train_df[target].values,verbose=True)
preds = clf.predict(valid_df[predictors])
tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': clf.feature_importances_})

tmp = tmp.sort_values(by='Feature importance',ascending=False)

plt.figure(figsize = (10,6))

plt.title('Features importance',fontsize=15)

s = sns.barplot(x='Feature',y='Feature importance',data=tmp)

s.set_xticklabels(s.get_xticklabels(),rotation=90)

plt.show()  
cm = pd.crosstab(valid_df[target].values, preds, rownames=['Actual'], colnames=['Predicted'])

fig, (ax1) = plt.subplots(ncols=1, figsize=(6,6))

sns.heatmap(cm, 

            xticklabels=['Not Fraud', 'Fraud'],

            yticklabels=['Not Fraud', 'Fraud'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="Darkred", cmap="Reds")

plt.title('Confusion Matrix', fontsize=15)

plt.show()
roc_auc_score(valid_df[target].values, preds)
# Prepare the train and valid datasets

dtrain = xgb.DMatrix(train_df[predictors], train_df[target].values)

dvalid = xgb.DMatrix(valid_df[predictors], valid_df[target].values)

dtest = xgb.DMatrix(test_df[predictors], test_df[target].values)



#What to monitor (in this case, **train** and **valid**)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]



# Set xgboost parameters

params = {}

params['objective'] = 'binary:logistic'

params['eta'] = 0.039

params['silent'] = True

params['max_depth'] = 2

params['subsample'] = 0.8

params['colsample_bytree'] = 0.9

params['eval_metric'] = 'auc'

params['random_state'] = RANDOM_STATE
model = xgb.train(params, 

                dtrain, 

                MAX_ROUNDS, 

                watchlist, 

                early_stopping_rounds=EARLY_STOP, 

                maximize=True, 

                verbose_eval=VERBOSE_EVAL)
fig, (ax) = plt.subplots(ncols=1, figsize=(8,5))

xgb.plot_importance(model, height=0.9, title="Features importance (XGBoost)", ax=ax, color="violet") 

plt.show()
preds = model.predict(dtest)
roc_auc_score(test_df[target].values, preds)
params = {

          'boosting_type': 'gbdt',

          'objective': 'binary',

          'metric':'auc',

          'learning_rate': 0.05,

          'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)

          'max_depth': 4,  # -1 means no limit

          'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)

          'max_bin': 100,  # Number of bucketed bin for feature values

          'subsample': 0.9,  # Subsample ratio of the training instance.

          'subsample_freq': 1,  # frequence of subsample, <=0 means no enable

          'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.

          'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)

          'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization

          'nthread': 8,

          'verbose': 0,

          'scale_pos_weight':150, # because training data is extremely unbalanced 

         }
dtrain = lgb.Dataset(train_df[predictors].values, 

                     label=train_df[target].values,

                     feature_name=predictors)



dvalid = lgb.Dataset(valid_df[predictors].values,

                     label=valid_df[target].values,

                     feature_name=predictors)
evals_results = {}



model = lgb.train(params, 

                  dtrain, 

                  valid_sets=[dtrain, dvalid], 

                  valid_names=['train','valid'], 

                  evals_result=evals_results, 

                  num_boost_round=MAX_ROUNDS,

                  early_stopping_rounds=2*EARLY_STOP,

                  verbose_eval=VERBOSE_EVAL, 

                  feval=None)
fig, (ax) = plt.subplots(ncols=1, figsize=(8,5))

lgb.plot_importance(model, height=0.9, title="Features importance (LightGBM)", ax=ax,color="pink") 

plt.show()
preds = model.predict(test_df[predictors])
roc_auc_score(test_df[target].values, preds)
kf = KFold(n_splits = NUMBER_KFOLDS, random_state = RANDOM_STATE, shuffle = True)



# Create arrays and dataframes to store results

oof_preds = np.zeros(train_df.shape[0])

test_preds = np.zeros(test_df.shape[0])

feature_importance_df = pd.DataFrame()

n_fold = 0

for train_idx, valid_idx in kf.split(train_df):

    train_x, train_y = train_df[predictors].iloc[train_idx],train_df[target].iloc[train_idx]

    valid_x, valid_y = train_df[predictors].iloc[valid_idx],train_df[target].iloc[valid_idx]

    

    evals_results = {}

    model =  LGBMClassifier(

                  nthread=-1,

                  n_estimators=2000,

                  learning_rate=0.01,

                  num_leaves=80,

                  colsample_bytree=0.98,

                  subsample=0.78,

                  reg_alpha=0.04,

                  reg_lambda=0.073,

                  subsample_for_bin=50,

                  boosting_type='gbdt',

                  is_unbalance=False,

                  min_split_gain=0.025,

                  min_child_weight=40,

                  min_child_samples=510,

                  objective='binary',

                  metric='auc',

                  silent=-1,

                  verbose=-1,

                  feval=None)

    model.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 

                eval_metric= 'auc', verbose= VERBOSE_EVAL, early_stopping_rounds= EARLY_STOP)

    

    oof_preds[valid_idx] = model.predict_proba(valid_x, num_iteration=model.best_iteration_)[:, 1]

    test_preds += model.predict_proba(test_df[predictors], num_iteration=model.best_iteration_)[:, 1] / kf.n_splits

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = predictors

    fold_importance_df["importance"] = clf.feature_importances_

    fold_importance_df["fold"] = n_fold + 1

    

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

    del model, train_x, train_y, valid_x, valid_y

    gc.collect()

    n_fold = n_fold + 1

train_auc_score = roc_auc_score(train_df[target], oof_preds)

print('Full AUC score %.6f' % train_auc_score)
pred = test_preds