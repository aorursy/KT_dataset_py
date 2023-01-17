import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostClassifier

import itertools

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.simplefilter(action='ignore', category=DeprecationWarning)

%matplotlib inline

print(os.listdir("../input"))
train_df = pd.read_csv("../input/train.csv")

train_df.head()
train_df.info()
def null_values(df):

        mis_val = df.isnull().sum()

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        print ("Dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        return mis_val_table_ren_columns
# Missing values statistics

miss_values = null_values(train_df)

miss_values.head(20)
test_df = pd.read_csv("../input/test.csv")

test_df.head()
test_df.shape
plt.figure(figsize=(5,8))

sns.countplot(x="target", data=train_df)

plt.xlabel("Target", fontsize=12)

plt.ylabel("Count", fontsize=12)

plt.title("Santander Customer Prediction", fontsize=15)

plt.show()
train_df.columns
#Randomly select some var features

var_features = ["var_0", "var_15", "var_22", "var_25", "var_39", "var_45", "var_52", "var_68", 

                "var_85", "var_99", "var_112", "var_135", "var_147", "var_158", "var_160", 

                "var_169", "var_173", "var_180", "var_192", "var_199"]
plt.figure(figsize=(20,25))

plt.title('Var features distributions')

i = 0



for var in var_features:

    i += 1

    plt.subplot(5, 4, i)

    sns.distplot(test_df.sample(10000)[var], label='Test set', hist=False)

    sns.distplot(train_df.sample(10000)[var], label='Train set', hist=False)

    plt.xlim((-100, 100))

    plt.legend()

    plt.xlabel(var, fontsize=12)



plt.show()
train_df.drop("ID_code", axis = 1, inplace=True)

test_df_new = test_df.drop("ID_code", axis = 1)

print(train_df.shape)

print(test_df_new.shape)
X = np.array(train_df.loc[:, train_df.columns != 'target'])

y = np.array(train_df.loc[:, train_df.columns == 'target'])

print('Shape of X: {}'.format(X.shape))

print('Shape of y: {}'.format(y.shape))
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)



print("X_train dataset: ", X_train.shape)

print("y_train dataset: ", y_train.shape)

print("X_valid dataset: ", X_valid.shape)

print("y_valid dataset: ", y_valid.shape)
# import time

# print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))

# print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))



# start = time.time()

# sm = SMOTE(random_state=2)

# X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())



# print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))

# print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))



# print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))

# print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

# print("Elapsed Time ", time.time() - start)
#https://www.kaggle.com/jesucristo/santander-magic-lgb/

params = {

    'bagging_freq': 5,

    'bagging_fraction': 0.33,

    'boost_from_average':'false',

    'boost': 'gbdt',

    'feature_fraction': 0.04,

    'learning_rate': 0.008,

    'max_depth': -1,

    'metric':'auc',

    'min_data_in_leaf': 80,

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 13,

    'num_threads': 8,

    'tree_learner': 'serial',

    'objective': 'binary',

    'verbosity': 1

}

# lg_train = lgb.Dataset(X_train_res, label= y_train_res.ravel())

# lg_valid = lgb.Dataset(X_valid, label=y_valid.ravel())

lg_train = lgb.Dataset(X_train, label= y_train.ravel())

lg_valid = lgb.Dataset(X_valid, label=y_valid.ravel())

model = lgb.train(params, lg_train, 10000, valid_sets=[lg_valid], 

                  early_stopping_rounds=1000, verbose_eval=1000)

fig, ax = plt.subplots(figsize=(12,10))

lgb.plot_importance(model, height=0.8, ax=ax, max_num_features=30)

ax.grid(False)

plt.ylabel('Feature', size=12)

plt.xlabel('Importance', size=12)

plt.title("Importance of the Features of LightGBM Model", fontsize=15)

plt.show()
pred_val_lgb = model.predict(X_valid, num_iteration=model.best_iteration)
pred_val_lgb.shape
# for i in range(0,pred_val.shape[0]):

#     if pred_val[i]>=.7:       # setting threshold to .7

#        pred_val[i]=1

#     else:  

#        pred_val[i]=0
#pred_val[0:10]
def plot_confusion_matrix(cm, classes, title='Confusion matrix', normalize=False, cmap=plt.cm.Blues):

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fmt = '.2f' if normalize else 'd'



    fig, ax = plt.subplots(figsize=(15, 10))

    ax.imshow(cm, interpolation='nearest', cmap=cmap)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, size=15)

    plt.colorbar()

    plt.grid(False)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    thresh = (cm.max()+cm.min()) / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label', size=12)

    plt.xlabel('Predicted label', size=12)
# conf_mat_val = confusion_matrix(pred_val, y_valid)

# plot_confusion_matrix(conf_mat_val, [0, 1], 

#                       title='Confusion matrix on Validation data', normalize=True)
pred_test_lgb = model.predict(test_df_new.values, num_iteration=model.best_iteration)
# for i in range(0,pred_test.shape[0]):

#     if pred_test[i]>=.7:       # setting threshold to .7

#        pred_test[i]=1

#     else:  

#        pred_test[i]=0
sub_df_lgb = pd.DataFrame({"ID_code":test_df["ID_code"].values})

#sub_df["target"] = pred_test.astype(int)

sub_df_lgb["target"] = pred_test_lgb

sub_df_lgb.to_csv("lgb_submission.csv", index=False)

sub_df_lgb.head()
# params = { "booster": "gbtree",

#           "objective": "binary:logistic",

#           "eta": 0.02,

#           "max_depth": 2,

#           "min_child_weight": 1, 

#         "subsample": 0.5,

#         "colsample_bytree": 0.5

#          }

#import xgboost as xgb
xgb_model = xgb.XGBClassifier(max_depth=2, learning_rate=0.02, n_estimators=3000, booster="gbtree", 

                        objective="binary:logistic", min_child_weight=1, subsample = 0.8, 

                        colsample_bytree = 0.6, random_state=1234)
# xgb_cv = xgb.cv(params, dtrain=xgb_train, num_boost_round=10000, nfold=5, stratified=True, 

#                 metrics='auc', early_stopping_rounds= 1000, verbose_eval=1000, seed=142)
xgb_model.fit(X_train, y_train.ravel(), eval_set=[(X_valid, y_valid.ravel())], eval_metric="auc", 

        early_stopping_rounds=1000, verbose=True)
pred_test_xgb = xgb_model.predict_proba(test_df_new.values)[:, 1]
pred_test_xgb_new = xgb_model.predict(test_df_new.values)
#pred_test_xgb_new[0:10]

pred_test_xgb[0:10]
sub_df_xgb = pd.DataFrame({"ID_code":test_df["ID_code"].values})

#sub_df["target"] = pred_test.astype(int)

sub_df_xgb["target"] = pred_test_xgb

sub_df_xgb.to_csv("xgb_submission.csv", index=False)

sub_df_xgb.head()
cb_model = CatBoostClassifier(iterations=3000, learning_rate=0.02, depth=2, objective="Logloss")
cb_model.fit(X_train, y_train.ravel(), eval_set=[(X_valid, y_valid.ravel())], 

             early_stopping_rounds=1000)
pred_test_cb = cb_model.predict(test_df_new.values, prediction_type="Probability")[:, 1]

pred_test_cb[0:10]
pred_test_cb.shape
sub_df_cb = pd.DataFrame({"ID_code":test_df["ID_code"].values})

#sub_df["target"] = pred_test.astype(int)

sub_df_cb["target"] = pred_test_cb

sub_df_cb.to_csv("cb_submission.csv", index=False)

sub_df_cb.head()
pred_ensemble = 0.4 * pred_test_lgb + 0.3 * pred_test_xgb + 0.3 * pred_test_cb

pred_ensemble[0:10]
sub_df_ensemble = pd.DataFrame({"ID_code":test_df["ID_code"].values})

#sub_df["target"] = pred_test.astype(int)

sub_df_ensemble["target"] = pred_ensemble

sub_df_ensemble.to_csv("ensemble_submission.csv", index=False)

sub_df_ensemble.head()