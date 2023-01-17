import pandas as pd
import numpy as np

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import warnings

%matplotlib inline
warnings.filterwarnings('ignore')
loans_2007 = pd.read_csv('../input/loans_2007.csv', low_memory=False)

print(loans_2007.shape)
loans_2007.head()
freq_loan_status = loans_2007['loan_status'].value_counts() 

fig, ax = plt.subplots(figsize=(12,7))
freq_loan_status.plot(kind='barh',alpha=0.75, rot=0, colormap=plt.cm.Accent)

plt.title('Loan Status History, 2007-2011', size=20)
plt.xlabel('# of Borrowers', size=12)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.tick_params(top="off", left="off", right="off", bottom='off')
# Remove loans that don't contain either Fully Paid or Charged Off
loans_2007 = loans_2007[(loans_2007['loan_status'] == 'Fully Paid') | (loans_2007['loan_status'] == 'Charged Off')]


# Map Fully Paid/Charged Off to 0/1
mapping_dict = {
    'loan_status':{
        'Fully Paid':1,
        'Charged Off':0
    }
}

loans_2007.replace(mapping_dict, inplace=True)

print("\n----------------------------------\nShape after target col processing\n----------------------------------\n", loans_2007.shape)
cols_drop_1 = ['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'grade', 
               'sub_grade', 'emp_title', 'issue_d', 'zip_code']

loans_2007.drop(cols_drop_1, axis=1, inplace=True)
cols_drop_2 = ['out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
              'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
              'last_pymnt_d', 'last_pymnt_amnt']

loans_2007.drop(cols_drop_2, axis=1, inplace=True)
print("\n--------------------------\nShape after dropping cols\n--------------------------\n", loans_2007.shape)
print("\n-----------------------------------\nColumns with only one unique value\n-----------------------------------")
      
for col in loans_2007.columns:
    # Drop any null values
    non_null = loans_2007[col].dropna()
    
    # Check the number of unique values
    unique_non_null = non_null.unique()
    num_true_unique = len(unique_non_null)
    
    # Remove the col if there is only 1 unique value
    if num_true_unique == 1:
        loans_2007.drop(col, axis=1, inplace=True)
        print(col)
null_values = loans_2007.isnull().sum()
print("\n-----------------------------------\nPercentage of missing values (%):\n-----------------------------------\n", null_values*100 / len(loans_2007))
more_than_1pct = ['pub_rec_bankruptcies']
loans_2007.drop(more_than_1pct, axis=1, inplace=True)
loans_2007.dropna(inplace=True)
print("\n----------------------\nShape after cleaning\n----------------------\n", loans_2007.shape)
object_cols = list(loans_2007.select_dtypes(include=['object']).columns)
print("\n----------------------\nObject columns\n----------------------")
print(object_cols)
print("\n-----------------------------\nUnique values in object cols\n-----------------------------")
for col in object_cols:
    print(col, ":", len(loans_2007[col].unique()))
cols_drop = ['addr_state', 'title']
loans_2007.drop(cols_drop, axis=1, inplace=True)

for col in ['last_credit_pull_d', 'earliest_cr_line']:
    loans_2007.loc[:, col] = pd.DatetimeIndex(loans_2007[col]).astype(np.int64)*1e-9
float_cols = ['int_rate', 'revol_util']
for col in float_cols:
    loans_2007[col] = loans_2007[col].str.rstrip('%').astype(float)
    
loans_2007[float_cols].head()
mapping_dict = {
    'emp_length': {
        '10+ years': 10,
        '9 years': 9,
        '8 years': 8,
        '7 years': 7,
        '6 years': 6,
        '5 years': 5,
        '4 years': 4,
        '3 years': 3,
        '2 years': 2,
        '1 year': 1,
        '< 1 year': 0,
        'n/a': 0
    }
}

loans_2007.replace(mapping_dict, inplace=True)
dummy_cols = ['home_ownership', 'verification_status', 'term', 'purpose']

for col in dummy_cols:
    dummy_df = pd.get_dummies(loans_2007[col])
    loans_2007 = pd.concat([loans_2007, dummy_df], axis=1)
    loans_2007.drop(col, axis=1, inplace=True)    
print("\n----------------------------------\nShape after pre-processing\n----------------------------------\n", loans_2007.shape)
loans_2007.head()
def tpr_fpr(actual, predicted):
    # FP
    fp_filter = (actual == 0) & (predicted == 1)
    fp = len(predicted[fp_filter])
    
    # TP
    tp_filter = (actual == 1) & (predicted == 1) 
    tp = len(predicted[tp_filter])

    # TN
    tn_filter = (actual == 0) & (predicted == 0)
    tn = len(predicted[tn_filter])
    
    # FN
    fn_filter = (actual == 1) &(predicted == 0)
    fn = len(predicted[fn_filter])

    tpr = tp  / (tp + fn)
    fpr = fp  / (fp + tn)
    
    return(fpr, tpr)
# Data preparation
target = 'loan_status'
X_train = loans_2007.loc[:, loans_2007.columns != target]
y_train = loans_2007[target]

# LR Model
lr = LogisticRegression()

# Train
predictions_lr = cross_val_predict(lr, X_train, y_train, cv=3)
predictions_lr = pd.Series(predictions_lr)

# Test performance - FPR and TPR
fpr_lr, tpr_lr = tpr_fpr(y_train, predictions_lr)

print("\n----------------------------------\nLogistic Regression\n----------------------------------")
print("FPR:", fpr_lr)
print("TPR:", tpr_lr)
penalty = {
    0: 7,
    1: 1
}

# LR Model
lr = LogisticRegression(class_weight=penalty)

# Train
predictions_lr = cross_val_predict(lr, X_train, y_train, cv=3)
predictions_lr = pd.Series(predictions_lr)

# Test performance - FPR and TPR
fpr_lr, tpr_lr = tpr_fpr(y_train, predictions_lr)

print("\n----------------------------------\nLogistic Regression\n----------------------------------")
print("FPR:", fpr_lr)
print("TPR:", tpr_lr)
penalty = {
    0: 10,
    1: 1
}

# RF Model
rf = RandomForestClassifier(class_weight=penalty, random_state=1)

# Train
predictions_rf = cross_val_predict(rf, X_train, y_train, cv=3)
predictions_rf = pd.Series(predictions_rf)

# Test performance - FPR and TPR
fpr_rf, tpr_rf = tpr_fpr(y_train, predictions_rf)

print("\n----------------------------------\nRandom Forests\n----------------------------------")
print("FPR:", fpr_rf)
print("TPR:", tpr_rf)
# Data Preparation
X = loans_2007.loc[:, loans_2007.columns != target]
y = loans_2007[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print("------------------------------------------------\nShapes:\n------------------------------------------------")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Model
lgb_params = {'num_leaves': 50,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1}

FOLDs = KFold(n_splits=5, shuffle=True, random_state=1989)

oof_lgb = np.zeros(len(X_train))
predictions_lgb = np.zeros(len(X_test))

features_lgb = list(X_train.columns)
feature_importance_df_lgb = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(X_train)):
    trn_data = lgb.Dataset(X_train.iloc[trn_idx], label=y_train.iloc[trn_idx])
    val_data = lgb.Dataset(X_train.iloc[val_idx], label=y_train.iloc[val_idx])

    print("LGB " + str(fold_) + "-" * 50)
    num_round = 2000
    clf = lgb.train(lgb_params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 500)
    oof_lgb[val_idx] = clf.predict(X_train.iloc[val_idx], num_iteration=clf.best_iteration)

    fold_importance_df_lgb = pd.DataFrame()
    fold_importance_df_lgb["feature"] = features_lgb
    fold_importance_df_lgb["importance"] = clf.feature_importance()
    fold_importance_df_lgb["fold"] = fold_ + 1
    feature_importance_df_lgb = pd.concat([feature_importance_df_lgb, fold_importance_df_lgb], axis=0)
    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / FOLDs.n_splits
    
# Test performance - FPR and TPR
fpr_lgb, tpr_lgb = tpr_fpr(y_test, np.round(predictions_lgb))

print("\n----------------------------------\nLightGBM\n----------------------------------")
print("FPR:", fpr_lgb)
print("TPR:", tpr_lgb)
xgb_params = {'eta': 0.001, 'max_depth': 7, 'subsample': 0.8, 'colsample_bytree': 0.8, 
          'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True}

FOLDs = KFold(n_splits=5, shuffle=True, random_state=1989)

oof_xgb = np.zeros(len(X_train))
predictions_xgb = np.zeros(len(X_test))

for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(X_train)):
    trn_data = xgb.DMatrix(data=X_train.iloc[trn_idx], label=y_train.iloc[trn_idx])
    val_data = xgb.DMatrix(data=X_train.iloc[val_idx], label=y_train.iloc[val_idx])
    watchlist = [(trn_data, 'train'), (val_data, 'valid')]
    print("xgb " + str(fold_) + "-" * 50)
    num_round = 2000
    xgb_model = xgb.train(xgb_params, trn_data, num_round, watchlist, early_stopping_rounds=10, verbose_eval=200)
    oof_xgb[val_idx] = xgb_model.predict(xgb.DMatrix(X_train.iloc[val_idx]), ntree_limit=xgb_model.best_ntree_limit+50)

    predictions_xgb += xgb_model.predict(xgb.DMatrix(X_test), ntree_limit=xgb_model.best_ntree_limit+50) / FOLDs.n_splits
    
    
# Test performance - FPR and TPR
fpr_xgb, tpr_xgb = tpr_fpr(y_test, np.round(predictions_xgb))

print("\n----------------------------------\nXGBoost\n----------------------------------")
print("FPR:", fpr_xgb)
print("TPR:", tpr_xgb)
# Test performance - FPR and TPR (Ensemble)
fpr_ens, tpr_ens = tpr_fpr(y_test, np.round(0.5*predictions_lgb + 0.5*predictions_xgb))

print("\n----------------------------------\nEnsemble\n----------------------------------")
print("FPR:", fpr_ens)
print("TPR:", tpr_ens)
import seaborn as sns

cols = (feature_importance_df_lgb[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df_lgb.loc[feature_importance_df_lgb.feature.isin(cols)]

plt.figure(figsize=(14,14))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')
