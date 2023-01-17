import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import random

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, log_loss, f1_score, roc_auc_score, average_precision_score, \
    precision_recall_curve, roc_curve, confusion_matrix

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 220)
pd.set_option('display.max_colwidth', None)
random_seed = 100
random.seed(random_seed)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
input_folder = '/kaggle/input/home-credit-default-risk/'

df_desc = pd.read_csv(input_folder + 'HomeCredit_columns_description.csv', encoding='cp1252')
df_app_train = pd.read_csv(input_folder + 'application_train.csv')
df_app_test = pd.read_csv(input_folder + 'application_test.csv')
df_pos_bal = pd.read_csv(input_folder + 'POS_CASH_balance.csv')
df_desc[['Table', 'Row', 'Description', 'Special']]
df_app_train.head()
print('Percentage of positive target in training data:', round(100 * df_app_train.TARGET.sum() / len(df_app_train), 2))
df_app_train.NAME_EDUCATION_TYPE.unique()
recurring_cat_cols = [col for col in df_app_train.columns if col[:4] in ['NAME', 'FLAG', 'REG_', 'LIVE']]
cat_cols = ['CODE_GENDER', 'OCCUPATION_TYPE', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START',
            'ORGANIZATION_TYPE', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE', 
            'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
            'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR'] + recurring_cat_cols

num_cols = [col for col in df_app_train.columns if col not in cat_cols + ['SK_ID_CURR', 'TARGET']]

# Features that don't seem too important:
drop_cols = ['WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'WALLSMATERIAL_MODE'] 
print('Number of categorical features:', len(cat_cols))
print('Number of numerical features:', len(num_cols))
df_app_train[num_cols].hist(bins=20, figsize=(50, 50))
plt.show()
print('Min value in AMT_INCOME_TOTAL:', df_app_train['AMT_INCOME_TOTAL'].min())
print('Min value in AMT_CREDIT:', df_app_train['AMT_CREDIT'].min())
print('Min value in AMT_ANNUITY:', df_app_train['AMT_ANNUITY'].min())
print('Min value in AMT_GOODS_PRICE:', df_app_train['AMT_GOODS_PRICE'].min())
print('Max values in DAYS_EMPLOYED less than 350k:', df_app_train[df_app_train['DAYS_EMPLOYED'] < 3.5*10**5].DAYS_EMPLOYED.max())
print('Min value in OBS_30_CNT_SOCIAL_CIRCLE:', df_app_train['OBS_30_CNT_SOCIAL_CIRCLE'].min())
print('Min value in DEF_30_CNT_SOCIAL_CIRCLE:', df_app_train['DEF_30_CNT_SOCIAL_CIRCLE'].min())
print('Min value in OBS_60_CNT_SOCIAL_CIRCLE:', df_app_train['OBS_60_CNT_SOCIAL_CIRCLE'].min())
print('Min value in DEF_60_CNT_SOCIAL_CIRCLE:', df_app_train['DEF_60_CNT_SOCIAL_CIRCLE'].min())
df_app_train[df_app_train['DAYS_EMPLOYED'] < 3.5*10**5].DAYS_EMPLOYED.plot(kind='hist', bins=20)
plt.title('Distribution of DAYS_EMPLOYED less than 350k')
plt.show()
for col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT','AMT_ANNUITY', 'AMT_GOODS_PRICE']:  
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    ax = ax.flatten()
    
    df_app_train[col].plot(kind='hist', bins=100, ax=ax[0])
    ax[0].set_title('Original ' + col)
    
    try:
        df_app_train[col].apply(np.log).plot(kind='hist', color='r', bins=75, ax=ax[1])
        ax[1].set_title('Log-transformed ' + col)
    except ValueError:
        print('Feature includes zero(s):' + col)
        
    plt.show()
df_pos_bal.head()
df_desc[df_desc.Table == 'POS_CASH_balance.csv'][['Row', 'Description', 'Special']]
df_pos_bal[df_pos_bal.SK_DPD_DEF != 0].sort_values(by=['SK_ID_CURR', 'MONTHS_BALANCE']).head()
df_pos_bal[df_pos_bal.SK_ID_CURR == 182943].sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE']).head(20)
class PreProcessor(BaseEstimator, TransformerMixin):
    
    # Transformer which currently only thresholds specified features 
    
    def __init__(self, thresh_feature_names: list, thresholds: list):
        # Each feature has one (upper) threshold
        self.thresh_feature_names = thresh_feature_names
        self.thresholds = thresholds
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X_t = X.copy()
        
        for col, thresh in zip(self.thresh_feature_names, self.thresholds):
            X_t.loc[X_t[col] > thresh, col] = np.nan
        
        return X_t
    
class CatEncoderNan(BaseEstimator, TransformerMixin):
    
    # Encode (nominal) categorical features while ignoring NaNs
    
    def __init__(self, features: list):
        self.features = features
        self.encoder_dict = dict()
        for feat in features:
            self.encoder_dict[feat] = LabelEncoder()
    
    def fit(self, X, y=None):
        for feat in self.features:
            self.encoder_dict[feat].fit(X[~X[feat].isna()][feat])
        return self
    
    def transform(self, X, y=None):
        X_t = X.copy()
        for feat in self.features:
            X_t.loc[~X[feat].isna(), feat] = self.encoder_dict[feat].transform(X[~X[feat].isna()][feat])

        return X_t
    
class FeatureGeneratorApplication(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X['CREDIT_LENGTH'] = X['AMT_CREDIT'] / X['AMT_ANNUITY']
        
        return X
thresh_feats = ['DAYS_EMPLOYED']
threshs = [0]
log_transform_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']

col_tf = ColumnTransformer(
    [
        ('num_log_tf', FunctionTransformer(func=np.log), log_transform_cols),
        ('cat_encoder', CatEncoderNan(cat_cols), cat_cols)
    ],
    remainder='passthrough'
)

feature_pipeline = Pipeline([
    ('preprocessor', PreProcessor(thresh_feats, threshs)),
    ('feature_generator', FeatureGeneratorApplication()),
    ('col_tf', col_tf)
])

output_feature_names = log_transform_cols + cat_cols + [col for col in df_app_train if col not in (log_transform_cols + cat_cols)] + \
    ['CREDIT_LENGTH']
df_train = pd.DataFrame(feature_pipeline.fit_transform(df_app_train), columns=output_feature_names, dtype='float32')
df_train[cat_cols] = df_train[cat_cols].astype('category')
df_train.head()
# X_train = df_train.drop(['SK_ID_CURR', 'TARGET'], axis=1)
# y_train = df_train.TARGET
# %%time

# num_iters = 3
# num_splits = 5

# param_space = {
#     'max_depth': list(range(2, 63)) + [None],
#     'num_leaves': range(7, 4096),
#     'subsample': [0.4, 1],
#     'colsample_bytree': [0.4, 1],
#     'reg_lambda': [0, 1 , 2],
#     'scale_pos_weight': [1, 6, 12]
# }

# max_depth = random.choices(param_space['max_depth'], k=num_iters)
# num_leaves = random.choices(param_space['num_leaves'], k=num_iters)
# subsample = random.choices(param_space['subsample'], k=num_iters)
# colsample_by_tree = random.choices(param_space['colsample_bytree'], k=num_iters)
# reg_lambda = random.choices(param_space['reg_lambda'], k=num_iters)
# scale_pos_weight = random.choices(param_space['scale_pos_weight'], k=num_iters)
    
# cv = KFold(n_splits=num_splits, shuffle=True, random_state=random_seed)

# scores_per_iter = list()
# for i in range(num_iters):
#     scores_per_iter.append(list())
    
# for train_index, valid_index in cv.split(X_train):

#     X_train_cv, X_valid_cv = X_train.iloc[train_index], X_train.iloc[valid_index]
#     y_train_cv, y_valid_cv = y_train[train_index], y_train[valid_index]
    
#     for i in range(num_iters):
        
#         train_set_cv = lgb.Dataset(X_train_cv, y_train_cv, categorical_feature=cat_cols)
#         valid_set_cv = lgb.Dataset(X_valid_cv, y_valid_cv, categorical_feature=cat_cols)
        
#         estimator = lgb.train(params={'metric': 'auc',
#                                       'num_iterations': 99999,
#                                       'max_depth': max_depth[i],
#                                       'num_leaves': num_leaves[i],
#                                       'subsample': subsample[i],
#                                       'colsample_by_tree': colsample_by_tree[i],
#                                       'reg_lambda': reg_lambda[i],
#                                       'scale_pos_weight': scale_pos_weight[i]},
#                               train_set=train_set_cv,
#                               valid_sets=[train_set_cv, valid_set_cv],
#                               valid_names=['training', 'validation'],
#                               early_stopping_rounds=20,
#                               verbose_eval=100)
        
#         auc_valid_score = estimator.best_score['validation']['auc']
#         scores_per_iter[i].append(auc_valid_score)
# mean_scores = list()
# std_scores = list()
# for i in range(num_iters):
#     mean_scores.append(np.mean(scores_per_iter[i]))
#     std_scores.append(np.std(scores_per_iter[i]))
# mean_scores
# std_scores
df_train[cat_cols] = df_train[cat_cols].astype('category')

X_train = df_train.drop(['SK_ID_CURR', 'TARGET'], axis=1)
y_train = df_train.TARGET
num_iters = 10
num_splits = 5

param_space = {
    'n_estimators': range(50, 300, 50),
    'max_depth': range(3, 9, 2),
    'subsample': [0.4, 1],
    'colsample_bytree': [0.4, 1],
    'reg_lambda': [0, 1 , 2],
    'scale_pos_weight': [1, 6, 12]
}

cv = KFold(n_splits=num_splits, shuffle=True, random_state=random_seed)

estimator = lgb.LGBMClassifier(random_state=random_seed)

random_search_cv = RandomizedSearchCV(estimator=estimator,
                                      param_distributions=param_space,
                                      n_iter=num_iters,
                                      scoring={'f1_score': make_scorer(f1_score),
                                               'roc_auc': make_scorer(roc_auc_score)},
                                      n_jobs=-1,
                                      cv=cv,
                                      refit='roc_auc',
                                      verbose=20,
                                      random_state=random_seed,
                                      return_train_score=True)
%%time

random_search_cv.fit(X_train, y_train)
random_search_cv.best_params_
random_search_cv.best_score_
X_train, X_valid, y_train, y_valid  = train_test_split(df_train.drop(['SK_ID_CURR', 'TARGET'], axis=1),
                                                       df_train.TARGET,
                                                       test_size=0.2,
                                                       random_state=random_seed,
                                                       shuffle=True)
random_search_cv.best_params_['n_estimators'] = 9999

estimator = lgb.LGBMClassifier(objective='binary',
                               **random_search_cv.best_params_,
                               random_state=random_seed)
estimator.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_valid, y_valid)],
              eval_names=['train', 'valid'],
              eval_metric='auc',
              verbose=25,
              callbacks=[lgb.early_stopping(stopping_rounds=200,
                                            first_metric_only=True)])
class_thresh = 0.5

y_pred_proba = estimator.predict_proba(X_valid)[:, 1]

prec, rec, _ = precision_recall_curve(y_valid, y_pred_proba)
fpr, tpr, _ = roc_curve(y_valid, y_pred_proba)

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax = ax.flatten()

ax[0].plot(rec, prec)
ax[0].set_title('Average precision on validation set: ' + str(round(average_precision_score(y_valid, y_pred_proba), 2)),
                fontsize=11, weight='bold')
ax[0].set_xlabel('Recall')
ax[0].set_ylabel('Precision')
ax[0].grid()

ax[1].plot(fpr, tpr)
ax[1].set_title('ROC-AUC on validation set: ' + str(round(roc_auc_score(y_valid, y_pred_proba), 2)),
                fontsize=11, weight='bold')
ax[1].set_xlabel('False positive rate')
ax[1].set_ylabel('True positive rate')
ax[1].grid()

cm = confusion_matrix(y_valid, [1 if pred >= class_thresh else 0 for pred in y_pred_proba])
sns.heatmap(cm, ax=ax[2], cmap='cividis', annot=True, fmt='d', annot_kws={'size': 11, 'weight': 'bold'}, cbar=False)
ax[2].set_title('Confusion matrix, threshold: ' + str(100 * class_thresh) + '%', fontsize=11, weight='bold')
plt.yticks(rotation=0)
ax[2].set_xlabel('Predicted')
ax[2].set_ylabel('Truth', rotation=0)

plt.show()