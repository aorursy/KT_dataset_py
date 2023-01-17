import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import r2_score, make_scorer
import lightgbm as lgbm
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
print('Train size:\t#rows = {}, #cols = {}\nTest size:\t#rows = {}, #cols = {}'\
      .format(train_df.shape[0], train_df.shape[1], test_df.shape[0], test_df.shape[1]))
train_df['train_set'] = 1
test_df['train_set'] = 0
train_ids = train_df.pop('Id')
test_ids = test_df.pop('Id')
y = train_df.pop('SalePrice')
df = train_df.append(test_df).reset_index(drop=True)
df.head()
cols_with_nulls, null_perc = [], []
for col in df.columns:
    if sum(df[col].isnull()) > 0:
        cols_with_nulls.append(col)
        null_perc.append(100 * sum(df[col].isnull()) / len(df))
null_counts = pd.DataFrame({'column': cols_with_nulls, 'null_perc': null_perc}).sort_values(by='null_perc',
                                                                                            ascending=False)
fig, ax = plt.subplots(1, 1, figsize=[10, 10])
sns.barplot(x='null_perc', y='column', data=null_counts, ax=ax)
ax.set_xlabel('Null percentage (%)')
ax.set_ylabel('')
plt.show()
null_threshold = .5
for col in df.columns:
    if (sum(df[col].isnull()) / len(df) > null_threshold) & (col != 'SalePrice'):
        df.drop(col, axis=1, inplace=True)
df.dtypes.value_counts().plot(kind='barh')
plt.title('Data types present with counts')
plt.show()
numerical_cols = df.columns[df.dtypes != 'object'].tolist()
print(numerical_cols)
ordinal_numericals = ['BsmtFullBath', 'BsmtHalfBath', 'Fireplaces', 'FullBath', 'GarageCars', 
                      'HalfBath', 'MoSold', 'OverallCond', 'OverallQual', 'TotRmsAbvGrd']
date_numericals = ['GarageYrBlt', 'YearBuilt', 'YearRemodAdd', 'YrSold']
categorical_numericals = ['MSSubClass']
continuous_feats = [col for col in numerical_cols if col not in ordinal_numericals and\
                                                     col not in date_numericals and\
                                                     col not in categorical_numericals and\
                                                     col != 'SalePrice' and col != 'train_set']
df[continuous_feats].describe()
non_numerical_cols = df.columns[df.dtypes == 'object'].tolist()
print(non_numerical_cols)
categorical_strings = ['BldgType', 'Condition1', 'Condition2', 'Electrical', 'Exterior1st', 
                       'Exterior2nd', 'Foundation', 'GarageType', 'Heating', 'HouseStyle', 
                       'LandContour', 'LotConfig', 'LotShape', 'MSZoning', 'MasVnrType', 
                       'Neighborhood', 'PavedDrive', 'RoofMatl', 'RoofStyle', 'SaleCondition',
                       'SaleType', 'Street', 'Utilities']
ordinal_strings = ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'ExterCond',
                  'ExterQual', 'FireplaceQu', 'Functional',  'GarageCond', 'GarageFinish', 'GarageQual',
                   'HeatingQC', 'KitchenQual', 'LandSlope']
boolean_strings = ['CentralAir']
all_categoricals = categorical_numericals + categorical_strings
df[all_categoricals] = df[all_categoricals].fillna('NA')

all_ordinals = ordinal_numericals + ordinal_strings
mode_df = pd.DataFrame(data=df[all_ordinals].mode().values.tolist()[0], index=all_ordinals)
df[all_ordinals] = df[all_ordinals].fillna(mode_df[0])

df[continuous_feats] = df[continuous_feats].fillna(df[continuous_feats].median())
df[date_numericals] = df[date_numericals].fillna(df[date_numericals].median())
print('Shape before one-hot encoding of categoricals =\t{}'.format(df.shape))
df = pd.get_dummies(df, columns=all_categoricals)
print('Shape after one-hot encoding of categoricals =\t{}'.format(df.shape))
ordinal_dict_1 = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
ordinal_dict_2 = {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
ordinal_dict_3 = {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
ordinal_dict_4 = {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7}
ordinal_dict_5 = {'Na': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
ordinal_dict_6 = {'Sev': 0, 'Mod': 1, 'Gtl': 2}
for cat in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu',
            'GarageQual', 'GarageCond', ]:
    df[cat] = df[cat].map(ordinal_dict_1)
    
for cat in ['BsmtExposure']:
    df[cat] = df[cat].map(ordinal_dict_2)
    
for cat in ['BsmtFinType1', 'BsmtFinType2']:
    df[cat] = df[cat].map(ordinal_dict_3)
    
for cat in ['Functional']:
    df[cat] = df[cat].map(ordinal_dict_4)
    
for cat in ['GarageFinish']:
    df[cat] = df[cat].map(ordinal_dict_5)
    
for cat in ['LandSlope']:
    df[cat] = df[cat].map(ordinal_dict_6)
    
# Check I haven't introduced more nulls
null_counts.set_index('column', inplace=True)
for col in ordinal_strings:
    if sum(df[col].isnull()) > 0:
        null_counts.loc[col, 'perc_after_encoding'] = 100 * sum(df[col].isnull()) / len(df)
        assert null_counts.loc[col, 'perc_after_encoding'] == null_counts.loc[col, 'null_perc'],\
        'Nulls introduced in feature {}, check corresponding dictionary'.format(col)
for cat in boolean_strings:
    encoder = LabelEncoder()
    df[cat] = encoder.fit_transform(df[cat])
df.head()
# Total prime living space - indoor, proper space
df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF']

# Total 'sub-prime' living space - outdoors + basements etc.
df['AltTotalSF'] = df['TotalBsmtSF'] + df['ScreenPorch'] + df['WoodDeckSF'] + df['OpenPorchSF'] + df['EnclosedPorch']\
                   + df['GarageArea'] + df['LowQualFinSF']
date_numericals.remove('YrSold')
for date_col in date_numericals:
    df[date_col] = df['YrSold'] - df[date_col]

# Create a Boolean of those properties that have had work done
df['RemodelledFlag'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)
# Guess the most important feature just to break the data out:
fig, ax = plt.subplots(1, 1, figsize=[6, 4])
ax.scatter(df.loc[df['train_set'] == 1, 'TotalSF'], y)
ax.set_xlabel('Total area (feet^2)')
ax.set_ylabel('Sale price ($)')
plt.show()
outlier_mask = (df['TotalSF'] > 4000) & (df['train_set'] == 1)
outlier_mask_target = outlier_mask[df['train_set'] == 1]
df = df[~outlier_mask]
y = y[~outlier_mask_target.values]
skew_array = skew(df[continuous_feats].fillna(df[continuous_feats].median()), axis=0)
skew_df = pd.DataFrame({'column': continuous_feats, 'skew': skew_array}).sort_values(by='skew', ascending=False)
fig, axs = plt.subplots(2, 2, figsize=[13, 8])
i = 0
for ax in axs.ravel():
    sns.distplot(df[skew_df.iloc[i, 0]], ax=ax)
    ax.set_title(skew_df.iloc[i, 0])
    i += 1
plt.tight_layout()
plt.show()
df[continuous_feats] = np.log1p(df[continuous_feats].fillna(df[continuous_feats].median()))
log_y = np.log1p(y)
new_skew_array = skew(df[continuous_feats].fillna(df[continuous_feats].median()), axis=0)
new_skew_df = pd.DataFrame({'column': continuous_feats, 'skew': new_skew_array})\
              .sort_values(by='skew', ascending=False)
fig, axs = plt.subplots(2, 2, figsize=[13, 8])
i = 0
for ax in axs.ravel():
    sns.distplot(df[skew_df.iloc[i, 0]], ax=ax)
    ax.set_title(skew_df.iloc[i, 0])
    i += 1
plt.tight_layout()
plt.show()
scaler = MinMaxScaler()
scaled_df = pd.DataFrame(data=scaler.fit_transform(df),
                         columns=df.columns)
scaled_df.head()
train_df = df[df['train_set'] == df['train_set'].max()].copy()
test_df = df[df['train_set'] == df['train_set'].min()].copy()
for dataset in [train_df, test_df]:
    dataset.drop('train_set', axis=1, inplace=True)
def rmsle(y_true, y_pred):
    assert len(y_pred) == len(y_true), 'Input arrays different lengths'
    return np.sqrt(np.mean(np.power(y_pred - y_true, 2)))
def rmsle_for_lgbm(preds, train_data):
    labels = train_data.get_label()
    return 'RMSLE', np.sqrt(np.mean(np.power(np.log1p(preds) - np.log1p(labels), 2))), False
rmsle_score = make_scorer(rmsle)
regressions_to_try = [LinearRegression(), Ridge(), Lasso(), ElasticNet()]
preds_list = []
for model in regressions_to_try:
    scores = cross_validate(model, train_df, log_y, cv=3, scoring=rmsle_score, return_train_score=False)
    preds_list.append(cross_val_predict(model, train_df, log_y, cv=3))
    print('{} RSMLE loss = {:.4f}'.format(type(model), scores['test_score'].mean()))
for alph in [1, 3, 6, 9, 12, 15]:
    test_model = Ridge(alpha=alph)
    scores = cross_validate(test_model, train_df, log_y, cv=3, scoring=rmsle_score, return_train_score=False)
    print('Alpha = {}, Ridge RSMLE loss = {:.4f}'.format(alph, scores['test_score'].mean()))
lin = Ridge(alpha=9)
lin.fit(train_df, log_y)
lin_preds = lin.predict(test_df)
fig, (ax, ax1) = plt.subplots(1, 2, figsize=[15,5])

ax.scatter(log_y, preds_list[1], label='Ridge predictions')
ax1.scatter(log_y, preds_list[2], label='Lasso predictions')

for axs in (ax, ax1):
    axs.plot([min(preds_list[1]), max(preds_list[1])], 
             [min(log_y), max(log_y)], label='y = x', color='k', linestyle='--')
    axs.set_xlabel('Prediction value ($)')
    axs.set_ylabel('Actual value ($)')
    axs.legend()

plt.show()
lgbm_model = lgbm.LGBMRegressor(
    boosting_type='gbdt',
    objective='huber',
    learning_rate=0.2,
    min_child_samples=30,
    colsample_bytree=0.9,
    max_depth=-1,
    num_leaves=31,
    reg_lambda=0,
    n_estimators=1000
)

skgbm_model = GradientBoostingRegressor(
    loss='huber',
    n_estimators=700,
    alpha=.5,
    max_depth=3,
    subsample=.6
)
ensembles_to_try = [lgbm_model, skgbm_model]
preds_list = []

for i in range(len(ensembles_to_try)):
    scores = cross_validate(ensembles_to_try[i], train_df, log_y, 
                            cv=3, 
                            scoring=rmsle_score, 
                            return_train_score=False)
    preds_list.append(cross_val_predict(ensembles_to_try[i], train_df, log_y, cv=3))
    print('{} RSMLE loss = {:.4f}'.format(type(ensembles_to_try[i]), scores['test_score'].mean()))
fig, (ax, ax1) = plt.subplots(1, 2, figsize=[15,5])

ax.scatter(log_y, preds_list[0], label='LGBM predictions', color='r')
ax1.scatter(log_y, preds_list[1], label='SKLearn GBM predictions', color='r')

for axs in (ax, ax1):
    axs.plot([min(preds_list[1]), max(preds_list[1])], 
             [min(log_y), max(log_y)], label='y = x', color='k', linestyle='--')
    axs.set_xlabel('Prediction value ($)')
    axs.set_ylabel('Actual value ($)')
    axs.legend()

plt.show()
gbm = skgbm_model.fit(train_df, log_y)
gbm_preds = gbm.predict(test_df)
fig, ax = plt.subplots(1, 1, figsize=[7,10])
features_importance = pd.DataFrame({'Feature': test_df.columns, 'Importance': gbm.feature_importances_})\
                      .sort_values(by='Importance', ascending=False)

top_n_feats_to_plot = 20
sns.barplot(x='Importance', y='Feature', data=features_importance[:top_n_feats_to_plot], orient='h', ax=ax)
plt.show()
linear_ratio = .6
averaged_preds = (1-linear_ratio)*gbm_preds + linear_ratio*lin_preds
output = pd.DataFrame({'Id': test_ids, 'SalePrice': np.expm1(averaged_preds)})
output.to_csv('submission.csv', index=False)
