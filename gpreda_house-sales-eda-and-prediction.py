import pandas as pd
import numpy as np
import sys
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
PATH="../input/"
os.listdir(PATH)
train_df = pd.read_csv(PATH+"train.csv")
test_df = pd.read_csv(PATH+"test.csv")
print("Train: rows:{} cols:{}".format(train_df.shape[0], train_df.shape[1]))
print("Test:  rows:{} cols:{}".format(test_df.shape[0], test_df.shape[1]))
train_df.head()
test_df.head()
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return np.transpose(pd.concat([total, percent], axis=1, keys=['Total', 'Percent']))
missing_data(train_df)
missing_data(test_df)
train_df.info()
train_df.describe()
categorical_columns = ['waterfront', 'view', 'condition', 'grade']
i = 0
plt.figure()
fig, ax = plt.subplots(2,2,figsize=(16,10))
for col in categorical_columns:
    i += 1
    plt.subplot(2,2,i)
    sns.boxplot(x=train_df[col],y=train_df['price'])
    plt.xlabel(col, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();
def plot_stats(feature):
    temp = train_df[feature].dropna().value_counts().head(50)
    df1 = pd.DataFrame({feature: temp.index,'Number of samples': temp.values})
    temp = test_df[feature].dropna().value_counts().head(50)
    df2 = pd.DataFrame({feature: temp.index,'Number of samples': temp.values})    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6))
    s = sns.barplot(x=feature,y='Number of samples',data=df1, ax=ax1)
    s.set_title("Train set")
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    s = sns.barplot(x=feature,y='Number of samples',data=df2, ax=ax2)
    s.set_title("Test set")
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.show()    
categorical_columns = ['waterfront', 'view', 'condition', 'grade']

for col in categorical_columns:
    plot_stats(col)
numerical_columns = ['bedrooms', 'bathrooms', 'floors']
i = 0
plt.figure()
fig, ax = plt.subplots(1,3,figsize=(18,4))
for col in numerical_columns:
    i += 1
    plt.subplot(1,3,i)
    sns.boxplot(x=train_df[col],y=train_df['price'])
    plt.xlabel(col, fontsize=8)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=8)
plt.show();
area_columns = ['sqft_living','sqft_lot','sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']

i = 0
plt.figure()
fig, ax = plt.subplots(3,2,figsize=(16,15))
for col in area_columns:
    i += 1
    plt.subplot(3,2,i)
    plt.scatter(x=train_df[col],y=train_df['price'],c='magenta', alpha=0.2)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('price', fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();
geo_columns = ['lat','long']

i = 0
plt.figure()
fig, ax = plt.subplots(1,2,figsize=(16,6))
for col in geo_columns:
    i += 1
    plt.subplot(1,2,i)
    plt.scatter(x=train_df[col],y=train_df['price'], c=train_df['zipcode'], alpha=0.2)
    plt.xlabel(col, fontsize=10)
    plt.ylabel('price', fontsize=10)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=10)
plt.show();
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure(figsize=(16,16))
ax=fig.add_subplot(1,1,1, projection="3d")
ax.scatter(train_df['lat'],train_df['long'],train_df['price'],c=train_df['zipcode'],alpha=.8)
ax.set(xlabel='\nLatitude',ylabel='\nLongitude',zlabel='\nPrice')
print("There are {} unique zipcodes.".format(train_df['zipcode'].nunique()))
plt.figure(figsize=(18,4))
sns.boxplot(x=train_df['zipcode'],y=train_df['price'])
plt.xlabel('zipcode', fontsize=8)
locs, labels = plt.xticks()
plt.tick_params(axis='x', labelsize=8, rotation=90)
plt.show();
plt.figure(figsize=(16,16))
plt.scatter(x=train_df['lat'],y=train_df['long'], c=train_df['zipcode'], cmap='Spectral')
plt.xlabel('lat', fontsize=12); plt.ylabel('long', fontsize=12)
plt.show();
for df in [train_df, test_df]:
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.weekofyear
    df['dayofyear'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    df['is_month_start'] = pd.to_numeric(df['date'].dt.is_month_start)
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['is_weekend'] = pd.to_numeric(df['dayofweek']>=5)
for df in [train_df, test_df]:
    df['med_lat'] = np.round(df['lat'],1) 
    df['med_long'] = np.round(df['long'],1) 
    df['build_old'] = 2019 - df['yr_built']
    df['sqft_living_diff'] = df['sqft_living'] - df['sqft_living15']
    df['sqft_lot_diff'] = df['sqft_lot'] - df['sqft_lot15']
    df['bedroom_bathroom_ratio'] = df['bedrooms'] / df['bathrooms']
train_df.head()
date_columns = ['year', 'month', 'dayofweek', 'quarter']
i = 0
plt.figure()
fig, ax = plt.subplots(2,2,figsize=(12,12))
for col in date_columns:
    i += 1
    plt.subplot(2,2,i)
    sns.boxplot(x=train_df[col],y=train_df['price'],showfliers=False)
    plt.xlabel(col, fontsize=8)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=8)
plt.show();
date_columns = ['year', 'month', 'dayofweek', 'quarter']
i = 0
plt.figure()
fig, ax = plt.subplots(2,2,figsize=(16,12))
for col in date_columns:
    i += 1
    plt.subplot(2,2,i)
    sns.boxplot(x=train_df[col],y=train_df['price'],showfliers=True)
    plt.xlabel(col, fontsize=8)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=8)
plt.show();
date_columns = ['dayofyear', 'weekofyear']
i = 0
plt.figure()
fig, ax = plt.subplots(2,1,figsize=(18,12))
for col in date_columns:
    i += 1
    plt.subplot(2,1,i)
    sns.boxplot(x=train_df[col],y=train_df['price'],showfliers=False)
    plt.xlabel(col, fontsize=8)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=8)
plt.show();
date_columns = ['dayofyear', 'weekofyear']
i = 0
plt.figure()
fig, ax = plt.subplots(2,1,figsize=(18,12))
for col in date_columns:
    i += 1
    plt.subplot(2,1,i)
    sns.boxplot(x=train_df[col],y=train_df['price'],showfliers=True)
    plt.xlabel(col, fontsize=8)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=8)
plt.show();
date_columns = ['is_month_start', 'is_weekend']
i = 0
plt.figure()
fig, ax = plt.subplots(1,2,figsize=(12,6))
for col in date_columns:
    i += 1
    plt.subplot(1,2,i)
    sns.boxplot(x=train_df[col],y=train_df['price'],showfliers=False)
    plt.xlabel(col, fontsize=8)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=8)
plt.show();
date_columns = ['is_month_start', 'is_weekend']
i = 0
plt.figure()
fig, ax = plt.subplots(1,2,figsize=(12,6))
for col in date_columns:
    i += 1
    plt.subplot(1,2,i)
    sns.boxplot(x=train_df[col],y=train_df['price'],showfliers=True)
    plt.xlabel(col, fontsize=8)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=8)
plt.show();
features = ['bedrooms','bathrooms','floors',
            'waterfront','view','condition','grade',
            'sqft_living','sqft_lot','sqft_above','sqft_basement','sqft_living15','sqft_lot15',
            'yr_built','yr_renovated',
            'lat', 'long','zipcode', 
            'date', 'dayofweek', 'weekofyear', 'dayofyear', 'quarter', 
            'is_month_start', 'month', 'year', 'is_weekend',
            'price']

mask = np.zeros_like(train_df[features].corr(), dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(18,18))
plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(train_df[features].corr(),linewidths=0.25,vmax=1.0,square=True,cmap="Blues", 
            linecolor='w',annot=True,mask=mask,cbar_kws={"shrink": .75});

#We are using 80-20 split for train-test
VALID_SIZE = 0.2
#We also use random state for reproducibility
RANDOM_STATE = 2019

train, valid = train_test_split(train_df, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True )
predictors = ['sqft_living', 'grade']
target = 'price'
train_X = train[predictors]
train_Y = train[target].values
valid_X = valid[predictors]
valid_Y = valid[target].values
RFC_METRIC = 'mse'  #metric used for RandomForrestClassifier
NUM_ESTIMATORS = 100 #number of estimators used for RandomForrestClassifier
NO_JOBS = 4 #number of parallel jobs used for RandomForrestClassifier
model = RandomForestRegressor(n_jobs=NO_JOBS, 
                             random_state=RANDOM_STATE,
                             criterion=RFC_METRIC,
                             n_estimators=NUM_ESTIMATORS,
                             verbose=False)
model.fit(train_X, train_Y)
preds = model.predict(valid_X)
def plot_feature_importance():
    tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': model.feature_importances_})
    tmp = tmp.sort_values(by='Feature importance',ascending=False)
    plt.figure(figsize = (7,4))
    plt.title('Features importance',fontsize=14)
    s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.show()   
plot_feature_importance()
print("RF Model score: ", model.score(train_X, train_Y))
def rmse(preds, y):
    return np.sqrt(mean_squared_error(preds, y))
print("Root mean squared error (valid set):",round(rmse(preds, valid_Y),2))
predictors = ['sqft_living', 'grade', 'sqft_above']
target = 'price'
train_X = train[predictors]
train_Y = train[target].values
valid_X = valid[predictors]
valid_Y = valid[target].values
model = RandomForestRegressor(n_jobs=NO_JOBS, 
                             random_state=RANDOM_STATE,
                             criterion=RFC_METRIC,
                             n_estimators=NUM_ESTIMATORS,
                             verbose=False)
model.fit(train_X, train_Y)
preds = model.predict(valid_X)
plot_feature_importance()
print("RF Model score: ", model.score(train_X, train_Y))
print("Root mean squared error (valid set):",round(rmse(preds, valid_Y),2))
predictors = ['sqft_living', 'sqft_lot',
              'sqft_above', 'sqft_living15',
              'waterfront', 'view', 'condition', 'grade',
             'bedrooms', 'bathrooms', 'floors',
             'zipcode', 
              'month', 'dayofweek', 
              'med_lat', 'med_long',
              'build_old', 'sqft_living_diff', 'sqft_lot_diff',
             ]
target = 'price'
train_X = train[predictors]
train_Y = train[target].values
valid_X = valid[predictors]
valid_Y = valid[target].values
model = RandomForestRegressor(n_jobs=NO_JOBS, 
                             random_state=RANDOM_STATE,
                             criterion=RFC_METRIC,
                             n_estimators=NUM_ESTIMATORS,
                             verbose=False)
model.fit(train_X, train_Y)
preds = model.predict(valid_X)
plot_feature_importance()
print("RF Model score: ", model.score(train_X, train_Y))
print("Root mean squared error (valid set):",round(rmse(preds, valid_Y),2))
test_X = test_df[predictors] 
predictions_RF = model.predict(test_X)
submission = pd.read_csv(PATH+"sample_submission.csv")
submission['price'] = predictions_RF
submission.to_csv('submission.csv', index=False)
param = {'num_leaves': 51,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "metric": 'rmse',
         "verbosity": -1,
         "nthread": 4,
         "random_state": 42}
predictors = ['sqft_living', 'sqft_lot',
              'sqft_above', 'sqft_living15',
              'waterfront', 'view', 'condition', 'grade',
             'bedrooms', 'bathrooms', 'floors',
             'zipcode', 
              'month', 'dayofweek', 
              'med_lat', 'med_long',
              'build_old', 'sqft_living_diff', 'sqft_lot_diff',
             ]
target = 'price'
#prepare fit model with cross-validation
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train_df))
predictions_lgb_cv = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df,train_df['price'].values)):
    strLog = "fold {}".format(fold_)
    print(strLog)
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][predictors], label=train_df.iloc[trn_idx][target])#, categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train_df.iloc[val_idx][predictors], label=train_df.iloc[val_idx][target])#, categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][predictors], num_iteration=clf.best_iteration)
    #feature importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = predictors
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    #predictions
    predictions_lgb_cv += clf.predict(test_df[predictors], num_iteration=clf.best_iteration) / folds.n_splits
    
strRMSE = "RMSE: {}".format(rmse(oof, train_df[target]))
print(strRMSE)
def plot_feature_importance_cv():
    cols = (feature_importance_df[["Feature", "importance"]]
            .groupby("Feature")
            .mean()
            .sort_values(by="importance", ascending=False).index)
    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

    plt.figure(figsize=(12,6))
    sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
    plt.title('LightGBM Features (averaged over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')
plot_feature_importance_cv()
submission = pd.read_csv(PATH+"sample_submission.csv")
submission['price'] = predictions_lgb_cv
submission.to_csv('submission_cv.csv', index=False)
predictions_blending = predictions_RF * 0.55 + predictions_lgb_cv * 0.45
submission = pd.read_csv(PATH+"sample_submission.csv")
submission['price'] = predictions_blending
submission.to_csv('submission_blending.csv', index=False)