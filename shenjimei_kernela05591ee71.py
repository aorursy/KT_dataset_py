import gc
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
market_train, news_train = env.get_training_data()
start = datetime(2013, 1, 1, 0, 0, 0).date()
market_train = market_train.loc[market_train['time'].dt.date >= start].reset_index(drop=True)
news_train = news_train.loc[news_train['time'].dt.date >= start].reset_index(drop=True)
market_train.head(5)
news_train.head(5)
def preprocess_news(news_train):
    drop_list = [
        'headline','sourceTimestamp','firstCreated','subjects','audiences','assetName'
    ]
    for col in ['headlineTag','provider','sourceId']:
        news_train[col], uniques = pd.factorize(news_train[col])
        del uniques
    news_train['assetCodes'] = news_train['assetCodes'].apply(lambda x: x[1:-1].replace("'", ""))
    return news_train
news_train = preprocess_news(news_train)
def unstack_asset_codes(news_train):
    codes = []
    indexes = []
    for i, values in news_train['assetCodes'].iteritems():
        explode = values.split(", ")
        codes.extend(explode)
        repeat_index = [int(i)]*len(explode)
        indexes.extend(repeat_index)
    index_df = pd.DataFrame({'news_index': indexes, 'assetCode': codes})
    del codes, indexes
    gc.collect()
    return index_df

index_df = unstack_asset_codes(news_train)
index_df.head()
def unstack_asset_codes(news_train):
    codes = []
    indexes = []
    for i, values in news_train['assetCodes'].iteritems():
        explode = values.split(", ")
        codes.extend(explode)
        repeat_index = [int(i)]*len(explode)
        indexes.extend(repeat_index)
    index_df = pd.DataFrame({'news_index': indexes, 'assetCode': codes})
    del codes, indexes
    gc.collect()
    return index_df

index_df = unstack_asset_codes(news_train)
index_df.head()
def merge_news_on_index(news_train, index_df):
    news_train['news_index'] = news_train.index.copy()

    # Merge news on unstacked assets
    news_unstack = index_df.merge(news_train, how='left', on='news_index')
    news_unstack.drop(['news_index', 'assetCodes'], axis=1, inplace=True)
    return news_unstack

news_unstack = merge_news_on_index(news_train, index_df)
del news_train, index_df
gc.collect()
news_unstack.head(5)
def group_news(news_frame):
    news_frame['date'] = news_frame.time.dt.date  # Add date column
    
    aggregations = ['mean']
    gp = news_frame.groupby(['assetCode', 'date']).agg(aggregations)
    gp.columns = pd.Index(["{}_{}".format(e[0], e[1]) for e in gp.columns.tolist()])
    gp.reset_index(inplace=True)
    # Set datatype to float32
    float_cols = {c: 'float32' for c in gp.columns if c not in ['assetCode', 'date']}
    return gp.astype(float_cols)

news_agg = group_news(news_unstack)
del news_unstack; gc.collect()
news_agg.head(5)
market_train['date'] = market_train.time.dt.date
df = market_train.merge(news_agg, how='left', on=['assetCode', 'date'])
del market_train, news_agg
gc.collect()
df.head(5)
def custom_metric(date, pred_proba, num_target, universe):
    y = pred_proba*2 - 1
    r = num_target.clip(-1,1) # get rid of outliers
    x = y * r * universe
    result = pd.DataFrame({'day' : date, 'x' : x})
    x_t = result.groupby('day').sum().values
    return np.mean(x_t) / np.std(x_t)
date = df.date
num_target = df.returnsOpenNextMktres10.astype('float32')
bin_target = (df.returnsOpenNextMktres10 >= 0).astype('int8')
universe = df.universe.astype('int8')
# Drop columns that are not features
df.drop(['returnsOpenNextMktres10', 'date', 'universe', 'assetCode', 'assetName', 'time'], 
        axis=1, inplace=True)
df = df.astype('float32')  # Set all remaining columns to float32 datatype
gc.collect()
train_index, test_index = train_test_split(df.index.values, test_size=0.3)
def evaluate_model(df, target, train_index, test_index, params):
    #model = XGBClassifier(**params)
    model = LGBMClassifier(**params)
    model.fit(df.iloc[train_index], target.iloc[train_index])
    return log_loss(target.iloc[test_index], model.predict_proba(df.iloc[test_index]))
# param_grid = {
#     'learning_rate': [0.1, 0.5, 0.02, 0.01],
#     'num_leaves': [15, 30, 40, 65],
#     'n_estimators': [20, 30, 50, 100, 200]
# }
# best_eval_score = 0
# for i in range(20):
#     params = {k: np.random.choice(v) for k, v in param_grid.items()}
#     score = evaluate_model(df, bin_target, train_index, test_index, params)
#     if score < best_eval_score or best_eval_score == 0:
#         best_eval_score = score
#         best_params = params
# print("Best evaluation logloss", best_eval_score)
df.head(5)
print(df.isnull().sum(axis=0))
# Checking feature correlations
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
corr = pd.concat([df, bin_target], axis=1).corr()
plt.figure(figsize=(14, 8))
plt.title('Overall Correlation of House Prices', fontsize=18)
sns.heatmap(corr, annot=False,cmap='RdYlGn', linewidths=0.2, annot_kws={'size':20})
plt.show()
print(df.columns)
# drop SourceId and after feature , if is no na ,let it be 1, otherwise be 0
def dropfeatureTooNa(df):
    columns = ['sourceId_mean', 'urgency_mean',
       'takeSequence_mean', 'provider_mean', 'bodySize_mean',
       'companyCount_mean', 'headlineTag_mean', 'marketCommentary_mean',
       'sentenceCount_mean', 'wordCount_mean', 'firstMentionSentence_mean',
       'relevance_mean', 'sentimentClass_mean', 'sentimentNegative_mean',
       'sentimentNeutral_mean', 'sentimentPositive_mean',
       'sentimentWordCount_mean', 'noveltyCount12H_mean',
       'noveltyCount24H_mean', 'noveltyCount3D_mean', 'noveltyCount5D_mean',
       'noveltyCount7D_mean', 'volumeCounts12H_mean', 'volumeCounts24H_mean',
       'volumeCounts3D_mean', 'volumeCounts5D_mean', 'volumeCounts7D_mean']
    new_df = df
    new_df['have_new_influence'] = 0
    new_df.loc[new_df['sourceId_mean'].notna(), 'have_new_influence'] = 1
    new_df = new_df.drop(columns, axis=1)
    return new_df
    
new_df = dropfeatureTooNa(df)
new_df.head(5)
# Checking feature correlations
import seaborn as sns
import matplotlib.pyplot as plt
corr = pd.concat([new_df, bin_target], axis=1).corr()
plt.figure(figsize=(14, 8))
plt.title('Overall Correlation of House Prices', fontsize=18)
sns.heatmap(corr, annot=False,cmap='RdYlGn', linewidths=0.2, annot_kws={'size':20})
plt.show()
print(new_df.isnull().sum(axis=0))
# filter missing data by median
new_df['returnsClosePrevMktres1'] = new_df['returnsClosePrevMktres1'].fillna(new_df['returnsClosePrevMktres1'].dropna().median())
new_df['returnsOpenPrevMktres1'] = new_df['returnsOpenPrevMktres1'].fillna(new_df['returnsOpenPrevMktres1'].dropna().median())
new_df['returnsClosePrevMktres10'] = new_df['returnsClosePrevMktres10'].fillna(new_df['returnsClosePrevMktres10'].dropna().median())
new_df['returnsOpenPrevMktres10'] = new_df['returnsOpenPrevMktres10'].fillna(new_df['returnsOpenPrevMktres10'].dropna().median())
print(new_df.isnull().sum(axis=0) > 0)
# define 
from sklearn.cross_validation import cross_val_score
def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, new_df, bin_target, scoring='neg_mean_squared_error', cv=10))
    return rmse
from sklearn.linear_model import Ridge
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 70, 90, 100, 500, 1000, 2000]
cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean() for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation")
plt.xlabel("Alpha")
plt.ylabel("Rmse")
# 500 looks like the optimal alpha level, so let's fit the Ridge model with this value
model_ridge = Ridge(alpha=500)
