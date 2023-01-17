import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train, news_train) = env.get_training_data()

# news_train['subjects'].nunique()
# news_train['provider'].nunique()
# print(news_train['audiences'].nunique())
# print(news_train['time'].nunique())
# print(news_train['sourceTimestamp'].nunique())
# print(news_train['firstCreated'].nunique())
print(market_train['time'].nunique())
# print(market_train['time'].head(10000))
# news_sub = pd.DataFrame()
# news_sub[['A','B','C']] = news_train[['assetName','sentimentNegative','sentimentNeutral','sentimentPositive']].groupby('assetName').count()
# news_sub[['a','b','c']] = news_train[['assetName','sentimentNegative','sentimentNeutral','sentimentPositive']].groupby('assetName').mean()
# market_train = market_train.join(news_sub,on='assetName',how='left')
cat_cols = ['assetCode']
num_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
                    'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
                    'returnsOpenPrevMktres10']

from sklearn.model_selection import train_test_split

# market_train = market_train.loc[pd.to_datetime(market_train['time']) >= datetime(2009, 1, 1)]
market_train = market_train.loc[pd.to_datetime(market_train['time']) >= pd.to_datetime('2009-01-01').tz_localize('UTC')]

train_indices, val_indices = train_test_split(market_train.index.values,test_size=0.25, random_state=23)

def encode(encoder, x):
    len_encoder = len(encoder)
    try:
        id = encoder[x]
    except KeyError:
        id = len_encoder
    return id

encoders = [{} for cat in cat_cols]


for i, cat in enumerate(cat_cols):
    print('encoding %s ...' % cat, end=' ')
    encoders[i] = {l: id for id, l in enumerate(market_train.loc[train_indices, cat].astype(str).unique())}
    market_train[cat] = market_train[cat].astype(str).apply(lambda x: encode(encoders[i], x))
    print('Done')

embed_sizes = [len(encoder) + 1 for encoder in encoders] #+1 for possible unknown assets

from sklearn.preprocessing import StandardScaler
# print(market_train['A'].describe())
market_train[num_cols] = market_train[num_cols].fillna(0)
print('scaling numerical columns')

scaler = StandardScaler()

#col_mean = market_train[col].mean()
#market_train[col].fillna(col_mean, inplace=True)
scaler = StandardScaler()
from datetime import datetime
print(market_train['time'].dtypes)
market_train[num_cols] = scaler.fit_transform(market_train[num_cols])

def get_input(market_train, indices):
    X_num = market_train.loc[indices, num_cols].values
    X = {'num':X_num}
    for cat in cat_cols:
        X[cat] = market_train.loc[indices, cat_cols].values
    y = (market_train.loc[indices,'returnsOpenNextMktres10'] >= 0).values
    r = market_train.loc[indices,'returnsOpenNextMktres10'].values
    u = market_train.loc[indices, 'universe']
    d = market_train.loc[indices, 'time'].dt.date
    return X,y,r,u,d

# r, u and d are used to calculate the scoring metric
X_train,y_train,r_train,u_train,d_train = get_input(market_train, train_indices)
X_valid,y_valid,r_valid,u_valid,d_valid = get_input(market_train, val_indices)
import warnings
warnings.filterwarnings(action ='ignore',category = DeprecationWarning)
from xgboost import XGBClassifier

from functools import partial
from hyperopt import hp, fmin, tpe
algo = partial(tpe.suggest, n_startup_jobs=10)
def auto_turing(args):
    model = XGBClassifier(n_jobs = 4, n_estimators = args['n_estimators'],max_depth=6)
    model.fit(X_train['num'],y_train.astype(int))
    confidence_valid = model.predict(X_valid['num'])*2 -1
    score = accuracy_score(confidence_valid>0,y_valid)
    print(args,score)
    return -score
# space = {"n_estimators":hp.choice("n_estimators",range(10,50))}
# print(fmin)
# best = fmin(auto_turing, space, algo=algo,max_evals=30)
# print(best)

#单机xgb程序
# model = XGBClassifier(n_jobs = 4, n_estimators = 47,max_depth=6)
# model.fit(X_train['num'],y_train.astype(int))
# confidence_valid = model.predict(X_valid['num'])*2 -1
# score = accuracy_score(confidence_valid>0,y_valid)
# print(score)


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None,min_samples_split=2, min_samples_leaf=1, 
                               min_weight_fraction_leaf=0.0,max_features='auto',max_leaf_nodes=None,min_impurity_split=1e-07, 
                               bootstrap=True,  oob_score=False, n_jobs=1,random_state=None, verbose=0,warm_start=False, class_weight=None)
model.fit(X_train['num'], y_train.astype(int))
confidence_valid = model.predict(X_valid['num'])*2 -1
score = accuracy_score(confidence_valid>0,y_valid)
print(score)

# distribution of confidence that will be used as submission
# plt.hist(confidence_valid, bins='auto')
# plt.title("predicted confidence")
# plt.show()

# calculation of actual metric that is used to calculate final score
r_valid = r_valid.clip(-1,1) # get rid of outliers. Where do they come from??
x_t_i = confidence_valid * r_valid * u_valid
data = {'day' : d_valid, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print(score_valid)
days = env.get_prediction_days()
n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
predicted_confidences = np.array([])
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    print(n_days,end=' ')
    
    t = time.time()

    market_obs_df['assetCode_encoded'] = market_obs_df[cat].astype(str).apply(lambda x: encode(encoders[i], x))

    market_obs_df[num_cols] = market_obs_df[num_cols].fillna(0)
    market_obs_df[num_cols] = scaler.transform(market_obs_df[num_cols])
    X_num_test = market_obs_df[num_cols].values
    X_test = {'num':X_num_test}
    X_test['assetCode'] = market_obs_df['assetCode_encoded'].values
    
    prep_time += time.time() - t
    
    t = time.time()
    market_prediction = model.predict(X_test['num'])*2 -1
    #print(type(market_prediction))
    #print(type(predicted_confidences))
    predicted_confidences = np.concatenate((predicted_confidences, market_prediction))
    prediction_time += time.time() -t
    
    t = time.time()
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':market_prediction})
    # insert predictions to template
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
    packaging_time += time.time() - t

env.write_submission_file()
total = prep_time + prediction_time + packaging_time
print(f'Preparing Data: {prep_time:.2f}s')
print(f'Making Predictions: {prediction_time:.2f}s')
print(f'Packing: {packaging_time:.2f}s')
print(f'Total: {total:.2f}s')
# distribution of confidence as a sanity check: they should be distributed as above
plt.hist(predicted_confidences, bins='auto')
plt.title("predicted confidence")
plt.show()