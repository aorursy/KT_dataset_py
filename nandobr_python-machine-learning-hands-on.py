#%matplotlib inline
import pandas as pd
import numpy as np
df = pd.read_csv('../input/train.csv', parse_dates=[0])
test = pd.read_csv('../input/test.csv', parse_dates=[0])
df.shape
df.head()
df.head()
test.head()
df.info()
df['count'].hist(bins=20)
df['count'] = np.log(df['count'] + 1)
1000 - 1100
10 - 11
df['count'].hist(bins=20)
df.rename(columns={'count':'rentals'}, inplace=True)
df.shape
df = df.append(test,sort=False)
df.shape
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['dayofweek'] = df['datetime'].dt.dayofweek
df['hour'] = df['datetime'].dt.hour
df.sort_values('datetime', inplace=True)
#df.temp.describe()
df['rolling_temp'] = df['temp'].rolling(4, min_periods=1).mean()
df['rolling_temp'].describe()
df.shape
df.head().T
df.year.value_counts()
test = df[df['rentals'].isnull()]
df = df[~df['rentals'].isnull()]
from sklearn.model_selection import train_test_split
# divide dataset de treino em treino(70%) e validação(30%)
train, valid = train_test_split(df, random_state=42)
train.shape, valid.shape
print("%.0f" % round((valid.shape[0]/train.shape[0])*100,1)+ '%')
removed_cols = ['rentals', 'casual', 'registered', 'datetime']
feats = [c for c in df.columns if c not in removed_cols]
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=42, max_depth=2)
dt.fit(train[feats], train['rentals'])
from fastai.structured import draw_tree
draw_tree(dt, train[feats], precision=3, size=40)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf.fit(train[feats], train['rentals'])
preds = rf.predict(valid[feats])
from sklearn.metrics import mean_squared_error
mean_squared_error(valid['rentals'], preds)**(1/2)
train_preds = rf.predict(train[feats])
mean_squared_error(train['rentals'], train_preds)**(1/2)
test['count'] = np.exp(rf.predict(test[feats]))
test[['datetime', 'count']].to_csv('rf.csv', index=False)
rf = RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=200)
rf.fit(train[feats], train['rentals'])
preds = rf.predict(valid[feats])
mean_squared_error(valid['rentals'], preds)**(1/2)
train_preds = rf.predict(train[feats])
mean_squared_error(train['rentals'], train_preds)**(1/2)
test['count'] = np.exp(rf.predict(test[feats]))
test[['datetime', 'count']].to_csv('rf_opt.csv', index=False)
rf = RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=200, oob_score=True)
rf.fit(df[feats], df['rentals'])
mean_squared_error(df['rentals'], rf.oob_prediction_)**(1/2)
rf.oob_score_
from sklearn.metrics import r2_score
r2_score(df['rentals'], rf.oob_prediction_)
r2_score(df['rentals'], df['rentals'])
r2_score(df['rentals'], np.full(rf.oob_prediction_.shape[0], df['rentals'].mean()))
r2_score(df['rentals'], np.full(rf.oob_prediction_.shape[0], 10000))
test['count'] = np.exp(rf.predict(test[feats]))
test[['datetime', 'count']].to_csv('rf_full.csv', index=False)
# Calcula a posição relativa no Kaggle Public Leaderboard
729/3251
# POSIÇÃO: 729 , TOP 22% da competição.
#-----------------------------------------------------------------------------------------------------------------------
train.shape , valid.shape
train, valid = df[df['day'] <= 15], df[df['day'] > 15]
train.shape , valid.shape
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=200, min_samples_split=4, max_features=0.9, max_depth=17, oob_score=True)
rf.fit(train[feats], train['rentals'])
preds = rf.predict(valid[feats])
from sklearn.metrics import mean_squared_error
mean_squared_error(valid['rentals'], preds)**(1/2)
feats = [c for c in feats if c not in ['day']]
rf = RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=200, min_samples_split=4, max_features=0.9, max_depth=17, oob_score=True)
rf.fit(df[feats], df['rentals'])
preds = rf.predict(df[feats])
mean_squared_error(df['rentals'], rf.oob_prediction_)**(1/2)
train_preds = rf.predict(df[feats])
mean_squared_error(df['rentals'], train_preds)**(1/2)
rf.predict(test[feats])
test['count'] = np.exp(rf.predict(test[feats]))
test[['datetime', 'count']].to_csv('rf_full_wo_day.csv', index=False)
feats = [c for c in feats if c not in ['month', 'holiday']]
rf = RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=200, min_samples_split=4, max_features=0.9, max_depth=17, oob_score=True)
rf.fit(df[feats], df['rentals'])
preds = rf.predict(df[feats])
mean_squared_error(df['rentals'], rf.oob_prediction_)**(1/2)
train_preds = rf.predict(df[feats])
mean_squared_error(df['rentals'], train_preds)**(1/2)
rf.predict(test[feats])
test['count'] = np.exp(rf.predict(test[feats]))
test[['datetime', 'count']].to_csv('rf_full_wo_month.csv', index=False)
def cv(df, test, k, feats, y_name):
    preds, score, fis = [], [], []
    
    chunk = df.shape[0] // k
    for i in range(k):
        if i + 1 < k:
            valid = df.iloc[i*chunk: (i+1)*chunk]
            train = df.iloc[: i*chunk].append(df.iloc[(i+1)*chunk:])
            
        else:
            valid = df.iloc[i*chunk:]
            train = df.iloc[: i*chunk] 

        rf = RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=200, min_samples_split=4, max_features=0.9, max_depth=17, oob_score=True)
        
        rf.fit(train[feats], train[y_name])
        
        score.append(mean_squared_error(valid[y_name], rf.predict(valid[feats]))**(1/2))  
        
        preds.append(rf.predict(test[feats]))  
        
        fis.append(rf.feature_importances_)
        
        print(i, 'OK')
    return pd.Series(score), pd.Series(preds).mean(), fis
score, preds, fis = cv(df, test, 20, feats, 'rentals')
score.mean()
test['count'] = np.exp(preds)
test[['datetime', 'count']].to_csv('rf_cv_wo_day.csv', index=False)
fi = pd.Series(pd.DataFrame(fis).mean().values, index=feats)
fi.sort_values().plot.barh(figsize=(20,10))
feats = [c for c in feats if c not in ['holiday']]
score, preds, fis = cv(df, test, 20, feats, 'rentals')
score.mean()
test['count'] = np.exp(preds)
test[['datetime', 'count']].to_csv('rf_cv_wo_holiday.csv', index=False)
