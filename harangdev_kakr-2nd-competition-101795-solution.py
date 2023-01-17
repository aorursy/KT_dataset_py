"""

import gc

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans



today = pd.to_datetime('2016-01-01')



tr = pd.read_csv('../input/train.csv').set_index('id')

te = pd.read_csv('../input/test.csv').set_index('id')

df = pd.concat([tr, te])

df['price_per_living'] = np.log1p(df['price']/df['sqft_living'])

df = df.drop('price', axis=1)

tr_idx = tr.index.tolist()

te_idx = te.index.tolist()

del tr, te; gc.collect();



df['was_renovated'] = (df['yr_renovated'] != 0).astype('uint8')

not_renovated = df[df['was_renovated'] == 0].index

df.loc[not_renovated, 'yr_renovated'] = df.loc[not_renovated, 'yr_built']

df['date'] = pd.to_datetime(df['date'].str[:8])

df['yr_built'] = pd.to_datetime({'year': df['yr_built'], 'month': [1]*len(df), 'day': [1]*len(df)})

df['yr_renovated'] = pd.to_datetime({'year': df['yr_renovated'], 'month': [1]*len(df), 'day': [1]*len(df)}, errors='coerce')



df['today-D-date'] = (today - df['date']).dt.days

df['today-D-yr_renovated'] = (today - df['yr_renovated']).dt.days

df['today-D-yr_built'] = (today - df['yr_built']).dt.days

df['date-D-yr_built'] = (df['date'] - df['yr_built']).dt.days

df['yr_renovated-D-yr_built'] = (df['yr_renovated'] - df['yr_built']).dt.days

df = df.drop(['date', 'yr_built', 'yr_renovated'], axis=1)



df['room_count'] = df['bedrooms'] + df['bathrooms']

df['sqft_living_per_rooms'] = df['sqft_living'] / (df['room_count']+1)

df['sqft_lot_per_rooms'] = df['sqft_lot'] / (df['room_count']+1)

df['room_per_floors'] = df['room_count'] / df['floors']



df['sqft_living_per_floors'] = df['sqft_living'] / df['floors']

df['sqft_lot_per_floors'] = df['sqft_lot'] / df['floors']



df['sqft_living_per_bedrooms'] = df['sqft_living'] / (df['bedrooms']+1)

df['sqft_lot_per_bedrooms'] = df['sqft_lot'] / (df['bedrooms']+1)



df['bedroom_per_floors'] = df['bedrooms'] / df['floors']



df['sqft_lot-D-sqft_living'] = df['sqft_lot'] - df['sqft_living']

df['sqft_lot-R-sqft_living'] = df['sqft_lot'] / df['sqft_living']



df['sqft_living15-D-sqft_living'] = df['sqft_living15'] - df['sqft_living']

df['sqft_living15-R-sqft_living'] = df['sqft_living15'] / df['sqft_living']



df['sqft_lot15-D-sqft_lot'] = df['sqft_lot15'] - df['sqft_lot']

df['sqft_lot15-R-sqft_lot'] = df['sqft_lot15'] / df['sqft_lot']



df['rooms_mul']=df['bedrooms']*df['bathrooms']

df['total_score']=df['condition']+df['grade']+df['view']



df['has_basement'] = (df['sqft_basement']>0).astype('uint8')

df['has_attic'] = ((df['floors'] % 1) != 0).astype('uint8')



for k in range(4, 60, 4):

    km = KMeans(32, n_jobs=8)

    df['clustering_'+str(k)] = km.fit_predict(df[['lat', 'long']])

    df['clustering_'+str(k)] = df['clustering_'+str(k)].astype(str)

    

df['zipcode'] = df['zipcode'].astype('str')

df['zipcode-3'] = df['zipcode'].str[2:3]

df['zipcode-4'] = df['zipcode'].str[3:4]

df['zipcode-5'] = df['zipcode'].str[4:5]

df['zipcode-34'] = df['zipcode'].str[2:4]

df['zipcode-45'] = df['zipcode'].str[3:5]

df['zipcode-35'] = df['zipcode'].str[2:3] + df['zipcode'].str[4:5]



cols =  ['zipcode']

for col in cols:

    val_count = df[col].value_counts()

    agg_cols = ['price_per_living']

    temp = df.groupby(col)[agg_cols].agg('mean').rename({k: str(col)+'_mean_'+str(k) for k in agg_cols}, axis=1)

    df = df.merge(temp, how='left', on=col)

    

def haversine_array(lat1, lng1, lat2, lng2): 

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2)) 

    AVG_EARTH_RADIUS = 6371 # in km 

    lat = lat2 - lat1 

    lng = lng2 - lng1 

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2 

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d)) 

    return h



lat2, long2 = df['lat'].values, df['long'].values

dist = pd.DataFrame([haversine_array(df.loc[i, 'lat'], df.loc[i, 'long'], lat2, long2) for i in df.index], index=df.index, columns=df.index, dtype='float32')

rel_cols = ['price_per_living', 'sqft_living', 'condition', 'view', 'grade']

for k in [1, 3, 10]:

    near = dist[(dist>0)&(dist<k)]

    neighbors = [near.loc[i].dropna().index.tolist() for i in near.index]

    cols = ['{}_{}'.format(col, k) for col in rel_cols]

    tmp = pd.DataFrame([df.loc[near, rel_cols].mean().values for near in neighbors], index=df.index, columns=cols)

    df = pd.concat([df, tmp], axis=1)

for col in rel_cols:

    df[col+'_10'] = df[col+'_10'].fillna(df[col+'_10'].mean())

    df[col+'_3'] = df[col+'_3'].fillna(df[col+'_10'])

    df[col+'_1'] = df[col+'_1'].fillna(df[col+'_3'])

    

df = pd.get_dummies(df)

skewness = pd.Series()

for col in [col for col in df.columns if col.startswith('sqft')]:

    if (df[col]<0).sum() == 0:

        skewness.loc[col] = df[col].skew()

skew_col = skewness[skewness>1].index.tolist()

df[skew_col] = np.log1p(df[skew_col])

feats = [col for col in df.columns if col != 'price_per_living']

df[feats] = StandardScaler().fit_transform(df[feats])



x_data = df.loc[tr_idx, feats]

y_data = df.loc['price_per_living']

test = df.loc[te_idx, feats]

"""
import pandas as pd

import numpy as np

from ast import literal_eval

from sklearn.model_selection import KFold

from sklearn.linear_model import Ridge, LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import LinearSVR, SVR

from catboost import CatBoostRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

import warnings

warnings.filterwarnings("ignore")
org_tr = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/train.csv')

org_te = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/test.csv')

x_data = pd.read_csv('../input/kakr-2nd-comp-processed-dataset/x_data.csv', index_col='id')

y_data = pd.Series.from_csv('../input/kakr-2nd-comp-processed-dataset/y_data.csv')

test = pd.read_csv('../input/kakr-2nd-comp-processed-dataset/test.csv', index_col='id')

params = pd.read_csv('../input/kakr-2nd-comp-processed-dataset/params.csv')

params['params'] = params['params'].apply(literal_eval)

params['feats'] = params['feats'].apply(literal_eval)
print(x_data.shape)

x_data.head()
test.head()
params
def rmse(pred, true):

    return -np.sqrt(np.mean((pred-true)**2))



def oof(algo, params, cols):

    prediction = np.zeros(len(x_data))

    test_prediction = np.zeros(len(test))

    for t, v in KFold(5, random_state=0).split(x_data):

        x_train = x_data[cols].iloc[t]

        x_val = x_data[cols].iloc[v]

        y_train = y_data.iloc[t]

        y_val = y_data.iloc[v]

        if algo.startswith('lgb'):

            model = LGBMRegressor(**params)

        elif algo == 'xgb':

            model = XGBRegressor(**params)

        elif algo == 'cb':

            model = CatBoostRegressor(**params)

        elif algo == 'ridge':

            model = Ridge(**params)

        elif algo == 'svm':

            model = LinearSVR(**params)

        elif algo == 'rbf':

            model = SVR(**params)

        elif algo == 'knn':

            model = KNeighborsRegressor(**params)

        if algo in ['ridge', 'svm', 'rbf', 'knn']:

            model.fit(x_train, y_train)

        else:

            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False, early_stopping_rounds=100)

        prediction[v] = np.expm1(model.predict(x_val))

        test_prediction += np.expm1(model.predict(test[cols]))/5

    score = rmse(prediction*org_tr['sqft_living'], org_tr['price'])

    print(score)

    return prediction, test_prediction
test_predictions = []

val_predictions = []

for i in params.index.tolist():

    row = params.loc[i]

    if row['algo'] == 'cb':

        row['params']['task_type'] = 'CPU'

    elif row['algo'].startswith('lgb'):

        row['params']['device_type'] = 'cpu'

        row['params'].pop('metric')

    val_pred, test_pred = oof(row['algo'], row['params'], row['feats'])

    val_predictions.append(val_pred)

    test_predictions.append(test_pred)
val_predictions_t = np.array(val_predictions).transpose()

lr = LinearRegression()

lr.fit(val_predictions_t, np.expm1(y_data))

rmse(lr.predict(val_predictions_t)*org_tr['sqft_living'], org_tr['price'])
test_prediction = lr.predict(np.array(test_predictions).transpose())*org_te['sqft_living']

submission = pd.DataFrame({'id': org_te['id'], 'price': test_prediction})

submission.to_csv('submissions.csv', index=False)