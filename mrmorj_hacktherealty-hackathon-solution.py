def metric(y_true, y_real):

    return -1*np.mean(np.exp(np.abs(y_true-y_real)) - 1)
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



train = pd.read_csv('/kaggle/input/yandex-exposition-task/exposition_train.tsv', sep='\t')

test = pd.read_csv('/kaggle/input/yandex-exposition-task/exposition_test.tsv', sep='\t')

submission = pd.read_csv('/kaggle/input/yandex-exposition-task/exposition_sample_submission.tsv', sep='\t')
train.head()
print(f'NaN value in train: {round(train.isna().sum().sum() / (train.shape[0]*train.shape[1]), 4)*100}%')

print(f'NaN value in test: {round(test.isna().sum().sum() / (test.shape[0]*test.shape[1]), 4)*100}%')
train.columns
train['target'].value_counts(normalize=True)
sns.countplot(data=train, x='parking').set_title('Parking count')

plt.xticks(

    rotation=45, 

    horizontalalignment='right')

plt.show()

sns.countplot(data=train, x='building_type').set_title('Building_type count')

plt.xticks(

    rotation=45, 

    horizontalalignment='right')

plt.show()

sns.countplot(data=train, x='renovation').set_title('Renovation count')

plt.xticks(

    rotation=45, 

    horizontalalignment='right')

plt.show()
sns.distplot(train['build_year']).set_title('Build_year distribution')

plt.show()

sns.distplot(train['ceiling_height']).set_title('Ceiling_height distribution')

plt.show()

sns.distplot(train['area']).set_title('Area distribution')

plt.show()
to_drop = ['site_id', 'target_string', 'main_image', 'id', 'total_area', 'building_id', 'day']



def drop_col(df, to_drop):

    df = df.drop(to_drop, axis=1)

    return df
train_df = drop_col(train, to_drop)
def replace(df):

    df.loc[df.build_year == 0, 'build_year'] = np.NaN

    df['build_year'] = df['build_year'].fillna((df.groupby(['building_series_id'])['build_year'].transform('median')))

        

    df.loc[(df.has_elevator==0) & (df.floor>5), 'has_elevator'] = 1

    

    df.loc[df.price<100, 'price'] *= 1000

    

    df.loc[(df.ceiling_height<2) | (df.ceiling_height>5), 'ceiling_height'] = np.NaN

    df['ceiling_height'] = df['ceiling_height'].fillna(df.groupby(['floors_total','flats_count'])['ceiling_height'].transform('median'))

    

    df = df[df.area>df.kitchen_area].reset_index(drop=True)



    

    return df
train_df = replace(train_df)

test_df = replace(test)
def mapping(df):

    

    balcony_map = {'UNKNOWN': 0, 'BALCONY': 1, 'LOGGIA':0, 'TWO_LOGGIA':0, 'TWO_BALCONY':2, 'BALCONY__LOGGIA':1,

              'BALCONY__TWO_LOGGIA':1, 'THREE_LOGGIA':0, 'THREE_BALCONY':2}

    loggia_map = {'UNKNOWN': 0, 'BALCONY': 0, 'LOGGIA':1, 'TWO_LOGGIA':2, 'TWO_BALCONY':0, 'BALCONY__LOGGIA':1,

              'BALCONY__TWO_LOGGIA':2, 'THREE_LOGGIA':2, 'THREE_BALCONY':0}



    df['expect_demolition'] = df['expect_demolition'].map({False:0,True:1})

    df['is_apartment'] = df['is_apartment'].map({False:0,True:1})

    df['has_elevator'] = df['has_elevator'].map({False:0,True:1})

    df['studio'] = df['studio'].map({False:0,True:1})

    df['num_balcony'] = df['balcony'].map(balcony_map)

    df['num_loggia'] = df['balcony'].map(loggia_map)

    

    return df
train_df = mapping(train_df)

test_df = mapping(test_df)
def smoothed_likelihood(df, column, alpha, target_column, test_df):

    global_mean = df[target_column].mean()

    nrows = df.groupby(column).count()[target_column].to_dict()

    local_mean = df.groupby(column).mean()[target_column].to_dict()

    if test_df is None:

        new_column = df[column].apply(lambda x: (local_mean[x]*nrows[x] + global_mean*alpha)/(nrows[x]+alpha))

    else:

        new_column = test_df[column].apply(lambda x: (local_mean[x]*nrows[x] + global_mean*alpha)/(nrows[x]+alpha) if x in local_mean.keys() else global_mean)

    return new_column
cat_columns = ['parking', 'unified_address','building_type','locality_name','renovation', 'building_series_id']



for col in cat_columns:

    train_df[col] = smoothed_likelihood(train_df, col, 0.15, 'target', None)

    test_df[col] = smoothed_likelihood(train_df, col, 0.15, 'target', test_df)
def haversine_dist(lat1,lng1,lat2,lng2):

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    radius = 6371  # Earth's radius taken from google

    lat = lat2 - lat1

    lng = lng2 - lng1

    d = np.sin(lat/2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng/2) ** 2

    h = 2 * radius * np.arcsin(np.sqrt(d))

    return h
def eng(df):

    

    df['years_old'] = 2020 - df['build_year']

    

    moskow_lat = 55.751244

    moskow_lon = 37.618423

    df['moskow_dist'] = np.sqrt((df['latitude'] - moskow_lat)**2 + (df['longitude'] - moskow_lon)**2)

    

    

    df['rot_45_x'] = (0.707 * df['latitude']) + (0.707 * df['longitude'])

    #df['rot_45_y'] = (0.707 * df['longitude']) + (0.707 * df['latitude'])

    df['rot_30_x'] = (0.866 * df['latitude']) + (0.5 * df['longitude'])

    df['rot_30_y'] = (0.866 * df['longitude']) + (0.5 * df['latitude'])

    

    df['haversine_moskow'] = haversine_dist(df['latitude'], df['longitude'], 55.751244, 37.618423)

        

    return df
train_df = eng(train_df)

test_df = eng(test_df)
from lightgbm import LGBMRegressor, LGBMClassifier

from sklearn.model_selection import StratifiedKFold, KFold

from catboost import CatBoostRegressor, CatBoostClassifier



seed = 47



kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)

lgbm = LGBMRegressor(random_state=seed, n_estimators=500, learning_rate=0.1)
def calc(X,y,X_test, model, cv, cols, oof):

    

    if cols is None:

        cols = X.columns

    X=X[cols]

    

    res=[]

    local_probs = pd.DataFrame()

    for i, (tdx, vdx) in enumerate(cv.split(X, y)):

        X_train, X_valid, y_train, y_valid = X.iloc[tdx], X.iloc[vdx], y.iloc[tdx], y.iloc[vdx]

        model.fit(X_train, y_train,

                 eval_set=[(X_train, y_train), (X_valid, y_valid)],

                 early_stopping_rounds=30, 

                  verbose=False)   

        preds = model.predict(X_valid)

        

        if oof==1:

            X_test=X_test[cols]

            oof_predict = model.predict(X_test)

            local_probs['fold_%i'%i] = oof_predict

            

        y_valid = y_valid.to_numpy().reshape(1,-1)[0]

        ll = metric(y_valid, np.round(preds))

        print(f'{i} Fold: {ll:.4f}')

        res.append(ll)

        

        

    print(f'AVG score: {np.mean(res)}')

    return np.mean(res), local_probs.mean(axis=1)
X = train_df.drop(['target', 'balcony', 'floor', 'ceiling_height','build_year','studio'], axis=1)#, 'unified_address', 'locality_name'

y = train_df[['target']]



_, res_df = calc(X,y,test_df, lgbm, kfold, None, 1)
submission['target'] = np.round(res_df).astype(int)

submission['target'].value_counts(normalize=True)

submission.to_csv('submission.tsv', sep='\t', index=False)