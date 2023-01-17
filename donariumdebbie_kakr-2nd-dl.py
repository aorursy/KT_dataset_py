import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore')

import os

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA

# from keras import models

# from keras import layers

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

import lightgbm as lgb

import xgboost as xgb

from sklearn.metrics import mean_squared_error as mse
train = pd.read_csv('../input/train.csv', parse_dates=['date'])

test  = pd.read_csv('../input/test.csv', parse_dates=['date'], index_col='id')

train.head()

train_y = train["price"].copy()



## feature engineering 한번에 하는 함수 

### train/test split, 



def feature_engineering(df, is_train = True):

    

    # feature 1 : sum and sub of latitude longitude 

    df['latlongsum'] = df['lat'] + df['long']

    df['latlongsub'] = df['lat'] - df['long']

    

    # feature 2 : month and year

    df['Month'] = df['date'].dt.month

    df['Year'] = df['date'].dt.year

    

    # feature 3 : renovated year update

    df.loc[df.yr_renovated==0,'yr_renovated']=df[df.yr_renovated==0].yr_built

    

    # feature 4 : zipfeatures (ref: https://www.kaggle.com/tmheo74/geo-data-eda-and-feature-engineering)

    df['zipcode_str'] = df['zipcode'].astype(str)  

    df['zipcode-3'] = 'z_' + df['zipcode_str'].str[2:3]

    df['zipcode-4'] = 'z_' + df['zipcode_str'].str[3:4]

    df['zipcode-5'] = 'z_' + df['zipcode_str'].str[4:5]

    df['zipcode-34'] = 'z_' + df['zipcode_str'].str[2:4]

    df['zipcode-45'] = 'z_' + df['zipcode_str'].str[3:5]

    df['zipcode-35'] = 'z_' + df['zipcode_str'].str[2:3] + df['zipcode_str'].str[4:5]

    df.drop(['zipcode_str'], 1, inplace=True)

    

    # drop useless columns

    if is_train:

        df.drop(["id"], 1, inplace=True)

        df.drop(["price"], 1, inplace=True)

        df.drop(["date"], 1, inplace=True)

    else: # test는 id랑 price가 없으므로

        df.drop(["date"], 1, inplace=True)

    

    # label encoding

    cat_cols = df.select_dtypes('object').columns

    for col in cat_cols:

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col])

        

    # feature 5 : pca -> date 드랍후에 해야됨. 

    pca1 = PCA(n_components=2)

    pca1.fit(df)

    coord_pca2 = pca1.transform(df)

    df['pca1'] = coord_pca2[:, 0]

    df['pca2'] = coord_pca2[:, 1]

    

    # feature 6 : pca (lat, long)

    coord = df[['lat','long']]

    pca2 = PCA(n_components=2)

    pca2.fit(coord)

    coord_pca = pca2.transform(coord)

    df['coord_pca1'] = coord_pca[:, 0]

    df['coord_pca2'] = coord_pca[:, 1]

    

    return df



# ## 평가함수 

# def eval(val_y, pred):

#     rmse = np.sqrt(mse(val_y, pred))

#     return rmse
train = feature_engineering(train, True)

test = feature_engineering(test, False)



def rmse_exp(predictions, dmat):

    labels = dmat.get_label()

    error = np.expm1(predictions) - np.expm1(labels)

    squared_error = np.square(error)

    mean = np.mean(squared_error)

    return ('rmse_exp', np.sqrt(mean))



xgb_params = {

    'eta': 0.02,

    'max_depth': 6,

    'subsample': 0.8,

    'colsample_bytree': 0.4,

    'objective': 'reg:linear',    

    'eval_metric': 'rmse',        

    'silent': True,               

    'n_estimators' : 100

}



train_y = np.log1p(train_y)
%%time

# transforming

dtrain = xgb.DMatrix(train, train_y)

dtest = xgb.DMatrix(test)



# cross validation

cv_output = xgb.cv(xgb_params,

                   dtrain,                        

                   num_boost_round=10000,        

                   early_stopping_rounds=200,    

                   nfold=5,                      

                   verbose_eval=200,             

                   feval=rmse_exp,               

                   maximize=False,

                   show_stdv=False,

                   seed = 1080

                   )



# scoring

best_rounds = cv_output.index.size

score = round(cv_output.iloc[-1]['test-rmse_exp-mean'], 2)



print(f'\nBest Rounds: {best_rounds}')

print(f'Best Score: {score}')



# plotting

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))

# cv_output[['train-rmse-mean', 'test-rmse-mean']].plot(ax=ax1)

# ax1.set_title('RMSE_log', fontsize=20)

# cv_output[['train-rmse_exp-mean', 'test-rmse_exp-mean']].plot(ax=ax2)

# ax2.set_title('RMSE', fontsize=20)



# plt.show()
model = xgb.train(xgb_params, dtrain, num_boost_round=best_rounds)

y_pred = model.predict(dtest)

y_pred_xgb = np.expm1(y_pred)

print(y_pred_xgb)
sub = pd.read_csv("../input/sample_submission.csv") 

sub1 = sub.copy()

sub1['price'] = y_pred_xgb
# 허태명님 커널참고하며 조금 바꿈



# 데이터 가져오기

def load_original_data():

    train = pd.read_csv('../input/train.csv', parse_dates=['date'])

    test = pd.read_csv('../input/test.csv', parse_dates=['date'], index_col='id')



    train_copy = train.copy()

    train_copy['data'] = 'train'

    test_copy = test.copy()

    test_copy['data'] = 'test'

    test_copy['price'] = np.nan



    # remove outlier

    train_copy = train_copy[~((train_copy['sqft_living'] > 12000) & (train_copy['price'] < 3000000))].reset_index(drop=True)



    # concat train, test data to preprocess

    data = pd.concat([train_copy, test_copy], sort=False).reset_index(drop=True)

    data = data[train_copy.columns]



    # 날짜피쳐 드랍전에 중요한거 추가

    data['Month'] = data['date'].dt.month

    data['Year'] = data['date'].dt.year

    

    data.drop('date', axis=1, inplace=True)

    data['zipcode'] = data['zipcode'].astype(str)



    # fix skew feature

    skew_columns = ['price']



    for c in skew_columns:

        data[c] = np.log1p(data[c])

        

    return data



RANDOM_SEED = 1080

np.random.seed(RANDOM_SEED)

def rmse_exp(y_true, y_pred):

    return np.sqrt(mean_squared_error(np.expm1(y_true), np.expm1(y_pred)))



def train_test_split(data, do_ohe=True):

    df = data.drop(['id','price','data'], axis=1).copy()

    cat_cols = df.select_dtypes('object').columns

    for col in cat_cols:

        if do_ohe:

            ohe_df = pd.get_dummies(df[[col]], prefix='ohe_'+col)

            df.drop(col, axis=1, inplace=True)

            df = pd.concat([df, ohe_df], axis=1)

        else:

            le = LabelEncoder()

            df[col] = le.fit_transform(df[col])



    train_len = data[data['data'] == 'train'].shape[0]

    X_train = df.iloc[:train_len]

    X_test = df.iloc[train_len:]

    y_train = data[data['data'] == 'train']['price']

    

    return X_train, X_test, y_train





def get_oof_lgb(X_train, y_train, X_test, lgb_param, verbose_eval=False, return_cv_score_only=False):



    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    oof = np.zeros(len(X_train))

    predictions = np.zeros(len(X_test))

    feature_importance_df = pd.DataFrame()



    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values, y_train.values)):

        if verbose_eval > 0: print(f'Fold : {fold_ + 1}')

        trn_data = lgb.Dataset(X_train.iloc[trn_idx], label=y_train.iloc[trn_idx])

        val_data = lgb.Dataset(X_train.iloc[val_idx], label=y_train.iloc[val_idx])



        num_round = 100000

        clf = lgb.train(lgb_param, trn_data, num_round, valid_sets=[trn_data, val_data],

                        verbose_eval=verbose_eval, early_stopping_rounds=200)

        oof[val_idx] = clf.predict(X_train.iloc[val_idx], num_iteration=clf.best_iteration)

        predictions += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

        

        cv_fold_score = rmse_exp(y_train.iloc[val_idx], oof[val_idx])

        

        if verbose_eval > 0: print(f'Fold {fold_ + 1} / CV-Score: {cv_fold_score:.6f}')

        

        fold_importance_df = pd.DataFrame()

        fold_importance_df['feature'] = X_train.columns.tolist()

        fold_importance_df['importance'] = clf.feature_importance('gain')

        fold_importance_df['fold'] = fold_ + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)



    cv_score = rmse_exp(y_train, oof)

    print(f'CV-Score: {cv_score:.6f}')

    if return_cv_score_only: return cv_score

    else: return oof, predictions, cv_score, feature_importance_df
data = load_original_data()



coord = data[['lat','long']]

pca = PCA(n_components=2)

pca.fit(coord)



coord_pca = pca.transform(coord)



data['coord_pca1'] = coord_pca[:, 0]

data['coord_pca2'] = coord_pca[:, 1]



# 피쳐추가 

data['latlongsum'] = data['lat'] + data['long']

data['latlongsub'] = data['lat'] - data['long']

data.loc[data.yr_renovated==0,'yr_renovated'] = data[data.yr_renovated==0].yr_built



# kmeans for lat, long

kmeans = KMeans(n_clusters=32, random_state=RANDOM_SEED).fit(coord)

coord_cluster = kmeans.predict(coord)

data['coord_cluster'] = coord_cluster

data['coord_cluster'] = data['coord_cluster'].map(lambda x: 'c_' + str(x).rjust(2, '0'))



X_train, X_test, y_train = train_test_split(data, do_ohe = False)

print(X_train.shape, X_test.shape)



lgb_param = {

    'objective': 'regression',

    'learning_rate': 0.05,

    'num_leaves': 15,

    'bagging_fraction': 0.7,

    'bagging_freq': 1,

    'feature_fraction': 0.7,

    'seed': RANDOM_SEED,

    'metric': ['rmse'],

}



oof, lgbm_pred, cv_score, fi_df = get_oof_lgb(X_train, y_train, X_test, lgb_param)
y_pred_lgbm = np.expm1(lgbm_pred)

sub2 = pd.read_csv("../input/sample_submission.csv") 

sub2['price'] = y_pred_lgbm

pred_df = sub.copy()

pred_df['price'] = sub1['price']*0.55 + sub2['price']*0.45
def export(pred):

    subm = pd.read_csv('../input/sample_submission.csv')

    subm['price'] = pred



    subm_num = 0

    subm_name = './subm_{}.csv'.format(str(subm_num).zfill(3))



    while os.path.isfile(subm_name):

        subm_num += 1

        subm_name = './subm_{}.csv'.format(str(subm_num).zfill(3))



    print(subm_name)

    subm.to_csv(subm_name, index=False)
export(pred_df['price'])