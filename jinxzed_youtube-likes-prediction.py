import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

import scipy.stats as stats

from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.compose import ColumnTransformer

from sklearn.decomposition import PCA, KernelPCA

from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error

from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, StratifiedShuffleSplit
def rmse(true, pred):

    return np.sqrt(mean_squared_error(true, pred))



def rmsle(true, pred):

    return np.sqrt(mean_squared_log_error(true, pred))



from sklearn.metrics import make_scorer



score_rmse = make_scorer(rmse, greater_is_better=False)
train = pd.read_csv('/kaggle/input/youtube-likes-prediction-av-hacklive/train.csv', parse_dates=['publish_date'])

test = pd.read_csv('/kaggle/input/youtube-likes-prediction-av-hacklive/test.csv', parse_dates=['publish_date'])

sample = pd.read_csv('/kaggle/input/youtube-likes-prediction-av-hacklive/sample_submission_cxCGjdN.csv')

print(train.shape, test.shape)
train.info()
train.isna().sum()
test.isna().sum()
train.skew()
train.head()
train.category_id = train.category_id.astype('int')

test.category_id = test.category_id.astype('int')
pt = PowerTransformer(method='yeo-johnson')

skew = ['views','dislikes','comment_count','likes']
i = 0

fig, ax = plt.subplots(2, 4, figsize=(12, 5))

ax = ax.flatten()

for col in skew:    

    sns.distplot(train[col], fit=stats.norm, ax=ax[i], label='Before Transformation')

    sns.distplot(pt.fit_transform(train[col][:,None]), fit=stats.norm, ax=ax[i+1], label='After Transformation')

    ax[i].legend(loc='best')

    ax[i+1].legend(loc='best')

    i+=2

plt.tight_layout()

plt.show()
sns.jointplot(x='views', y='likes', data=train, kind='scatter')

plt.show()
def date_features(data):

    data['day'] = data.publish_date.dt.day

    data['weekday'] = data.publish_date.dt.dayofweek

    data['month'] = data.publish_date.dt.month

    data['year'] = data.publish_date.dt.year

    data['is_weekend'] = np.where((data.weekday==5)|(data.weekday==6), 1, 0)

    data['week_year'] = data.publish_date.dt.weekofyear

    data['day_year'] = data.publish_date.dt.dayofyear

    data['quarter'] = data.publish_date.dt.quarter

    return data



train = date_features(train)

test = date_features(test)
train['title_len'] = train['title'].apply(lambda x: len(x))

train['description_len'] = train['description'].apply(lambda x: len(x))

train['tags_len'] = train['tags'].apply(lambda x: len(x))

train['channel_title_len'] = train['channel_title'].apply(lambda x: len(x))

train['publish_date_days_since_start'] = (train['publish_date'] - train['publish_date'].min()).dt.days

train['channel_title_num_videos'] = train['channel_title'].map(train['channel_title'].value_counts())

train['publish_date_num_videos'] = train['publish_date'].map(train['publish_date'].value_counts())



test['title_len'] = test['title'].apply(lambda x: len(x))

test['description_len'] = test['description'].apply(lambda x: len(x))

test['tags_len'] = test['tags'].apply(lambda x: len(x))

test['channel_title_len'] = test['channel_title'].apply(lambda x: len(x))

test['publish_date_days_since_start'] = (test['publish_date'] - test['publish_date'].min()).dt.days

test['channel_title_num_videos'] = test['channel_title'].map(test['channel_title'].value_counts())

test['publish_date_num_videos'] = test['publish_date'].map(test['publish_date'].value_counts())
train.head(3)
X = train.drop(['likes'], axis=1)

Y = train['likes']
pt_likes = PowerTransformer(method='yeo-johnson')



x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=1, stratify=X['country_code'])

y_train = pt_likes.fit_transform(y_train[:,None])

y_test = pt_likes.transform(y_test[:,None])

print(f'Train : {x_train.shape} Test : {x_test.shape}')
def score_metrics(model, train, train_a, test=None, test_a=None):

    trainpred = model.predict(train)

    print('Train R2 score : %.4f'%r2_score(train_a, trainpred))

    print('Train RMSE score : %.4f'%rmse(train_a, trainpred))

    if test is not None:

        testpred = model.predict(test)

        print('Test R2 score : %.4f'%r2_score(test_a, testpred))

        print('Test RMSE score : %.4f'%rmse(test_a, testpred))

        

def submission(model, name, test=test):

    pred = model.predict(test)

    sample['likes'] = pt_likes.inverse_transform(np.array(pred).reshape(-1,1))

    sample.to_csv(name+'.csv', index=False)
skew = ['views','dislikes','comment_count']

dummy = ['category_id','country_code']

drop = ['video_id','publish_date','title','channel_title','tags','description']

passthru = set(x_train.columns).difference(drop+dummy+skew)

scaler = StandardScaler()

label = OneHotEncoder(handle_unknown='ignore')

#label = OrdinalEncoder()



pt = PowerTransformer(method='yeo-johnson')



transformer = [('skew', pt, skew),

               ('onehot',label, dummy),

               ('pass','drop',drop)]



ct_skew = ColumnTransformer(transformers=transformer)



model_lr = LinearRegression(n_jobs=4)

model_lasso = Lasso(random_state=1, alpha=0.003)
pipe_lr = Pipeline([('skew_treat', ct_skew),

                    ('scaler', scaler),

                    ('model', model_lasso)], verbose=1)



pipe_lr.fit(x_train, y_train)

score_metrics(pipe_lr, x_train, y_train, x_test, y_test)
submission(pipe_lr, 'model_lr')
model_cat = CatBoostRegressor(random_state=1, verbose=0)

model_lgbm = LGBMRegressor(n_jobs=4, random_state=1)

model_xgb = XGBRegressor(n_jobs=4, random_state=1)



models = []

models.append(('LGBM', model_lgbm))

models.append(('XGB', model_xgb))

models.append(('CAT', model_cat))
for name, model in models:

    pipe = Pipeline([('skew_treat', ct_skew),

                     (name, model)], verbose=1)

    pipe.fit(x_train, y_train)

    score_metrics(pipe, x_train, y_train, x_test, y_test)
cat_feature = ['category_id','country_code']

ignored = ['video_id','title','channel_title','publish_date','tags','description']

model_cat = CatBoostRegressor(random_state=1, cat_features = cat_feature, verbose=0, one_hot_max_size=255,

                             max_depth=10, learning_rate=0.08, n_estimators=1000 )



model_cat.fit(x_train.drop(ignored, axis=1), y_train, plot=True, eval_set=(x_test.drop(ignored, axis=1), y_test))



score_metrics(model_cat, x_train.drop(ignored, axis=1), y_train, x_test.drop(ignored, axis=1), y_test)
submission(model_cat, 'model_cat',

           test=test.drop(columns=['video_id','title','channel_title','publish_date','tags','description']))
plt.figure(figsize=(16,5))

plt.bar(model_cat.feature_names_, model_cat.feature_importances_)

plt.xticks(rotation=90)

plt.show()
from IPython.display import FileLink, FileLinks

FileLink('./model_cat.csv')