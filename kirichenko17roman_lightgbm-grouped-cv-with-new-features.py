import numpy as np 

import pandas as pd

import lightgbm as lgb

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
train = pd.read_csv('../input/insta_train.csv')

test = pd.read_csv('../input/insta_test.csv')
v = TfidfVectorizer(

    max_features=None, 

    max_df=0.1,

    stop_words='english', 

    min_df=0.0003)



x = v.fit(train['caption'].fillna('').values)



a = v.transform(train['caption'].fillna('').values)

b = v.transform(test['caption'].fillna('').values)
a.shape, b.shape
model = PCA(n_components=200, random_state=0)

model.fit(a.toarray())

transformed = model.transform(a.toarray())

transformed_test = model.transform(b.toarray())
model = PCA(n_components=20, random_state=0)

transformed = model.fit_transform(a.toarray())

# model.transform(a.toarray())

transformed_test = model.transform(b.toarray())
transformed.shape, transformed_test.shape
%matplotlib inline



features = range(model.n_components_)

plt.figure(figsize=(10, 5))

plt.bar(features,np.cumsum(model.explained_variance_ratio_))

plt.xticks(features)

plt.ylabel('variance')

plt.xlabel('PCA feature')

plt.show()
train = pd.concat([train, pd.DataFrame(transformed)], axis=1)

test = pd.concat([test, pd.DataFrame(transformed_test)], axis=1)
train.head()
kmeans = KMeans(

    n_clusters=30,

    random_state=0

)



kmeans_train = kmeans.fit(a)
train['cluster'] = kmeans_train.labels_

test['cluster'] = kmeans.predict(b)
y = train['likes']
def ing(df):

    df['user_len'] = df['user'].str.len()

    df['username_len'] = df['username'].str.len()

    df['img_len'] = df['img'].str.len()

    df['description_len'] = df['description'].str.len()

    df['caption_upper'] = df['caption'].str.findall(r'[A-Z]').str.len()

    df['po_co'] = df['posts']/(df['comments']+1)

    df['po_co_pow'] = df['posts']*(df['comments'])

    df['fol'] = df['followers']/(df['followings']+1)

    df['fol_pow'] = df['followers']*(df['followings'])

    df['act'] = df['comments']/(df['followers']+1)

    df['pos'] = df['posts']/(df['followers']+1)

    df['pof'] = df['posts']/(df['followings']+1)

    df['user_count'] = df.groupby('user')['posts'].transform('count')

    df['language_encoded'] = df.groupby('user')['language_encoded'].transform('median')

    df['hashtag_max'] = df.groupby('user')['hashtag'].transform('max')

    df['hashtag_min'] = df.groupby('user')['hashtag'].transform('min')

    df['hashtag_mean'] = df.groupby('user')['hashtag'].transform('mean')

    df['hashtag_std'] = df.groupby('user')['hashtag'].transform('std')

    df['hashtag_mean_diff'] = df['hashtag'] / (df['hashtag_mean']+1)

    df['comments_max'] = df.groupby('user')['comments'].transform('max')

    df['comments_min'] = df.groupby('user')['comments'].transform('min')

    df['comments_mean'] = df.groupby('user')['comments'].transform('mean')

    df['comments_std'] = df.groupby('user')['comments'].transform('std')

    df['comments_mean_diff'] = df['comments'] / (df['comments_mean']+1)

    df['bot'] = df['user'].str.count('_|#|\.|,').astype(int)

    df['bot_caption'] = pd.to_numeric(df['caption'].str.count('_|#|\.|,'), errors='coerce')

    df['numbers_user'] = df['user'].str.contains('0|1|2|3|4|5|6|7|8|9').astype(int)

    df['numbers_caption'] = pd.to_numeric(df['caption'].str.contains('0|1|2|3|4|5|6|7|8|9'), errors='coerce')

    df['s1080.1080'] = pd.to_numeric(df['img'].str.contains('1080\.1080'), errors='coerce')

    df['e35'] = pd.to_numeric(df['img'].str.contains('e35'), errors='coerce')

    return df
train = ing(train).select_dtypes(exclude=['object']).drop(['likes'], axis=1).fillna(0)

test = ing(test).select_dtypes(exclude=['object']).fillna(0)
estimator = lgb.LGBMRegressor(

    metric='mae',

    learning_rate=0.005,

    n_estimators = 5500, 

    num_leaves = 10, 

    colsample_bytree = 0.59, 

    subsample = 0.5, 

    max_bin = 100,

    reg_alpha = 1, 

    bagging_freq = 5,

    reg_lamba=10,

    min_data = 1)



param_grid = {

   # 'learning_rate': [0.01, 0.02],

   # 'n_estimators': [500, 1000, 2000],

   # 'min_child_samples': [1,10,50],

   # 'num_leaves': [5,10,20],

   # 'colsample_bytree': [0.89],

   # 'subsample': [0.9],

   # 'reg_alpha': [0,1,10],

   # 'reg_lamba': [0,1, 10]

   # 'min_data': [1, 10, 20, 100]

}



from sklearn.metrics import make_scorer



def smape(A, F):

    A = 10**(A)-1

    F = 10**(F)-1

    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))



scoring = {'smape': make_scorer(smape, greater_is_better=False)}



gbm = GridSearchCV(estimator, 

                   param_grid, 

                   cv=5, 

                   scoring=scoring, 

                   refit=False, verbose=3, n_jobs = -1)

gbm.fit(train, 

        np.log10(y+1), 

        groups = np.concatenate([np.repeat(0, 8308), 

                                 np.repeat(1, 8308), 

                                 np.repeat(2, 8308), 

                                 np.repeat(3, 8308), 

                                 np.repeat(4, 8308)]))

print("CV MAPE:", gbm.cv_results_['mean_test_smape'][0])
#gbm.cv_results_
#feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importances_, train.columns.astype('str'))), columns=['Value','Feature'])

#plt.figure(figsize=(10, 20))

#sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

#plt.title('LightGBM Features (avg over folds)')

#plt.tight_layout()

#plt.show()
class exp_smape(object):

    def get_final_error(self, error, weight):

        return error / (weight + 1e-38)



    def is_max_optimal(self):

        return False



    def evaluate(self, approxes, target, weight):

        # approxes is a list of indexed containers

        # (containers with only __len__ and __getitem__ defined),

        # one container per approx dimension.

        # Each container contains floats.

        # weight is a one dimensional indexed container.

        # target is float.

        

        # weight parameter can be None.

        # Returns pair (error, weights sum)

        

        assert len(approxes) == 1

        assert len(target) == len(approxes[0])



        approx = approxes[0]



        error_sum = 0.0

        weight_sum = 0.0



        for i in range(len(approx)):

            w = 1.0 if weight is None else weight[i]

            weight_sum += w

            A = 10**(target[i])-1

            F = 10**(approx[i])-1

            error_sum += -w * -(100 * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F))))



        return error_sum, weight_sum
from catboost import CatBoostRegressor, FeaturesData, Pool



train_pool = Pool(

    data = train,

    label = np.log10(y+1)

)



test_pool = Pool(

    data = test

)



model = CatBoostRegressor(

    iterations=300, 

    learning_rate=0.1, 

    depth=7, 

    loss_function = "MAE", 

    eval_metric = exp_smape()

)

# Fit model

model.fit(train_pool)

# Get predictions

preds = model.predict(test_pool)
(10**preds)-1
estimator.fit(train, np.log10(y+1))

print(10**(estimator.predict(test))-1)

prediction = pd.read_csv('../input/sample_submission.csv')

prediction['likes'] = ((10**(estimator.predict(test)))-1)*0.6 + ((10**preds)-1)*0.4

prediction.to_csv('exp_submission.csv', index=False)
#from catboost import cv

#

#params = {"iterations": 1000,

#          "learning_rate": 0.1,

#          "depth": 7,

#          "loss_function": "MAE",

#          "eval_metric": exp_smape(),

#          "verbose": True}

#

#scores = cv(train_pool, params = params, fold_count=5, shuffle = False, early_stopping_rounds = 200, verbose_eval = 10)
#scores