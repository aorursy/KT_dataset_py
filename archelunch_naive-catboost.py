import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import catboost as cboost

import gc

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import Ridge

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split, cross_val_score

import xgboost as xgb

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.ensemble import RandomForestRegressor

%matplotlib inline
def handle_missing_inplace(dataset):

    dataset['category_name'].fillna(value='missing', inplace=True)

    dataset['brand_name'].fillna(value='missing', inplace=True)

    dataset['item_description'].fillna(value='missing', inplace=True)





def cutting(dataset):

    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]

    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'

    pop_category = dataset['category_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]

    dataset.loc[~dataset['category_name'].isin(pop_category), 'category_name'] = 'missing'





def to_categorical(dataset):

    dataset['category_name'] = dataset['category_name'].astype('category')

    dataset['brand_name'] = dataset['brand_name'].astype('category')

    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')
train = pd.read_table('../input/train.tsv', engine='c')

test = pd.read_table('../input/test.tsv', engine='c')

nrow_train = train.shape[0]
y = np.log1p(train["price"])

merge: pd.DataFrame = pd.concat([train, test])

submission: pd.DataFrame = test[['test_id']]
NUM_BRANDS = 4000

NUM_CATEGORIES = 1000

NAME_MIN_DF = 10

MAX_FEATURES_ITEM_DESCRIPTION = 50000

handle_missing_inplace(merge)

cutting(merge)

to_categorical(merge)
cv = CountVectorizer(min_df=NAME_MIN_DF)

X_name = cv.fit_transform(merge['name'])

cv = CountVectorizer()

X_category = cv.fit_transform(merge['category_name'])

tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,

                         ngram_range=(1, 3),

                         stop_words='english')

X_description = tv.fit_transform(merge['item_description'])

lb = LabelBinarizer(sparse_output=True)

X_brand = lb.fit_transform(merge['brand_name'])

X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],

                                          sparse=True).values)

sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()

X = sparse_merge[:nrow_train]

X_test = sparse_merge[nrow_train:]
model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3)

model.fit(X, y)

predsR = model.predict(X=X_test)

model = Ridge(solver="lsqr", fit_intercept=False, random_state=145, alpha = 3)

model.fit(X, y)

predsR2 = model.predict(X=X_test)

model = Ridge(solver="sag", fit_intercept=False, random_state=205, alpha = 3)

model.fit(X, y)

predsR3 = model.predict(X=X_test)
model = RandomForestRegressor(max_features='log2', min_weight_fraction_leaf=0.1)

model.fit(X, y)

predsL = model.predict(X=X_test)

# model = GaussianProcessRegressor(random_state=145, alpha = 3)

# model.fit(X, y)

# predsL2 = model.predict(X=X_test)

# # Create train and test Pool of train

# ptrain = cboost.Pool(pd.DataFrame(X.toarray()), y)

# ptest = cboost.Pool(pd.DataFrame(X_test.toarray()))
# # Tune your parameters here!

# cboost_params = {

#     'nan_mode': 'Min',

#     'loss_function': 'RMSE',  # Try 'LogLinQuantile' as well

#     'iterations': 500,

#     'learning_rate': 0.76,

#     'depth': 3,

#     'verbose': True

# }



# cboost_params2 = {

#     'nan_mode': 'Min',

#     'loss_function': 'RMSE',  # Try 'LogLinQuantile' as well

#     'iterations': 500,

#     'learning_rate': 0.85,

#     'depth': 3,

#     'verbose': True

# }

# best_iter = cboost_params['iterations']  # Initial 'guess' it not using CV

# best_iter2 = cboost_params2['iterations']

# # cv_result = cboost.cv(cboost_params, ptrain_sub, fold_count=3)



# # df_cv_result = pd.DataFrame({'train': cv_result['RMSE_train_avg'],

# #                              'valid': cv_result['RMSE_test_avg']})



# # # Best results

# # print('Best results:')

# # best_iter = df_cv_result.valid.argmin()+1

# # df_cv_bestresult = df_cv_result.iloc[best_iter-1]

# # print(df_cv_bestresult)



# # fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# # df_cv_result.plot(ax=ax[0])



# # ax[1].plot(df_cv_result.train, df_cv_result.valid, 'o-')

# # ax[1].scatter([df_cv_bestresult['train']], [df_cv_bestresult['valid']], c='red')

# # ax[1].set_xlabel('train')

# # ax[1].set_ylabel('valid')
# # Train model on full data

# model = cboost.CatBoostRegressor(**dict(cboost_params, verbose=False, iterations=best_iter))

# fit_model = model.fit(ptrain)

# predsL = fit_model.predict(ptest).clip(0)

# # Train model on full data

# model = cboost.CatBoostRegressor(**dict(cboost_params2, verbose=False, iterations=best_iter2))

# fit_model = model.fit(ptrain)

# predsL2 = fit_model.predict(ptest).clip(0)
# Predict test and save to .csv

preds1 = np.expm1(predsR3*0.24 + predsR2*0.24 + predsR*0.52)

preds2 = np.expm1(predsL) #+ predsL2*0.5)

submission['price'] = preds1*0.75 + preds2*0.25

submission.to_csv('submission.csv', index=False)
!head submission.csv