# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 13:24:27 2018

@author: Michael Apers, borrowed heavily from Tilii at 
https://www.kaggle.com/tilii7/cross-validation-weighted-linear-blending-errors
"""
###EDA

# Imports of everything one might need
import string
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import mean_squared_error
import scipy
from scipy import sparse
from scipy.optimize import minimize
from IPython.display import display
import lightgbm as lgb
import gc

#time everything
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod(
            (datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' %
              (thour, tmin, round(tsec, 2)))
        
start_time = timer(None)   
def load_test():
    for df in pd.read_csv('../input/test.tsv', sep='\t', chunksize=700000):
        yield df
        
def rmse_min_func(weights):
    final_prediction = 0
    for weight, prediction in zip(weights, blend_train):
        final_prediction += weight * prediction
    return np.sqrt(mean_squared_error(y_train, final_prediction))        
# Definitions
pd.set_option('display.float_format', lambda x: '%.3f' % x)
njobs = -1

# Get data
train = pd.read_csv("../input/train.tsv", sep='\t')
test_chunk_count = 0
def DescriptionNotMeaningless(row):
    if (row == "No description yet" or row == "No category name given" or row == "No brand name given" or len(row) == 1 or row == u"\uFFFC" or row.isspace() or not row or all(j in string.punctuation for j in row)):
        r = 0
    else:
        r = 1
    return r

for test in load_test():
    if test_chunk_count == 0:
        # Drop Id column
        tr_ids = train['train_id'].copy()
        train.drop("train_id", axis = 1, inplace = True)
        te_ids = test['test_id'].copy()
        test.drop("test_id", axis = 1, inplace = True)
        PRICE = np.log1p(train['price'].copy())  #well log(price+1) at least...


        #creating a category for item descriptions that are empty or meaningless

        train.item_description = train.item_description.fillna("No description yet")
        test.item_description = test.item_description.fillna("No description yet")

        train['has_description'] = train.item_description.apply(lambda row : DescriptionNotMeaningless(row))
        test['has_description'] = test.item_description.apply(lambda row : DescriptionNotMeaningless(row))

        train.brand_name = train.brand_name.fillna("No brand name given")
        test.brand_name = test.brand_name.fillna("No brand name given")
        timer(start_time)
        brand_names = train.brand_name.unique()
        test.brand_name.replace([name for name in test.brand_name if name not in brand_names], 'No brand name given', inplace=True)
        timer(start_time)
        train['has_brand'] = train.brand_name.apply(lambda row : DescriptionNotMeaningless(row))
        test['has_brand'] = test.brand_name.apply(lambda row : DescriptionNotMeaningless(row))

        train.category_name = train.category_name.fillna("No category name given")
        test.category_name = test.category_name.fillna("No category name given")
        category_names = train.category_name.unique()
        test.category_name.replace([name for name in test.category_name if name not in category_names], 'No category name given', inplace=True)
        train['has_category'] = train.category_name.apply(lambda row : DescriptionNotMeaningless(row))
        test['has_category'] = test.category_name.apply(lambda row : DescriptionNotMeaningless(row))

        #brand price
        train['brand_price'] = train.groupby('brand_name')['price'].transform(np.mean)
        mapping = dict(train[['brand_name', 'brand_price']].values)
        test['brand_price'] = test.brand_name.replace(mapping)

        timer(start_time)

        #category price
        train['category_price'] = train.groupby('category_name')['price'].transform(np.mean)
        mapping2 = dict(train[['category_name', 'category_price']].values)
        test['category_price'] = test.category_name.replace(mapping2)

        timer(start_time)


        train['item_description'] = train['item_description'].astype('category')
        train['brand_name'] = train['brand_name'].astype('category')
        train['has_brand'] = train['has_brand'].astype('category')
        train['has_description'] = train['has_description'].astype('category')
        train['has_category'] = train['has_category'].astype('category')
        train['category_name'] = train['category_name'].astype('category')
        test['item_description'] = test['item_description'].astype('category')
        test['brand_name'] = test['brand_name'].astype('category')
        test['has_brand'] = test['has_brand'].astype('category')
        test['has_description'] = test['has_description'].astype('category')
        test['has_category'] = test['has_category'].astype('category')
        test['category_name'] = test['category_name'].astype('category')


        numerical_features = train.drop(['price'],axis=1).select_dtypes(exclude = ["object", "category"]).columns
        stdSc = StandardScaler()
        train.loc[:, numerical_features] = stdSc.fit_transform(train.loc[:, numerical_features])
        test.loc[:, numerical_features] = stdSc.transform(test.loc[:, numerical_features])
        timer(start_time)
        df = pd.concat([train, test])
        nrow_train = train.shape[0]
        del train,test
        gc.collect()


        print('Encodings')
        count = CountVectorizer(min_df=10)
        X_name = count.fit_transform(df['name'])


        print('Category Encoders')
        count_category = CountVectorizer()
        X_category = count_category.fit_transform(df['category_name'])

        print('Descp encoders')
        count_descp = TfidfVectorizer(max_features=50000,
                                      ngram_range=(1, 3),
                                      stop_words='english')
        X_descp = count_descp.fit_transform(df['item_description'])

        print('Brand encoders')
        vect_brand = LabelBinarizer(sparse_output=True)
        X_brand = vect_brand.fit_transform(df['brand_name'])


        print('Dummy Encoders')
        X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(df[[
            'item_condition_id', 'has_description', 'has_brand', 'has_category', 'shipping'
        ]], sparse=True).values)

        df_clean = scipy.sparse.hstack((X_dummies, X_brand, X_category, X_descp, X_name,
                                         scipy.sparse.csr_matrix(df['brand_price']).T,
                                         scipy.sparse.csr_matrix(df['category_price']).T)).tocsr()

        train = df_clean[:nrow_train]
        test = df_clean[nrow_train:]

        print(train.shape)
        print(test.shape)
        timer(start_time)

                # ##maybe can mess with skewing numbers##
        ##############################################################

        #renaming to match modeling convention
        y_train = PRICE

        #COPY PASTE FROM Tilii's https://www.kaggle.com/tilii7/cross-validation-weighted-linear-blending-errors:

        # This number of folds is forced by time limit
        folds = 2

        sgd_cv_sum = 0
        ridge_cv_sum = 0
        lgb_cv_sum = 0
        lgb_pred = []
        sgd_pred = []
        ridge_pred = []
        lgb_fpred = []
        sgd_fpred = []
        ridge_fpred = []

        avreal = y_train
        lgb_avpred = np.zeros(train.shape[0])
        sgd_avpred = np.zeros(train.shape[0])
        ridge_avpred = np.zeros(train.shape[0])
        idpred = tr_ids

        blend_train = []
        blend_test = []
        model1 = []
        model2 = []
        model3 = []
        train_time = timer(None)
        kf = KFold(n_splits=folds, random_state=1001)
        for i, (train_index, val_index) in enumerate(kf.split(train, y_train)):
            start_time = timer(None)
            Xtrain, Xval = train[train_index], train[val_index]
            ytrain, yval = y_train[train_index], y_train[val_index]

            model = SGDRegressor(penalty='l2',
                                 loss='squared_epsilon_insensitive',
                                 max_iter=200,
                                 tol=0.00001,
                                 epsilon=0.0001,
                                 learning_rate='invscaling',
                                 fit_intercept=False,
                                 alpha=1e-10,
                                 l1_ratio=0.09,
                                 shuffle=True,
                                 verbose=0,
                                 random_state=1001)
            model.fit(Xtrain, ytrain)
            sgd_scores_val = model.predict(Xval)
            sgd_RMSLE = np.sqrt(mean_squared_error(yval, sgd_scores_val))
            print('Fold %02d SGD RMSLE: %.6f' % ((i + 1), sgd_RMSLE))
            sgd_y_pred = model.predict(test)
            model1.append(model)

            model = Ridge(alpha=4.75,
                          solver='sag',
                          fit_intercept=False,
                          random_state=1001,
                          max_iter=1000)
            model.fit(Xtrain, ytrain)
            ridge_scores_val = model.predict(Xval)
            ridge_RMSLE = np.sqrt(mean_squared_error(yval, ridge_scores_val))
            print(' Fold %02d Ridge RMSLE: %.6f' % ((i + 1), ridge_RMSLE))
            ridge_y_pred = model.predict(test)
            model2.append(model)

            params = {
                'boosting': 'gbdt',
                'max_bin'          :1000,
                'max_depth': 7,
                'min_data_in_leaf': 80,
                'num_leaves': 40,
                'learning_rate': 0.75,
                'objective': 'regression',
                'metric': 'rmse',
                'nthread': 4,
                'bagging_freq': 1,
                'subsample': 0.94,
                'colsample_bytree': 0.7,
                'min_child_weight': 17,
                'is_unbalance': False,
                'verbose': -1,
                'seed': 1001
            }

            dtrain = lgb.Dataset(Xtrain, label=ytrain)
            dval = lgb.Dataset(Xval, label=yval)
            watchlist = [dtrain, dval]
            watchlist_names = ['train', 'val']

            model = lgb.train(params,
                              train_set=dtrain,
                              num_boost_round=1800,
                              valid_sets=watchlist,
                              valid_names=watchlist_names,
                              early_stopping_rounds=80,
                              verbose_eval=80)
            lgb_scores_val = model.predict(Xval)
            lgb_RMSLE = np.sqrt(mean_squared_error(yval, lgb_scores_val))
            print(' Fold %02d LightGBM RMSLE: %.6f' % ((i + 1), lgb_RMSLE))
            lgb_y_pred = model.predict(test)
            model3.append(model)

            del Xtrain, Xval
            gc.collect()

            timer(start_time)

            sgd_avpred[val_index] = sgd_scores_val
            ridge_avpred[val_index] = ridge_scores_val
            lgb_avpred[val_index] = lgb_scores_val

            if i > 0:
                sgd_fpred = sgd_pred + sgd_y_pred
                ridge_fpred = ridge_pred + ridge_y_pred
                lgb_fpred = lgb_pred + lgb_y_pred
            else:
                sgd_fpred = sgd_y_pred
                ridge_fpred = ridge_y_pred
                lgb_fpred = lgb_y_pred
            sgd_pred = sgd_fpred
            ridge_pred = ridge_fpred
            lgb_pred = lgb_fpred
            sgd_cv_sum = sgd_cv_sum + sgd_RMSLE
            ridge_cv_sum = ridge_cv_sum + ridge_RMSLE
            lgb_cv_sum = lgb_cv_sum + lgb_RMSLE

        timer(train_time)

        sgd_cv_score = (sgd_cv_sum / folds)
        ridge_cv_score = (ridge_cv_sum / folds)
        lgb_cv_score = (lgb_cv_sum / folds)
        sgd_oof_RMSLE = np.sqrt(mean_squared_error(avreal, sgd_avpred))
        ridge_oof_RMSLE = np.sqrt(mean_squared_error(avreal, ridge_avpred))
        lgb_oof_RMSLE = np.sqrt(mean_squared_error(avreal, lgb_avpred))

        print('Average SGD RMSLE:	%.6f' % sgd_cv_score)
        print(' Out-of-fold SGD RMSLE:	%.6f' % sgd_oof_RMSLE)
        print('Average Ridge RMSLE:	%.6f' % ridge_cv_score)
        print(' Out-of-fold Ridge RMSLE:	%.6f' % ridge_oof_RMSLE)
        print('Average LightGBM RMSLE:	%.6f' % lgb_cv_score)
        print(' Out-of-fold LightGBM RMSLE:	%.6f' % lgb_oof_RMSLE)
        sgd_score = round(sgd_oof_RMSLE, 6)
        ridge_score = round(ridge_oof_RMSLE, 6)
        lgb_score = round(lgb_oof_RMSLE, 6)

        sgd_mpred = sgd_pred / folds
        ridge_mpred = ridge_pred / folds
        lgb_mpred = lgb_pred / folds

        blend_time = timer(None)

        blend_train.append(sgd_avpred)
        blend_train.append(ridge_avpred)
        blend_train.append(lgb_avpred)
        blend_train = np.array(blend_train)

        blend_test.append(sgd_mpred)
        blend_test.append(ridge_mpred)
        blend_test.append(lgb_mpred)
        blend_test = np.array(blend_test)

        print('\n Finding Blending Weights ...')
        res_list = []
        weights_list = []
        #for k in range(1000):
        for k in range(20):
            starting_values = np.random.uniform(size=len(blend_train))

            #######
            # I used to think that weights should not be negative - many agree with that.
            # I've come around on that issues as negative weights sometimes do help.
            # If you don't think so, just swap the two lines below.
            #######

        #    bounds = [(0, 1)]*len(blend_train)
            bounds = [(-1, 1)] * len(blend_train)

            res = minimize(rmse_min_func,
                           starting_values,
                           method='L-BFGS-B',
                           bounds=bounds,
                           options={'disp': False,
                                    'maxiter': 100000})
            res_list.append(res['fun'])
            weights_list.append(res['x'])
            print('{iter}\tScore: {score}\tWeights: {weights}'.format(
                iter=(k + 1),
                score=res['fun'],
                weights='\t'.join([str(item) for item in res['x']])))

        bestSC = np.min(res_list)
        bestWght = weights_list[np.argmin(res_list)]
        weights = bestWght
        blend_score = round(bestSC, 6)

        print('\n Ensemble Score: {best_score}'.format(best_score=bestSC))
        print('\n Best Weights: {weights}'.format(weights=bestWght))

        train_prices = np.zeros(len(blend_train[0]))
        test_prices = np.zeros(len(blend_test[0]))

        print('\n Your final model:')
        for k in range(len(blend_test)):
            print(' %.6f * model-%d' % (weights[k], (k + 1)))
            test_prices += blend_test[k] * weights[k]

        for k in range(len(blend_train)):
            train_prices += blend_train[k] * weights[k]
        submission = test_prices
        test_chunk_count = 1
        del train
        gc.collect()
    else:
        # Drop Id column
        te_ids_new = test['test_id'].copy()
        test.drop("test_id", axis = 1, inplace = True)



        #creating a category for item descriptions that are empty or meaningless


        test.item_description = test.item_description.fillna("No description yet")
        test['has_description'] = test.item_description.apply(lambda row : DescriptionNotMeaningless(row))

        test.brand_name = test.brand_name.fillna("No brand name given")
        timer(start_time)

        test.brand_name.replace([name for name in test.brand_name if name not in brand_names], 'No brand name given', inplace=True)
        timer(start_time)

        test['has_brand'] = test.brand_name.apply(lambda row : DescriptionNotMeaningless(row))


        test.category_name = test.category_name.fillna("No category name given")
        test.category_name.replace([name for name in test.category_name if name not in category_names], 'No category name given', inplace=True)
        test['has_category'] = test.category_name.apply(lambda row : DescriptionNotMeaningless(row))

        #brand price
        test['brand_price'] = test.brand_name.replace(mapping)

        timer(start_time)

        #category price
        test['category_price'] = test.category_name.replace(mapping2)

        timer(start_time)



        test['item_description'] = test['item_description'].astype('category')
        test['brand_name'] = test['brand_name'].astype('category')
        test['has_brand'] = test['has_brand'].astype('category')
        test['has_description'] = test['has_description'].astype('category')
        test['has_category'] = test['has_category'].astype('category')
        test['category_name'] = test['category_name'].astype('category')


        test.loc[:, numerical_features] = stdSc.transform(test.loc[:, numerical_features])
        df = test
        timer(start_time)


        print('Encodings')
        X_name = count.transform(df['name'])


        print('Category Encoders')
        X_category = count_category.transform(df['category_name'])

        print('Descp encoders')
        X_descp = count_descp.transform(df['item_description'])

        print('Brand encoders')
        X_brand = vect_brand.transform(df['brand_name'])


        print('Dummy Encoders')
        X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(df[[
            'item_condition_id', 'has_description', 'has_brand', 'has_category', 'shipping'
        ]], sparse=True).values)

        df_clean = scipy.sparse.hstack((X_dummies, X_brand, X_category, X_descp, X_name,
                                         scipy.sparse.csr_matrix(df['brand_price']).T,
                                         scipy.sparse.csr_matrix(df['category_price']).T)).tocsr()

        test = df_clean

        print(test.shape)
        timer(start_time)
        test_chunk_count = test_chunk_count+1
        
        sgd_pred = sum([model.predict(test) for model in model1])
        
        ridge_pred = sum([model.predict(test) for model in model2])
        
        lgb_pred = sum([model.predict(test) for model in model3])
        
        sgd_mpred = sgd_pred / folds
        ridge_mpred = ridge_pred / folds
        lgb_mpred = lgb_pred / folds
        
        blend_test = []
        blend_test.append(sgd_mpred)
        blend_test.append(ridge_mpred)
        blend_test.append(lgb_mpred)
        blend_test = np.array(blend_test)

        test_prices = np.zeros(len(blend_test[0]))
        for k in range(len(blend_test)):
            print(' %.6f * model-%d' % (weights[k], (k + 1)))
            test_prices += blend_test[k] * weights[k]
        submission = np.concatenate([submission, test_prices], axis=0)
        te_ids = np.concatenate([te_ids, te_ids_new], axis=0)
#save for submission
submission = np.expm1(submission)
submission = pd.DataFrame({
        "test_id": te_ids,
        "price": submission
    })
submission.to_csv('submission.csv', index=False)
timer(blend_time)
timer(start_time)