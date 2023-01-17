import gc

import time

import numpy as np

import pandas as pd



from scipy.sparse import csr_matrix, hstack



from sklearn.linear_model import Ridge

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelBinarizer, StandardScaler

from sklearn.model_selection import train_test_split

import lightgbm as lgb



NUM_BRANDS = 4000

NUM_CATS = 4000

NAME_MIN_DF = 10

MAX_FEATURES_ITEM_DESCRIPTION = 50000





def handle_missing_inplace(dataset):

    dataset['category_name'].fillna(value='missing', inplace=True)

    dataset['brand_name'].fillna(value='missing', inplace=True)

    dataset['item_description'].fillna(value='missing', inplace=True)



def cutting(dataset):

    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]

    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'

    pop_category = dataset['category_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATS]

    dataset.loc[~dataset['category_name'].isin(pop_category), 'category_name'] = 'missing'



def to_categorical(dataset):

    dataset['category_name'] = dataset['category_name'].astype('category')

    dataset['brand_name'] = dataset['brand_name'].astype('category')

    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')



start_time = time.time()

train = pd.read_table('../input/train.tsv', engine='c')

test = pd.read_table('../input/test.tsv', engine='c')

print('[{}] Finished to load data'.format(time.time() - start_time))



nrow_train = train.shape[0]

y = np.log1p(train["price"])

merge: pd.DataFrame = pd.concat([train, test])

submission: pd.DataFrame = test[['test_id']]

del train

del test

gc.collect()



handle_missing_inplace(merge)

print('[{}] Finished to handle missing'.format(time.time() - start_time))

cutting(merge)

print('[{}] Finished to cut'.format(time.time() - start_time))

to_categorical(merge)

print('[{}] Finished to convert categorical'.format(time.time() - start_time))

cv = CountVectorizer(min_df=NAME_MIN_DF)

X_name = cv.fit_transform(merge['name'])

print('[{}] Finished count vectorize `name`'.format(time.time() - start_time))



cv = CountVectorizer()

X_category = cv.fit_transform(merge['category_name'])

print('[{}] Finished count vectorize `category_name`'.format(time.time() - start_time))



tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,

                     ngram_range=(1, 3),

                     stop_words='english')

X_description = tv.fit_transform(merge['item_description'])

print('[{}] Finished TFIDF vectorize `item_description`'.format(time.time() - start_time))



lb = LabelBinarizer(sparse_output=True)

X_brand = lb.fit_transform(merge['brand_name'])

print('[{}] Finished label binarize `brand_name`'.format(time.time() - start_time))



X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],

                                      sparse=True).values)

print('[{}] Finished to get dummies on `item_condition_id` and `shipping`'.format(time.time() - start_time))



sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()

print('[{}] Finished to create sparse merge'.format(time.time() - start_time))



X = sparse_merge[:nrow_train]

X_test = sparse_merge[nrow_train:]
# Make a 20% holdout set so that we can benchmark our implementations.

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 144)

# I've found that in practice that Public Leaderboard scores are about 0.003 better than holdout

# scores (e.g., 0.468 in holdout will be about 0.465 on Public Leadearboard).

#

# I've also found locally on my laptop that the holdout score is very close to the 5-fold

# CV score and that fold variance is quite low. Good for us!



def rmse(predicted, actual):

    return np.sqrt(((predicted - actual) ** 2).mean())



def test_model(model):

    start_time = time.time()

    model.fit(X_train, y_train)

    train_finished = time.time()

    preds = model.predict(X_valid)

    print('RMSLE %f, train in %.4f sec, predict in %.4f sec' % (rmse(preds, y_valid), (train_finished - start_time), (time.time() - train_finished)))



for solver in ['lsqr', 'sparse_cg', 'sag', 'saga']:

    print('Solver =', solver)

    intercept = True if solver == 'sag' else False # Sklearn says only sag can fit intercept. Maybe saga can too?

    for alpha in [0.01, 0.03, 0.05, 0.1, 0.5, 1, 3, 5, 10]:

        print('Alpha =', alpha)

        model = Ridge(alpha=alpha, copy_X=True, fit_intercept=intercept, max_iter=100,

                      normalize=False, random_state=101, solver=solver, tol=0.01)

        test_model(model)
# To be more precise, focus on the best alphas with lower tolerance and no iteration cap.



# Though it looks like the same relationship between alpha and score does not hold, and that

# scores are not actually guaranteed to be better!



for solver in ['lsqr', 'sag', 'saga']:

    print('Solver =', solver)

    intercept = True if solver == 'sag' else False

    for alpha in [1, 3, 5]:

        print('Alpha =', alpha)

        model = Ridge(alpha=alpha, copy_X=True, fit_intercept=intercept, max_iter=None,

                      normalize=False, random_state=101, solver=solver, tol=0.001)

        test_model(model)

    

# Room for further improvement is left as an excercise for the reader. :D