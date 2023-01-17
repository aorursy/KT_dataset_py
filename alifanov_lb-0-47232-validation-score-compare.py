# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import scipy

import time



from sklearn.linear_model import Ridge, LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelBinarizer



from scipy.sparse import csr_matrix, hstack
np.random.seed(7)



NUM_BRANDS = 2500

NAME_MIN_DF = 10

MAX_FEATURES_ITEM_DESCRIPTION = 50000



start_time = time.time()

tcurrent   = start_time
def rmsle(y, y0):

    return np.sqrt(np.mean(np.square(y - y0)))



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



print('[{}] Finished to load data'.format(time.time() - start_time))

print('Train shape: ', train.shape)

print('Test shape: ', test.shape)
nrow_train = train.shape[0]

y = np.log1p(train["price"])

merge = pd.concat([train, test])

submission = test[['test_id']]
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

                         ngram_range=(2, 2),

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



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=7)
model = Ridge(solver="sag", fit_intercept=True, random_state=7, alpha=3)

model.fit(X_train, y_train)

print('[{}] Finished to train ridge sag'.format(time.time() - start_time))

preds = model.predict(X=X_test)

print('[{}] Finished to predict ridge sag'.format(time.time() - start_time))
print('RMSLE: {}'.format(rmsle(y_val, model.predict(X_val))))
submission['price'] = np.expm1(preds)

submission.to_csv("sample_submission.csv", index=False)