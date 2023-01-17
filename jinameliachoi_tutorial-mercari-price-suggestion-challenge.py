# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.linear_model import Ridge, Lasso, LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



df = pd.read_csv('/kaggle/input/mercari-price-suggestion-challenge/train.tsv', sep='\t')

print(df.shape)

df.head(3)
# 피처 타입과 null 값 확인하기

print(df.info())
# target 분포 확인하기

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



y_train_df = df['price']

plt.figure(figsize=(6, 4))

sns.distplot(y_train_df, kde=False)
# 왜곡되어 있는 값이므로 log 취해보기

y_train_df = np.log1p(df['price'])

sns.distplot(y_train_df, kde=False)
# 정규분포와 유사한 분포를 띔

# 데이터 세트에서 원래 값을 로그로 변환된 값으로 수정

df['price'] = np.log1p(df['price'])

df['price'].head(3)
# 다른 피처값 살펴보기

print('Shipping 값 유형:\n', df['shipping'].value_counts())

print('item_condition_id 값 유형:\n', df['item_condition_id'].value_counts())
# `item_description` 값은 Null 값이 별로 없지만, description에 별도 설명이 없는 경우 'No description yet' 값으로 이뤄짐

boolean_cond = df['item_description'] == 'No description yet'

df[boolean_cond]['item_description'].count()
# 위의 값 또한 사용할 수 없는 값이기 때문에 적절한 값으로 변경이 필요합니다.

# category_name 은 대/중/소분류로 '/'를 기준으로 정보가 나뉘어져 있습니다

# category_name에서 '/'을 기준으로 단어를 토큰화해 각각 별도의 피처로 저장하고 알고리즘을 학습시키겠습니다



# apply lambda에서 호출되는 대, 중, 소 분할 함수 생성. 대, 중, 소 값을 리스트로 반환

def split_cat(category_name):

    try:

        return category_name.split('/')

    except:

        return ['Other_Null', 'Other_Null', 'Other_Null']

    

# 위의 split_cat()을 apply lambda에서 호출해 칼럼을 df에 생성

df['cat_dae'], df['cat_jung'], df['cat_so'] = zip(*df['category_name'].apply(lambda x: split_cat(x)))



# 대분류만 값의 유형과 건수를 살펴보고, 중, 소분류는 값이 유형이 많으므로 분류 개수만 추출

print('대분류 유형: \n', df['cat_dae'].value_counts())

print('중분류 유형: \n', df['cat_jung'].nunique())

print('소분류 유형: \n', df['cat_so'].nunique())
# 마지막으로 일괄적으로 null 값을 'Other Null'로 동일하게 변경합니다

df['brand_name'] = df['brand_name'].fillna(value='Other_Null')

df['category_name'] = df['category_name'].fillna(value='Other_Null')

df['item_description'] = df['item_description'].fillna(value='Other_Null')



df.isnull().sum()
print('brand name의 유형 건수: ', df['brand_name'].nunique())

print('brand name sample 5건:\n', df['brand_name'].value_counts()[:5])
print('name의 유형 건수: ', df['name'].nunique())

print('name sample 5건:\n', df['name'].value_counts()[:5])
pd.set_option('max_colwidth', 200)



# item_description의 평균 문자열 크기

print('item_description 평균 문자열 크기:', df['item_description'].str.len().mean())

df['item_description'][:2]
# name 속성에 대한 피처 벡터화 변환

cnt_vec = CountVectorizer()

X_name = cnt_vec.fit_transform(df.name)



# item_description 속성에 대한 피처 벡터화 변환

tfidf_descp = TfidfVectorizer(max_features=50000,

                             ngram_range = (1,3),

                             stop_words = 'english')

X_descp = tfidf_descp.fit_transform(df['item_description'])



print('name vectorization shape: ', X_name.shape)

print('item_description vectorization shape: ', X_descp.shape)
# 인코딩 대상 칼럼을 모두 LabelBinarizer로 원-핫 인코딩 변환을 합니다

from sklearn.preprocessing import LabelBinarizer



lb_brand_name = LabelBinarizer(sparse_output=True) # 원핫인코딩 지정 파라미터

X_brand = lb_brand_name.fit_transform(df['brand_name'])



lb_item_cond_id = LabelBinarizer(sparse_output=True)

X_item_cond_id = lb_item_cond_id.fit_transform(df['item_condition_id'])



lb_shipping = LabelBinarizer(sparse_output=True)

X_shipping = lb_shipping.fit_transform(df['shipping'])



lb_cat_dae = LabelBinarizer(sparse_output=True) 

X_cat_dae = lb_cat_dae.fit_transform(df['cat_dae'])



lb_cat_jung = LabelBinarizer(sparse_output=True) 

X_cat_jung = lb_cat_jung.fit_transform(df['cat_jung'])



lb_cat_so = LabelBinarizer(sparse_output=True)

X_cat_so = lb_cat_so.fit_transform(df['cat_so'])
# 제대로 인코딩 됐는지 확인하기

print(type(X_brand), type(X_item_cond_id), type(X_shipping))

print('X_brand shape:{0}, X_item_cond_id shape:{1}, X_shipping shape:{2}'.format(X_brand.shape, X_item_cond_id.shape, X_shipping.shape))



print(type(X_cat_dae), type(X_cat_jung), type(X_cat_so))

print('X_cat_dae shape:{0}, X_cat_jung shape:{1}, X_cat_so shape:{2}'.format(X_cat_dae.shape, X_cat_jung.shape, X_cat_so.shape))

from scipy.sparse import hstack

import gc



sparse_matrix_list = (X_name, X_descp, X_brand, X_item_cond_id,

                     X_shipping, X_cat_dae, X_cat_jung, X_cat_so)



X_features_sparse = hstack(sparse_matrix_list).tocsr()

print(type(X_features_sparse), X_features_sparse.shape)



del X_features_sparse

gc.collect() # 결합 데이터를 메모리에서 삭제
# price 값이 로그로 변환된 값이기 때문에 지수 변환을 수행해주어 RMSLE 적용하게 함수 만듬

def rmsle(y, y_pred):

    # underflow, overflow를 막기 위해 log1p로 rmsle 계산

    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y_pred), 2)))



def evaluate_org_price(y_test, preds):

    preds_exmpm = np.expm1(preds)

    y_test_exmpm = np.expm1(y_test)

    

    # rmsle로 값 추출

    rmsle_result = rmsle(y_test_exmpm, preds_exmpm)

    return rmsle_result
# 학습용 데이터를 생성하고, 모델을 학습/예측하는 로직을 별도 함수로 만듬

def model_train_predict(model, matrix_list):

    X = hstack(matrix_list).tocsr()

    X_train, X_test, y_train, y_test = train_test_split(X, df['price'],

                                                       test_size=.2, random_state=156)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    

    del X, X_train, X_test, y_train

    gc.collect()

    

    return preds, y_test
# Ridge



linear_model = Ridge(solver='lsqr', fit_intercept=False)



sparse_matrix_list = (X_name, X_brand, X_item_cond_id,

                     X_shipping, X_cat_dae, X_cat_jung, X_cat_so)



linear_preds, y_test = model_train_predict(model = linear_model,

                                          matrix_list = sparse_matrix_list)

print('Item description을 제외했을 때 rmsle 값: ', evaluate_org_price(y_test, linear_preds))





sparse_matrix_list = (X_name, X_descp, X_brand, X_item_cond_id,

                     X_shipping, X_cat_dae, X_cat_jung, X_cat_so)



linear_preds, y_test = model_train_predict(model = linear_model,

                                          matrix_list = sparse_matrix_list)

print('Item description을 포함했을 때 rmsle 값: ', evaluate_org_price(y_test, linear_preds))
from lightgbm import LGBMRegressor



sparse_matrix_list = (X_name, X_descp, X_brand, X_item_cond_id,

                     X_shipping, X_cat_dae, X_cat_jung, X_cat_so)



lgbm_model = LGBMRegressor(n_estimators=200,

                          learning_rate=0.5,

                          num_leaves=125,

                          random_state=156)



lgbm_preds, y_test = model_train_predict(model = lgbm_model,

                                          matrix_list = sparse_matrix_list)

print('Item description을 포함했을 때 rmsle 값: ', evaluate_org_price(y_test, lgbm_preds))