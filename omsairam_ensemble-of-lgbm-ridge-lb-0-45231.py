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



from sklearn.linear_model import Ridge, LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelBinarizer

import lightgbm as lgb



import gc
NUM_BRANDS = 2500

NAME_MIN_DF = 10

MAX_FEAT_DESCP = 50000



print("Reading in Data")



df_train = pd.read_csv('../input/train.tsv', sep='\t')

df_test = pd.read_csv('../input/test.tsv', sep='\t')



df = pd.concat([df_train, df_test], 0)

nrow_train = df_train.shape[0]

y_train = np.log1p(df_train["price"])



del df_train

gc.collect()



print(df.memory_usage(deep = True))
df["category_name"] = df["category_name"].fillna("Other").astype("category")

df["brand_name"] = df["brand_name"].fillna("unknown")



pop_brands = df["brand_name"].value_counts().index[:NUM_BRANDS]

df.loc[~df["brand_name"].isin(pop_brands), "brand_name"] = "Other"



df["item_description"] = df["item_description"].fillna("None")

df["item_condition_id"] = df["item_condition_id"].astype("category")

df["brand_name"] = df["brand_name"].astype("category")



print(df.memory_usage(deep = True))
print("Encodings")

count = CountVectorizer(min_df=NAME_MIN_DF)

X_name = count.fit_transform(df["name"])



print("Category Encoders")

unique_categories = pd.Series("/".join(df["category_name"].unique().astype("str")).split("/")).unique()

count_category = CountVectorizer()

X_category = count_category.fit_transform(df["category_name"])



print("Descp encoders")

count_descp = TfidfVectorizer(max_features = MAX_FEAT_DESCP, 

                              ngram_range = (1,3),

                              stop_words = "english")

X_descp = count_descp.fit_transform(df["item_description"])



print("Brand encoders")

vect_brand = LabelBinarizer(sparse_output=True)

X_brand = vect_brand.fit_transform(df["brand_name"])



print("Dummy Encoders")

X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(df[[

    "item_condition_id", "shipping"]], sparse = True).values)
X = scipy.sparse.hstack((X_dummies, 

                         X_descp,

                         X_brand,

                         X_category,

                         X_name)).tocsr()



print([X_dummies.shape, X_category.shape, 

       X_name.shape, X_descp.shape, X_brand.shape])



X_train = X[:nrow_train]

X_test = X[nrow_train:]
params = {

    'learning_rate': 0.75,

    'application': 'regression',

    'max_depth': 3,

    'num_leaves': 100,

    'verbosity': -1,

    'metric': 'RMSE',

}





train_X, valid_X, train_y, valid_y = train_test_split(X_train, y_train, test_size = 0.1, random_state = 144) 

d_train = lgb.Dataset(train_X, label=train_y, max_bin=8192)

d_valid = lgb.Dataset(valid_X, label=valid_y, max_bin=8192)

watchlist = [d_train, d_valid]



model = lgb.train(params, train_set=d_train, num_boost_round=2200, valid_sets=watchlist, \

early_stopping_rounds=50, verbose_eval=100) 

preds = model.predict(X_test)



model = Ridge(solver = "lsqr", fit_intercept=False)



print("Fitting Model")

model.fit(X_train, y_train)



preds += model.predict(X_test)

preds /= 2
df_test["price"] = np.expm1(preds)

df_test[["test_id", "price"]].to_csv("submission_LGBM_Ridge_3.csv", index = False)
import numpy as np

import pandas as pd

import scipy



from sklearn.linear_model import Ridge, LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelBinarizer



import gc



NUM_BRANDS = 2500

NAME_MIN_DF = 10

MAX_FEAT_DESCP = 10000



print("Reading in Data")



df_train2 = pd.read_csv('../input/train.tsv', sep='\t')

df_test2 = pd.read_csv('../input/test.tsv', sep='\t')



df = pd.concat([df_train2, df_test2], 0)

nrow_train = df_train2.shape[0]

y_train = np.log(df_train2["price"]+1)



del df_train2

gc.collect()



print(df.memory_usage(deep = True))



df["category_name"] = df["category_name"].fillna("Other").astype("category")

df["brand_name"] = df["brand_name"].fillna("unknown")



pop_brands = df["brand_name"].value_counts().index[:NUM_BRANDS]

df.loc[~df["brand_name"].isin(pop_brands), "brand_name"] = "Other"



df["item_description"] = df["item_description"].fillna("None")

df["item_condition_id"] = df["item_condition_id"].astype("category")

df["brand_name"] = df["brand_name"].astype("category")



print(df.memory_usage(deep = True))



print("Encodings")

count = CountVectorizer(min_df=NAME_MIN_DF)

X_name = count.fit_transform(df["name"])



print("Category Encoders")

unique_categories = pd.Series("/".join(df["category_name"].unique().astype("str")).split("/")).unique()

count_category = CountVectorizer()

X_category = count_category.fit_transform(df["category_name"])



print("Descp encoders")

count_descp = TfidfVectorizer(max_features = MAX_FEAT_DESCP, 

                              ngram_range = (1,3),

                              stop_words = "english")

X_descp = count_descp.fit_transform(df["item_description"])



print("Brand encoders")

vect_brand = LabelBinarizer(sparse_output=True)

X_brand = vect_brand.fit_transform(df["brand_name"])



print("Dummy Encoders")

X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(df[[

    "item_condition_id", "shipping"]], sparse = True).values)



X = scipy.sparse.hstack((X_dummies, 

                         X_descp,

                         X_brand,

                         X_category,

                         X_name)).tocsr()



print([X_dummies.shape, X_category.shape, 

       X_name.shape, X_descp.shape, X_brand.shape])



X_train = X[:nrow_train]

model = Ridge(solver = "lsqr", fit_intercept=False)



print("Fitting Model")

model.fit(X_train, y_train)



X_test = X[nrow_train:]

preds1 = model.predict(X_test)



#submission = df_test2#[["test_id"]]

#df_test2["price1"] =preds1







df_test2["price1"] = np.exp(preds1)-1







#df_test2[["test_id", "price"]].to_csv("submission_ridge.csv", index = False)

type(X_test)
X=X_train 

y=y_train



def rmsle(predictions, targets):

    predictions = np.exp(predictions) - 1

    targets = np.exp(targets) - 1

    return np.sqrt(((predictions - targets) ** 2).mean())



def rmsle_lgb(labels, preds):

    return 'rmsle', rmsle(preds, labels), False



print('Training model...')

from sklearn.model_selection import train_test_split



from lightgbm import LGBMRegressor

X_train, X_test2, y_train, y_test2 = train_test_split(X, y, test_size=0.3, random_state=42)

lgbm_params = {'n_estimators': 1000, 'learning_rate': 0.4, 'max_depth': 10,

               'num_leaves': 31, 'subsample': 0.9, 'colsample_bytree': 0.8,

               'min_child_samples': 50, 'n_jobs': 4}

model = LGBMRegressor(**lgbm_params)

model.fit(X_train, y_train,

         eval_set=[(X_test2, y_test2)],

         eval_metric=rmsle_lgb,

         early_stopping_rounds=100,

         verbose=True)



print('Generating submission...')



preds2 = model.predict(X_test)

df_test2['price2']=np.exp(preds2) - 1



df_test2['price']=(df_test2['price2']+df_test2['price1'])/2

df_test2[["test_id", "price"]].to_csv("submission.csv", index = False)

df_test[["test_id", "price"]].head()
df_test2[["test_id", "price"]].head()
new = df_test[["test_id", "price"]].copy()

new.head()
new2 = df_test2[["test_id", "price"]].copy()

new2.head()
new['price'] = new2['price']*.6 + new['price']*.4

new.head()
new.to_csv("lgbm_ridge_ensemble2.csv", float_format='%.4f', index=None)