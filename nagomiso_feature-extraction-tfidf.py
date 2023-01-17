import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfTransformer

import lightgbm as lgb
pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)



dtypes = {f"feat_{i}": "int32" for i in range(1, 94)}

dtypes["id"] = "int32"

dtypes["target"] = "string"
df_train = pd.read_csv(

    "/kaggle/input/otto-group-product-classification-challenge/train.csv",

    dtype=dtypes

).set_index("id")
class_to_order = dict()

order_to_class = dict()



for idx, col in enumerate(df_train.target.unique()):

    order_to_class[idx] = col

    class_to_order[col] = idx



df_train["target_ord"] = df_train["target"].map(class_to_order).astype("int16")

feature_columns = [

    col for col in df_train.columns if col.startswith("feat_")

]

target_column = ["target_ord"]
X_train, X_valid, y_train, y_valid = train_test_split(

    df_train[feature_columns], df_train[target_column],

    test_size=0.3, random_state=42,

    stratify=df_train[target_column]

)
tfidf = TfidfTransformer()

tfidf_feature_train = tfidf.fit_transform(X_train).toarray().astype("float32")

tfidf_feature_valid = tfidf.transform(X_valid).toarray().astype("float32")
X_train_tfidf = np.hstack((X_train.values, tfidf_feature_train))

X_valid_tfidf = np.hstack((X_valid.values, tfidf_feature_valid))
params = {

    'objective': "multiclass",

    'metric': {"multi_logloss"},

    'num_class': 9,

    'seed': 42,

    'lambda_l1': 0.0036682603550733813,

    'lambda_l2': 8.924549306063208,

    'num_leaves': 113,

    'feature_fraction': 0.48000000000000004,

    'bagging_fraction': 1.0,

    'bagging_freq': 0,

    'min_child_samples': 20

}
dataset_train = lgb.Dataset(X_train_tfidf, y_train)

dataset_valid = lgb.Dataset(X_valid_tfidf, y_valid)
booster = lgb.train(

    params,

    dataset_train,

    feature_name=(

        [f"feat_{i}" for i in range(1, 94)]

        + [f"tfidf_{i}" for i in range(1, 94)]

    ),

    num_boost_round=500,

    valid_sets=dataset_valid,

    early_stopping_rounds=20,

)
best_iteration = booster.best_iteration

print(best_iteration)
lgb.plot_importance(

    booster,

    max_num_features=30,

    figsize=(12, 10),

    dpi=300,

);
df_test = pd.read_csv(

    "/kaggle/input/otto-group-product-classification-challenge/test.csv",

    dtype=dtypes

).set_index("id")
tfidf = TfidfTransformer()



tfidf_feature_train_all = tfidf.fit_transform(df_train[feature_columns]).toarray().astype("float32")

X_train_all_tfidf = np.hstack((df_train[feature_columns].values, tfidf_feature_train_all))

dataset_train_all = lgb.Dataset(X_train_all_tfidf, df_train[target_column])
booster = lgb.train(

    params,

    dataset_train_all,

    feature_name=(

        [f"feat_{i}" for i in range(1, 94)]

        + [f"tfidf_{i}" for i in range(1, 94)]

    ),

    num_boost_round=best_iteration,

)
df_test
tfidf_feature_test = tfidf.transform(df_test).toarray()

X_test_tfidf = np.hstack((df_test[feature_columns].values, tfidf_feature_test))
pred = booster.predict(X_test_tfidf)
for idx, col in order_to_class.items():

    df_test[col] = pred[:,idx]
df_test[[f"Class_{i}" for i in range(1, 10)]].to_csv('submission.csv', index=True)