import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

from datetime import datetime

from scipy.special import logsumexp



from catboost import Pool, cv, CatBoostClassifier, CatBoostRegressor

from sklearn.metrics import mean_squared_error, classification_report
train = pd.read_csv("/kaggle/input/caltech-cs155-2020/train.csv")

test = pd.read_csv("/kaggle/input/caltech-cs155-2020/test.csv")

df = pd.concat([train,test],sort=False)

print(df.shape)

print(df.columns)

df.tail()
train.head()
test.head()
## y is binary.

display(train["y"].describe())
bid_cols = ['bid1','bid2', 'bid3', 'bid4', 'bid5']

bid_vol_cols = ['bid1vol', 'bid2vol', 'bid3vol', 'bid4vol', 'bid5vol']

ask_cols = ['ask1', 'ask2', 'ask3', 'ask4', 'ask5',]

ask_vol_cols = ['ask1vol','ask2vol', 'ask3vol', 'ask4vol', 'ask5vol']



group_cols = {"bid_cols":bid_cols,"bid_vol_cols":bid_vol_cols,"ask_cols":ask_cols,"ask_vol_cols":ask_vol_cols}
for group in group_cols.keys():

    print(group)

    df[f"{group}_max"] = df[group_cols[group]].max(axis=1)

    df[f"{group}_min"] = df[group_cols[group]].min(axis=1)

    df[f"{group}_spread"] = df[f"{group}_max"].div(df[f"{group}_min"])

    df[f"{group}_logsumexp"] = df[group_cols[group]].apply(logsumexp)

    

    df[f"{group}_max"] = df[group_cols[group]].max(axis=1)

    

df["last_price_div__mid"] = df["last_price"].div(df["mid"])
df["date"] = pd.to_datetime("1.1.2019")

df["date"] = df["date"] + pd.to_timedelta(df["id"]/2,unit="s") # 500 ms per row



df["date"].describe()
train = df.loc[~df.y.isna()]

print(f"train shape {train.shape[0]}")

test = df.loc[df.y.isna()]

print(f"test shape {test.shape[0]}")
train.drop(["id"],axis=1).to_csv("train_hft.csv.gz",index=False,compression="gzip")

test.to_csv("test_hft_nodates.csv.gz",index=False,compression="gzip")
# we don't know if the test set has a temporal split, so we'll just try a random split for now

X = train.drop(["id","date","y"],axis=1)

y = train["y"]
train_pool = Pool(data=X,label = y)
# ### hyperparameter tuning example grid for catboost : 

# grid = {'learning_rate': [0.05, 0.1],

#         'depth': [6, 11],

# #         'l2_leaf_reg': [1, 3,9],

# #        "iterations": [1000],

#        "custom_metric":['Logloss', 'AUC']}



# model = CatBoostClassifier()



# ## can also do randomized search - more efficient typically, especially for large search space - `randomized_search`

# grid_search_result = model.grid_search(grid, 

#                                        train_pool,

#                                        plot=True,

#                                        refit = True, #  refit best model on all data

#                                       partition_random_seed=42)



# print(model.get_best_score())
model = CatBoostClassifier()

    

model.fit(train_pool, plot=True,silent=True)

print(model.get_best_score())
feature_importances = model.get_feature_importance(train_pool)

feature_names = X.columns

for score, name in sorted(zip(feature_importances, feature_names), reverse=True):

    if score > 0.2:

        print('{0}: {1:.2f}'.format(name, score))
import shap

shap.initjs()



explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(train_pool)



# visualize the training set predictions

# SHAP plots for all the data is very slow, so we'll only do it for a sample. Taking the head instead of a random sample is dangerous! 

shap.force_plot(explainer.expected_value,shap_values[0,:300], X.iloc[0,:300])
# summarize the effects of all the features

shap.summary_plot(shap_values, X)
## todo : PDP features +- from shap
test["Predicted"] = model.predict(test.drop(["id","date","y"],axis=1),prediction_type='Probability')[:,1]

test[["id","Predicted"]].to_csv("submission.csv",index=False)