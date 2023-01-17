# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
hist_df = pd.read_csv('../input/historical_transactions.csv')
for feat in ['authorized_flag', 'installments', 'category_1', 'category_2', 'category_3', 'month_lag']:
    print (feat, len(hist_df[feat].unique()))
print(hist_df.month_lag.unique(), hist_df.installments.unique())
feats = []
for feat in ['authorized_flag', 'category_1', 'category_3', 'category_2']:
    for feat_val in hist_df[feat].unique():
        name = feat + ":" + (str(feat_val))
        
        hist_df[name] = hist_df[feat].apply(lambda x: 1 if x == feat_val else 0)
        
        gdf = hist_df.groupby("card_id")[name].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
        gdf.columns = ["card_id", "sum_{}".format(name), "mean_{}".format(name), "std_{}".format(name), "min_{}".format(name), "max_{}".format(name)]
        hist_df.drop(name, axis=1, inplace=True)
        
        train_df = pd.merge(train_df, gdf, on="card_id", how="left")
        test_df = pd.merge(test_df, gdf, on="card_id", how="left")
        
        del gdf
        import gc; gc.collect()
        import time; time.sleep(10)
        
        for u in ['sum', 'mean', 'std', 'min', 'max']:
            feats.append("{}_{}".format(u, name))
        print ("{} done".format(name))
gdf = hist_df.groupby("card_id")
gdf = gdf["installments"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = ["card_id", "sum_installments", "mean_installments", "std_installments", "min_installments", "max_installments"]
train_df = pd.merge(train_df, gdf, on="card_id", how="left")
test_df = pd.merge(test_df, gdf, on="card_id", how="left")

gdf = hist_df.groupby("card_id")
gdf = gdf["month_lag"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = ["card_id", "sum_month_lag", "mean_month_lag", "std_month_lag", "min_month_lag", "max_month_lag"]
train_df = pd.merge(train_df, gdf, on="card_id", how="left")
test_df = pd.merge(test_df, gdf, on="card_id", how="left")

gdf = hist_df.groupby("card_id")
gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = ["card_id", "sum_hist_trans", "mean_hist_trans", "std_hist_trans", "min_hist_trans", "max_hist_trans"]
train_df = pd.merge(train_df, gdf, on="card_id", how="left")
test_df = pd.merge(test_df, gdf, on="card_id", how="left")
del hist_df, gdf
import gc; gc.collect()
new_trans_df = pd.read_csv("../input/new_merchant_transactions.csv")
new_trans_df.head()
for feat in ['authorized_flag', 'category_1', 'category_3', 'category_2']:
    for feat_val in new_trans_df[feat].unique():
        name = 'new_' + feat + ":" + (str(feat_val))
        
        new_trans_df[name] = new_trans_df[feat].apply(lambda x: 1 if x == feat_val else 0)
        
        gdf = new_trans_df.groupby("card_id")[name].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
        gdf.columns = ["card_id", "sum_{}".format(name), "mean_{}".format(name), "std_{}".format(name), "min_{}".format(name), "max_{}".format(name)]
        new_trans_df.drop(name, axis=1, inplace=True)
        
        train_df = pd.merge(train_df, gdf, on="card_id", how="left")
        test_df = pd.merge(test_df, gdf, on="card_id", how="left")
        
        del gdf
        import gc; gc.collect()
        import time; time.sleep(5)
        
        for u in ['sum', 'mean', 'std', 'min', 'max']:
            feats.append("{}_{}".format(u, name))
        print ("{} done".format(name))
gdf = new_trans_df.groupby("card_id")
gdf = gdf["installments"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = ["card_id", "new_sum_installments", "new_mean_installments", "new_std_installments", "new_min_installments", "new_max_installments"]
train_df = pd.merge(train_df, gdf, on="card_id", how="left")
test_df = pd.merge(test_df, gdf, on="card_id", how="left")

gdf = new_trans_df.groupby("card_id")
gdf = gdf["month_lag"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = ["card_id", "new_sum_month_lag", "new_mean_month_lag", "new_std_month_lag", "new_min_month_lag", "new_max_month_lag"]
train_df = pd.merge(train_df, gdf, on="card_id", how="left")
test_df = pd.merge(test_df, gdf, on="card_id", how="left")

gdf = new_trans_df.groupby("card_id")
gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = ["card_id", "new_sum_hist_trans", "new_mean_hist_trans", "new_std_hist_trans", "new_min_hist_trans", "new_max_hist_trans"]
train_df = pd.merge(train_df, gdf, on="card_id", how="left")
test_df = pd.merge(test_df, gdf, on="card_id", how="left")
del new_trans_df, gdf
import gc; gc.collect()
feat_cols = feats + \
    ["sum_installments", "mean_installments", "std_installments", "min_installments", "max_installments"] + \
    ["sum_month_lag", "mean_month_lag", "std_month_lag", "min_month_lag", "max_month_lag"] + \
    ["sum_hist_trans", "mean_hist_trans", "std_hist_trans", "min_hist_trans", "max_hist_trans"] + \
    ["new_sum_installments", "new_mean_installments", "new_std_installments", "new_min_installments", "new_max_installments"] + \
    ["new_sum_month_lag", "new_mean_month_lag", "new_std_month_lag", "new_min_month_lag", "new_max_month_lag"] + \
    ["new_sum_hist_trans", "new_mean_hist_trans", "new_std_hist_trans", "new_min_hist_trans", "new_max_hist_trans"]
feats_subset = [feat for feat in feat_cols if feat in train_df.columns]

print(len(feat_cols), len(feats_subset))

X = train_df[feats_subset].values
y = train_df["target"].values

test_X = test_df[feats_subset].values
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.1, random_state=2018)
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(
    loss='ls', 
    n_estimators=100,  
    max_depth=4, 
    verbose=2, 
    random_state=2018
)
model.fit(np.nan_to_num(train_X), train_y)
dict(zip(feat_cols, model.feature_importances_))
np.sqrt(np.mean(np.square(model.predict(np.nan_to_num(val_X)) - val_y))), np.sqrt(np.mean(np.square(np.mean(train_y) - val_y)))
sub_df = pd.DataFrame({"card_id":test_df["card_id"].values})
sub_df["target"] = model.predict(np.nan_to_num(test_X))
sub_df.to_csv("submission.csv", index=False)
