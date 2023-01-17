!pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887 2>/dev/null 1>/dev/null
from fastai.structured import rf_feat_importance

from fastai.structured import train_cats,proc_df
import pandas as pd

import numpy as np 

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV
import eli5

from sklearn import linear_model

import lightgbm as lgb
train = pd.read_csv("../input/train.csv")
train_cats(train)
train.drop("id",axis=1,inplace=True)
df, y, nas,_ = proc_df(train, 'target',do_scale=True)
df.head()
m = RandomForestRegressor(n_estimators=100)

m.fit(df, y)
fi = rf_feat_importance(m,df)
def plot_fi(fi):

    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi[fi.imp>0.005])
model = lgb.LGBMRegressor(n_estimators = 100)

model.fit(df, y)
fi = pd.DataFrame()

fi["feature"] = df.columns

fi["importance"] = model.feature_importances_
top_imp_f = fi[fi.importance>5]

top_imp_f = top_imp_f.sort_values(by="importance", ascending=False)

        

top_imp_f.plot('feature', 'importance', 'barh', figsize=(12,7), legend=False)
eli5.show_weights(model, top=40)