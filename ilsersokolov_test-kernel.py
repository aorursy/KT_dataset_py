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
df = pd.read_csv('../input/train.tsv', sep='\t')
print(df.shape)
df_test = pd.read_csv('../input/test_stg2.tsv', sep='\t')
print(df_test.shape)
from sklearn.pipeline import make_union, make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

class LabelEncoderPipelineFriendly(LabelEncoder):
    
    def fit(self, X, y=None):
        """this would allow us to fit the model based on the X input."""
        super(LabelEncoderPipelineFriendly, self).fit(X)
        
    def transform(self, X, y=None):
        return super(LabelEncoderPipelineFriendly, self).transform(X).reshape(-1, 1)

    def fit_transform(self, X, y=None):
        return super(LabelEncoderPipelineFriendly, self).fit(X).transform(X).reshape(-1, 1)

def get_y(df):
    return np.log(df["price"]+1)

def get_nums(df):
    return df[["shipping","item_condition_id"]]

def get_cat_name(df):
    return df[["category_name"]]

def get_brand_name(df):
    return df[["brand_name"]]

vec = make_union(*[
    make_pipeline(FunctionTransformer(get_nums, validate=False)),
    make_pipeline(FunctionTransformer(get_cat_name, validate=False),SimpleImputer(strategy="constant",fill_value="unknown"),LabelEncoderPipelineFriendly()),
    make_pipeline(FunctionTransformer(get_brand_name, validate=False),SimpleImputer(strategy="constant",fill_value="unknown"),LabelEncoderPipelineFriendly()),
])
import xgboost as xgb

xgb_reg = xgb.XGBRegressor(reg_lambda=10, n_estimators=150, max_depth=10, learning_rate=1)
y = get_y(df)
X = vec.fit_transform(df)
X_test = vec.fit_transform(df_test)
xgb_reg.fit(X,y)
y_pred = xgb_reg.predict(X_test)
# y_pred = xgb_reg.predict(X)
y_pred = np.exp(y_pred)-1
y_pred[np.where(y_pred<0)]=0
df_y_test = pd.DataFrame(df_test['test_id'].copy())
# df_y_test = df_y_test.rename(columns={'train_id':'test_id'})
df_y_test['price'] = pd.Series(y_pred, index=df_y_test.index)
print(df_y_test.shape)
df_y_test.to_csv("submission_xdb1.csv", index=False)