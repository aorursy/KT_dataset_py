import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
data_raw = pd.read_csv('../input/train.csv')
data_raw = data_raw.drop('item_id', axis=1).drop('user_id', axis=1).drop('title', axis=1).drop('description', axis=1)\
        .drop('activation_date', axis=1).drop('user_type', axis=1).drop('image', axis=1).drop('image_top_1', axis=1)
region_set = data_raw['region'].unique()
def convert_region(x):
    tp = np.where(region_set == x['region'])
    return tp[0][0]
data_raw['region'] = data_raw.apply(convert_region, axis=1)
city_set = data_raw['city'].unique()
def convert_city(x):
    tp = np.where(city_set == x['city'])
    return tp[0][0]
data_raw['city'] = data_raw.apply(convert_city, axis=1)
cat_set = data_raw['category_name'].unique()
def convert_cat(x):
    tp = np.where(cat_set == x['category_name'])
    return tp[0][0]
data_raw['category_name'] = data_raw.apply(convert_cat, axis=1)
p1_set = data_raw['param_1'].unique()
def convert_p1(x):
    tp = np.where(p1_set == x['param_1'])
    try:
        return tp[0][0]
    except Exception:
        return 0
data_raw['param_1'] = data_raw.apply(convert_p1, axis=1)
p2_set = data_raw['param_2'].unique()
def convert_p2(x):
    tp = np.where(p2_set == x['param_2'])
    try:
        return tp[0][0]
    except Exception:
        return 0
data_raw['param_2'] = data_raw.apply(convert_p2, axis=1)
p3_set = data_raw['param_3'].unique()
def convert_p3(x):
    tp = np.where(p3_set == x['param_3'])
    try:
        return tp[0][0]
    except Exception:
        return 0
data_raw['param_3'] = data_raw.apply(convert_p3, axis=1)
parent_category_name_set = data_raw['parent_category_name'].unique()
def convert_parent_category_name(x):
    tp = np.where(parent_category_name_set == x['parent_category_name'])
    return tp[0][0]
data_raw['parent_category_name'] = data_raw.apply(convert_parent_category_name, axis=1)
x_train = data_raw.drop('deal_probability', axis=1).as_matrix()
y_train = data_raw['deal_probability'].as_matrix()
import xgboost as xgb
xgdmat=xgb.DMatrix(x_train, y_train)
params={'seed':0,'colsample_bytree':0.8,'objective':'reg:linear','max_depth':24,'min_child_weight':24}
final_gb=xgb.train(params,xgdmat)
test_df = pd.read_csv('../input/test.csv')
test_df = test_df.drop('user_id', axis=1).drop('title', axis=1).drop('description', axis=1)\
        .drop('activation_date', axis=1).drop('user_type', axis=1).drop('image', axis=1).drop('image_top_1', axis=1)
def convert_region(x):
    tp = np.where(region_set == x['region'])
    return tp[0][0]
# creating region catagory as numerical feature
test_df['region'] = test_df.apply(convert_region, axis=1)
def convert_city(x):
    tp = np.where(city_set == x['city'])
    try:
        return tp[0][0]
    except Exception:
        return 0
test_df['city'] = test_df.apply(convert_city, axis=1)
def convert_cat(x):
    tp = np.where(cat_set == x['category_name'])
    try:
        return tp[0][0]
    except Exception:
        return 0
test_df['category_name'] = test_df.apply(convert_cat, axis=1)
def convert_p1(x):
    tp = np.where(p1_set == x['param_1'])
    try:
        return tp[0][0]
    except Exception:
        return 0
test_df['param_1'] = test_df.apply(convert_p1, axis=1)
def convert_p2(x):
    tp = np.where(p2_set == x['param_2'])
    try:
        return tp[0][0]
    except Exception:
        return 0
test_df['param_2'] = test_df.apply(convert_p2, axis=1)
def convert_p3(x):
    tp = np.where(p3_set == x['param_3'])
    try:
        return tp[0][0]
    except Exception:
        return 0
test_df['param_3'] = test_df.apply(convert_p3, axis=1)
def convert_parent_category_name(x):
    tp = np.where(parent_category_name_set == x['parent_category_name'])
    try:
        return tp[0][0]
    except Exception:
        return 0
test_df['parent_category_name'] = test_df.apply(convert_parent_category_name, axis=1)
x_test = test_df.drop('item_id', axis=1).as_matrix()
tesdmat=xgb.DMatrix(x_test)
y_pred=final_gb.predict(tesdmat)
y_pred[y_pred<0] = 0
result = pd.DataFrame({ 'deal_probability': y_pred, 'item_id': test_df['item_id']})
result.to_csv('submission.csv', index=False)
