import pandas as pd, numpy as np

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
print("Data Quantity")

print("| # of Train Data : {}".format(len(train_df)))

print("| # of Test Data : {}".format(len(test_df)))
train_df.columns
train_vars = ['bedrooms', 'bathrooms', 'sqft_living',

       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',

       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',

       'lat', 'long', 'sqft_living15', 'sqft_lot15']



cate_vars = ['zipcode', 'waterfront']
train_df.head(5)
print('Train Data View')

train_df.describe()
out_index = np.array([False] * len(train_df))



out_index = out_index | np.array(train_df['sqft_living'] > 13000)

out_index = out_index | np.array((train_df['price']>2555000) & (train_df['grade'] == 8))

out_index = out_index | np.array((train_df['price']>5555000) & (train_df['grade'] == 11))



# Useful?

#train_df = train_df.loc[out_index == False]
for df in [train_df, test_df]:

    df['conv_date'] = [1 if values[:4] == '2014' else 0 for values in df.date ]

    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    df['sqft_ratio'] = df['sqft_living'] / df['sqft_lot']

    df['sqft_total_size'] = df['sqft_living'] + df['sqft_lot'] + df['sqft_above'] + df['sqft_basement']

    df['sqft_total15'] = df['sqft_living15'] + df['sqft_lot15'] 

    

    df['is_renovated'] = df['yr_renovated'] - df['yr_built']

    df['is_renovated'] = df['is_renovated'].apply(lambda x: 0 if x <= 0 else 1)

#     df['date'] = df['date'].astype('int')
def display_var(var_i, df):

    df[df.columns[var_i]].value_counts().plot(kind='bar')

    plt.title(var_i)

    plt.xlabel(df.columns[var_i])

    plt.ylabel('Count')

    plt.show()
display_var(3, train_df)

display_var(4, train_df)

display_var(7, train_df)

display_var(8, train_df)

display_var(9, train_df)

display_var(10, train_df)

display_var(11, train_df)

display_var(21, train_df)
train_df.describe()
train_df.columns
skew_columns = ['sqft_ratio', 'sqft_total_size', 'sqft_total15', 'sqft_above', 'sqft_basement','sqft_living','sqft_lot','sqft_living15', 'sqft_lot15']



minimax_columns = ['lat', 'long', 'total_rooms', 'view', 'condition', 'grade', 'bedrooms', 'bathrooms', 'floors',

                   'sqft_ratio', 'sqft_total_size', 'sqft_total15', 'sqft_above', 'sqft_basement','sqft_living','sqft_lot','sqft_living15', 'sqft_lot15']



etc_vars = ['id', 'date']
for c in skew_columns:

    train_df[c] = np.log1p(train_df[c].values)

    test_df[c] = np.log1p(test_df[c].values)
train_df['price'] = np.log1p(train_df['price'].values)
from sklearn.preprocessing import minmax_scale
concat_df = pd.concat([train_df, test_df], axis=0, sort=False)

for col in minimax_columns:

    col_name = col

    norm_value = minmax_scale(concat_df[col_name])

    train_df[col_name] = norm_value[:len(train_df)]

    test_df[col_name] = norm_value[len(train_df):]
train_df.head(3)
from sklearn.model_selection import train_test_split



def sep_target(df, target_vars):

    return df.drop(target_vars, axis=1), df[target_vars]

def col_trim(df, remove_cols):

    return df.drop(remove_cols, axis=1)





target_vars = ['price']

trn_df, trn_y = sep_target(train_df, target_vars)

trn_df = col_trim(trn_df, etc_vars)

tst_df = col_trim(test_df, etc_vars)



trn_x_full, val_x_full , trn_y_full, val_y_full = train_test_split(trn_df, trn_y, test_size = 0.0, random_state = 9109)

trn_x, val_x , trn_y, val_y = train_test_split(trn_df, trn_y, test_size = 0.2, random_state = 9109)
trn_y_v, val_y_v = map(np.ravel, [trn_y.values, val_y.values])

trn_y_full_v, val_y_full_v = map(np.ravel, [trn_y_full.values, val_y_full.values])
np.shape(np.ravel(trn_y_v))
train_vars = tst_df.columns
from sklearn import ensemble, linear_model

from sklearn.metrics import mean_squared_error



def eval(y_pred, y_true):

    print('error : {}'.format(np.sqrt(mean_squared_error(y_true= np.expm1(y_true), y_pred=np.expm1(y_pred)))))

    

def predeval(x, y_true, clf):

    print('error : {}'.format(np.sqrt(mean_squared_error(y_true= np.expm1(y_true), y_pred=np.expm1(clf.predict(x))))))
import lightgbm as lgb



param = {'objective':'regression', 'metric':'rmse', 'num_iteration':1000, 'learning_rate':0.05, 'early_stopping_round':30,

         'max_depth':-1, 'num_leaves':15, 'feature_fraction':0.6,

         'num_threads':-1}



train_data = lgb.Dataset(trn_x, label=trn_y)

validation_data = lgb.Dataset(val_x, label=val_y, reference=train_data)

bst = lgb.train(param, train_data, valid_sets=[train_data, validation_data])

bst.save_model('model.txt', num_iteration=bst.best_iteration)





eval(y_pred=bst.predict(val_x), y_true=val_y_v)
clf = ensemble.GradientBoostingRegressor(criterion='friedman_mse', n_estimators = 400, max_depth = 5, min_samples_split = 2,

          learning_rate = 0.1, loss = 'ls')

clf.fit(trn_x, trn_y_v)

print("r^2 : ",clf.score(val_x, val_y_v))

predeval(val_x, val_y_v, clf)
# clf = ensemble.RandomForestRegressor(n_estimators=200, max_depth=15)

# clf.fit(trn_x, trn_y_v)

# print("r^2 : ",clf.score(val_x, val_y_v))

# predeval(val_x, val_y_v, clf)



# clf = linear_model.Ridge(alpha=4.0, tol=0.001)

# clf.fit(trn_x, trn_y_v)

# print("r^2 : ",clf.score(val_x, val_y_v))

# predeval(val_x, val_y_v, clf)
pred_lgb = bst.predict(tst_df, num_iteration=bst.best_iteration)
clf = ensemble.GradientBoostingRegressor(criterion='friedman_mse', n_estimators = 400, max_depth = 5, min_samples_split = 2,

          learning_rate = 0.1, loss = 'ls')

clf.fit(trn_x_full, trn_y_full_v)



pred_gbm = clf.predict(tst_df)
pred_avg = np.mean([pred_lgb, pred_gbm], axis=0)
def export(pred):

    subm = pd.read_csv('../input/sample_submission.csv')

    subm['price'] = pred



    subm_num = 0

    subm_name = './subm_{}.csv'.format(str(subm_num).zfill(3))



    while os.path.isfile(subm_name):

        subm_num += 1

        subm_name = './subm_{}.csv'.format(str(subm_num).zfill(3))



    print(subm_name)

    subm.to_csv(subm_name, index=False)
export(np.expm1(pred_lgb))

export(np.expm1(pred_gbm))

export(np.expm1(pred_avg))