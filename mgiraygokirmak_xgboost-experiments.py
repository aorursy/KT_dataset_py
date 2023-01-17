import pandas as pd
from tqdm import tqdm
import numpy as np
import gc

pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
pd.options.display.max_seq_items = 1000
exceptional_days = pd.read_csv("../input/exceptional-days/exceptional_days.txt",parse_dates=['date'])
actions = pd.read_csv('../input/trendyol-project/dailyProductActions.csv',parse_dates=['date'])
products = pd.read_csv('../input/trendyol-project/product.csv')
submission = pd.read_csv('../input/trendyol-project/SampleSubmission.csv')
actions['saleDay'] = 0
actions["week"] = pd.DatetimeIndex(actions["date"]).weekofyear
actions.loc[actions[actions.date.isin(exceptional_days.date.unique())].index,'saleDay'] = 1
actions.loc[actions["week"] == 2, "week"] = 11
actions.loc[actions["week"] == 1, "week"] = 10
actions.loc[actions["week"] == 52, "week"] = 9
actions.loc[actions["week"] == 51, "week"] = 8
actions.loc[actions["week"] == 50, "week"] = 7
actions.loc[actions["week"] == 49, "week"] = 6
actions.loc[actions["week"] == 48, "week"] = 5
actions.loc[actions["week"] == 47, "week"] = 4
actions.loc[actions["week"] == 46, "week"] = 3
actions.loc[actions["week"] == 45, "week"] = 2
actions.loc[actions["week"] == 44, "week"] = 1
actions.fillna(0, inplace=True)
weekly = (actions.groupby(["productid", "week"], as_index=False)
             .agg({'saleDay':'sum', 'stock':'mean', 'clickcount':'sum','favoredcount':'sum', 'soldquantity':'sum', }).sort_values('week'))
product_means = (weekly.groupby(["productid"], as_index=False)
               .agg({'soldquantity':'mean','stock':'mean','clickcount':'mean','favoredcount':'mean'})
               .rename(columns={'soldquantity':'soldquantitymean','stock':'stockmean','clickcount':'clickcountmean','favoredcount':'favoredcountmean'}))
product_means.set_index("productid", inplace=True)
train_set = pd.DataFrame(index=list(products['productid']))
for row in tqdm(weekly.itertuples()):
    train_set.at[row[1], 'prior_week_'        + str(row[2])] = 1
    train_set.at[row[1], 'saleday_week_'      + str(row[2])] = row[3]
    train_set.at[row[1], 'stock_week_'        + str(row[2])] = row[4]
    train_set.at[row[1], 'clickcount_week_'   + str(row[2])] = row[5]
    train_set.at[row[1], 'favoredcount_week_' + str(row[2])] = row[6]
    train_set.at[row[1], 'soldquantity_week_' + str(row[2])] = row[7]
products['kadin'] = 0
products['erkek'] = 0
products.loc[products[(products.gender==2)|(products.gender==3)].index,'kadin'] = 1
products.loc[products[(products.gender==1)|(products.gender==3)].index,'erkek'] = 1
products.drop(columns=['gender'],inplace=True)
train_set = pd.concat([products.set_index('productid'), product_means, train_set], axis=1, join='inner')
train_set.reset_index(inplace=True)
train_set.rename(columns={'index':'productid'},inplace=True)
train_set.fillna(0, inplace=True)
del product_means, weekly, actions, products, exceptional_days
gc.collect()
submission = submission.merge(train_set,how='left',on=['productid'])
submission.drop(columns=['sales','soldquantity_week_11'], inplace=True)
y = train_set['soldquantity_week_11']
colsToTrain = train_set.drop(['soldquantity_week_11'], axis=1)
X = colsToTrain
print('X.shape = ' + str(X.shape))
print('y.shape = ' + str(y.shape))
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(X, y,
                                                      random_state=6,
                                                      test_size=0.15)
X_test = submission.copy()
del colsToTrain, train_set, X, y , submission
gc.collect()
import xgboost as xgb
model = xgb.XGBRegressor(objective = 'reg:linear',
                         metric = 'rmse',
                         n_estimators = 50000,
                         max_depth = 6,
                         learning_rate = 0.001,
                         tree_method = 'gpu_hist',
                         verbosity = 0)
%%time
model.fit(x_train,y_train,
          eval_metric='rmse',
          eval_set=[(x_train, y_train), (x_valid, y_valid)])
del x_train, y_train, x_valid, y_valid
gc.collect()
ax = xgb.plot_importance(model)
fig = ax.figure
fig.set_size_inches(20, 20)
preds = model.predict(X_test)
preds[preds < 0] = 0
subm = pd.DataFrame()
subm['productid'] = X_test.productid.values
subm['sales'] = preds
subm.to_csv('submission.csv', index=False)