import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
plt.style.use('seaborn')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
sample_submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
sales_train = sales_train.merge(items[['item_id','item_category_id']],on='item_id',how='left')

item_prices = sales_train.groupby(['item_id'],as_index=False).agg({
    'item_price':'mean'
})

test = test.merge(items[['item_id','item_category_id']],on='item_id',how='left')
test = test.merge(item_prices,on='item_id',how='left')
train = sales_train.pivot_table(index=['shop_id','item_category_id','item_id'],columns='date_block_num',values='item_cnt_day',aggfunc='sum',fill_value=0).reset_index()
train = train.applymap(lambda x: x if x >= 0 else 0)
train
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
train_copy = train[['shop_id', 'item_id','item_category_id'] + list(range(34))].copy()
X = train_copy.iloc[:,  (train_copy.columns != 33)].values
y = train_copy.iloc[:,  (train_copy.columns == 33)].values
t = xgb.DMatrix(X,y)
xgr = xgb.train(
    dict(
        objective='count:poisson',
        max_depth=10
    ),
    t
)
preds = xgr.predict(xgb.DMatrix(train_copy.iloc[:,  (train_copy.columns != 33)].values))
rmse = np.sqrt(metrics.mean_squared_error(preds,train_copy.iloc[:, train_copy.columns == 33].values))
rmse
xgb.plot_importance(xgr,height=.5);
metrics.mean_absolute_error(preds,train_copy.iloc[:, train_copy.columns == 33].values)
final_test = test.drop(['item_price'],axis=1).merge(train, how = 'left', on = ['shop_id', 'item_id','item_category_id']).fillna(0.0)
final_test['shop_id'] = final_test['shop_id'].astype(int)
final_test['item_id'] = final_test['item_id'].astype(int)
final_test['item_category_id'] = final_test['item_category_id'].astype(int)

# Move one month forth
d = dict(zip(final_test.columns[4:],list(np.array(list(final_test.columns[4:])) - 1)))
final_test  = final_test.rename(d, axis = 1)
final_test
final_pred = xgr.predict(xgb.DMatrix(final_test.iloc[:, (final_test.columns != 'ID') & (final_test.columns != -1)].values))
sub_df  = final_test.copy()
sub_df['item_cnt_month'] = final_pred
sub_df = sub_df[['ID','item_cnt_month']]
sub_df
sub_df.to_csv('submission.csv',index=False)