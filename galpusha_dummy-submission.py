import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.decomposition import IncrementalPCA, PCA
import numpy as np
train_df = pd.read_csv('../input/train.tsv', sep='\t')
train_df = train_df.iloc[:10000, :]
test_df = pd.read_csv('../input/test.tsv', sep='\t')
X = train_df[['item_condition_id','brand_name', 'shipping']]
y = train_df['price']
X_t = test_df[['item_condition_id','brand_name', 'shipping']]
X['brand_name'] = X['brand_name'].fillna('unknown')
X_t['brand_name'] = X_t['brand_name'].fillna('unknown')
brand_dummies = pd.get_dummies(X['brand_name'])
brand_dummies_t = pd.get_dummies(X_t['brand_name'])
brand_dummies['item_condition_id'] = X['item_condition_id']
brand_dummies['shipping'] = X['shipping']
brand_dummies_t['item_condition_id'] = X_t['item_condition_id']
brand_dummies_t['shipping'] = X_t['shipping']
pca = PCA(n_components=100, svd_solver='randomized')
X_train, X_test, y_train, y_test = train_test_split(brand_dummies, y)
brand_dummies_comps_train = pca.fit_transform(X_train)
brand_dummies_comps_test = pca.transform(X_test)
X_train_comps = brand_dummies_comps_train[:,:20]
X_test_comps = brand_dummies_comps_test[:,:20]
regr = RandomForestRegressor(n_estimators=300)
regr.fit(X_train_comps, y_train)
regr.score(X_train_comps, y_train)
regr.score(X_test_comps, y_test)
r2_score(y_true=y_test, y_pred=regr.predict(X_test_comps))
brand_dummies_t_choose_columns = brand_dummies_t[[col for col in X_train.columns if col in brand_dummies_t.columns]]
brand_dummies_t_choose_columns.shape
cols_to_add = [col for col in X_train.columns if col not in brand_dummies_t_choose_columns.columns]
for col in cols_to_add:
    brand_dummies_t_choose_columns[col] = 0
comps_to_predict = pca.transform(brand_dummies_t_choose_columns)
y_pred_result = regr.predict(comps_to_predict[:, :20])
result_df = pd.DataFrame({'price':y_pred_result})
result_df = result_df.reset_index()
result_df.rename(columns={'index': 'test_id'})
result_df.to_csv('submission.csv')



