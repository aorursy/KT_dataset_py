import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/big-mart-sales-prediction/train_v9rqX0R.csv')
test = pd.read_csv('/kaggle/input/big-mart-sales-prediction/test_AbJTz2l.csv')
train.head()
test.head()
train.shape, test.shape
train.columns, test.columns
train.info(), test.info()
train.isnull().sum() , test.isnull().sum()
train.dtypes, test.dtypes
df = pd.concat([train, test], axis = 0)
df.shape
df.isnull().sum()
df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
df.isnull().sum()
sns.heatmap(df.isnull(), yticklabels = False, cmap = 'viridis', cbar = False)
sns.distplot(train['Item_Outlet_Sales'], kde = False, hist_kws={ "linewidth": 3, "alpha": 1, "color": "g"})
obj_col = df.select_dtypes('object').columns
obj_col
num_col = df.select_dtypes(exclude='object').columns
num_col
fig = plt.figure(figsize = (15, 4))
for i in range(1,4):
    plt.subplot(1, 3, i)
    sns.distplot(df[df[['Item_MRP', 'Item_Visibility', 'Item_Weight']].columns[i-1]], kde = False, color = 'b')

sns.countplot(df['Item_Fat_Content'])
df['Item_Fat_Content'].value_counts()
df['Item_Fat_Content'] = df['Item_Fat_Content'].map({'LF': 'Low Fat', 'low fat' : 'Low Fat', 'reg': 'Regular', 'Low Fat': 'Low Fat', 'Regular': 'Regular'})
df['Item_Fat_Content'].value_counts()
plt.figure(figsize = (15, 4))
sns.countplot(df['Item_Type'])
plt.xticks(rotation = 45)
plt.show();
plt.figure(figsize = (15, 4))

plt.subplot(1,2,1)
sns.countplot(df['Outlet_Identifier'])
plt.xticks(rotation = 45)

plt.subplot(1,2,2)
sns.countplot(df['Outlet_Size'])
plt.xticks(rotation = 45)
plt.figure(figsize = (15, 4))

plt.subplot(1,2,1)
sns.countplot(df['Outlet_Establishment_Year'])
plt.xticks(rotation = 45)

plt.subplot(1,2,2)
sns.countplot(df['Outlet_Type'])
plt.xticks(rotation = 45)
plt.figure(figsize = (15, 5))
sns.scatterplot(x = 'Item_Weight', y = 'Item_Outlet_Sales', data = df)
plt.figure(figsize = (15, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x = 'Item_Visibility', y = 'Item_Outlet_Sales', data = df)

plt.subplot(1,2,2)
sns.scatterplot(x= 'Item_MRP', y = 'Item_Outlet_Sales', data = df)
fig = plt.figure(figsize = (15, 12))

ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)
sns.boxplot(x = 'Item_Type', y = 'Item_Outlet_Sales', data = df, ax=ax1)
plt.xticks(rotation = 45)
ax1.set_title(' Target vs Item_Type')

ax2 = plt.subplot2grid((2,2), (1,0), colspan=1)
sns.boxplot(x = 'Item_Fat_Content', y = 'Item_Outlet_Sales', data = df, ax=ax2)
plt.xticks(rotation = 45)
ax2.set_title(' Target vs Item_Fat_Content')

ax3 = plt.subplot2grid((2,2), (1,1), colspan=1)
sns.boxplot(x = 'Outlet_Identifier', y = 'Item_Outlet_Sales', data = df, ax=ax3)
plt.xticks(rotation = 45)
ax3.set_title(' Target vs Outlet_Identifier')

plt.show();
plt.figure(figsize = (10, 12))

plt.subplot(3, 1, 1)
sns.boxplot( x = 'Outlet_Size', y = 'Item_Outlet_Sales', data = df)

plt.subplot(3, 1, 2)
sns.boxplot( x = 'Outlet_Location_Type', y = 'Item_Outlet_Sales', data = df)

plt.subplot(3, 1, 3)
sns.boxplot( x = 'Outlet_Type', y = 'Item_Outlet_Sales', data = df)

plt.show();
df['Outlet_Size'] = df['Outlet_Size'].fillna('NAN')
df['Outlet_Size'].value_counts()
sns.violinplot( x = 'Outlet_Size', y = 'Item_Outlet_Sales', data = df)
df['Outlet_Size'].replace('NAN', 'Small', inplace = True)
# mean of the Item_Visibility non-zero values. 
Item_vis_mean = df[df['Item_Visibility'] != 0]['Item_Visibility'].mean()
Item_vis_mean
df['Item_Visibility'].replace(0.0, Item_vis_mean, inplace = True)
sns.scatterplot(x = 'Item_Visibility', y = 'Item_Outlet_Sales', data = df)
df['Item_Type'].value_counts()
df['Item_new_type'] = df['Item_Type']
non_perishable = ["Baking Goods", "Canned", "Frozen Foods", "Hard Drinks", "Health and Hygiene", "Household", "Soft Drinks"]
perishable = ["Breads", "Breakfast", "Dairy", "Fruits and Vegetables", "Meat", "Seafood"]

df['Item_new_type'] = df['Item_new_type'].apply(lambda x: 'perishable' if x in perishable else ('non perishable' if x in non_perishable else 'no food'))
df['Item_new_type'].value_counts()
sns.countplot(df['Item_new_type'])
sns.boxplot(x = 'Item_new_type', y = 'Item_Outlet_Sales', data = df)
df['Item_Identity'] =  [x[:2] for x in df['Item_Identifier']]
df['Item_Identity'].value_counts()
df['Outlet_Establishment_Year'].value_counts()
df['Age'] = [2013-x for x in df['Outlet_Establishment_Year']]
df['Age'].hist()
sns.scatterplot(x= 'Age', y= 'Item_Outlet_Sales', data = df)
df['price_per_unit_wt'] = df['Item_MRP']/df['Item_Weight']
sns.lmplot(x= 'price_per_unit_wt', y= 'Item_Outlet_Sales', data = df)
fig, axes = plt.subplots(1, 1, figsize = (10, 8))
sns.scatterplot(x= 'Item_MRP', y= 'Item_Outlet_Sales', hue = 'Item_Fat_Content',
                size = 'Item_Weight', data = df)
plt.plot([69,69], [0,5000])
plt.plot([137,137], [0,6500])
plt.plot([203,203], [0,9500])
df['Item_MRP_bins'] = pd.cut(df['Item_MRP'], bins = [25,69,137,203,270], 
                            labels = ['a', 'b', 'c', 'd'], right = True)
df_final = df[['Item_Fat_Content',  'Item_Outlet_Sales',
        'Item_Visibility', 'Outlet_Identifier',
       'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type', 'Item_new_type',
       'Item_Identity', 'Age', 'price_per_unit_wt', 'Item_MRP_bins']]
df_final.info()
obj_col_final = df_final.select_dtypes('object').columns
obj_col_final
for col in obj_col_final:
    df_final[col] = df_final[col].astype('category')
df_final.info()
cat_col = df_final.select_dtypes('category').columns
cat_col
df_final = pd.get_dummies(data = df_final, columns = cat_col, drop_first=True)
df_final.info()
df_final.shape
train_final = df_final.iloc[:8523, :]
test_final = df_final.iloc[8523:, :]
train_final.shape, test_final.shape
test_final.drop('Item_Outlet_Sales', axis = 1, inplace = True)
test_final.shape
from sklearn.model_selection import train_test_split
X = train_final.copy()
X.drop('Item_Outlet_Sales', axis = 1, inplace = True)
y = train_final['Item_Outlet_Sales']
X.shape , y.shape
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_lr = lr.predict(X_test)
from sklearn.metrics import mean_squared_error

lr_score = np.sqrt(mean_squared_error(y_test, y_lr))
lr_score
from sklearn.model_selection import cross_val_score

score = cross_val_score(lr, X_train, y_train, cv = 10, scoring = 'neg_mean_squared_error')
lr_score_cross = np.sqrt(-score)
np.mean(lr_score_cross), np.std(lr_score_cross)

from sklearn.linear_model import Ridge
r = Ridge(alpha= 0.05, solver = 'cholesky')
r.fit(X_train, y_train)
y_r = r.predict(X_test)
r_score = np.sqrt(mean_squared_error(y_test, y_r))
print(r_score)
score = cross_val_score(r, X_train, y_train, cv = 10, scoring = 'neg_mean_squared_error')
r_score_cross = np.sqrt(-score)
np.mean(r_score_cross), np.std(r_score_cross)
from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.01)
lasso.fit(X_train, y_train)
y_lasso = lasso.predict(X_test)
lasso_score = np.sqrt(mean_squared_error(y_test, y_lasso))
print(lasso_score)

score = cross_val_score(lasso, X_train, y_train, cv = 10, scoring = 'neg_mean_squared_error')
lasso_score_cross = np.sqrt(-score)
np.mean(lasso_score_cross), np.std(lasso_score_cross)
from sklearn.linear_model import ElasticNet

en = ElasticNet(alpha = 0.01, l1_ratio= 0.5)
en.fit(X_train, y_train)
y_en = en.predict(X_test)
en_score = np.sqrt(mean_squared_error(y_test, y_en))
print(en_score)
score = cross_val_score(en, X_train, y_train, cv = 10, scoring='neg_mean_squared_error')
en_score_cross = np.sqrt(-score)
np.mean(en_score_cross), np.std(en_score_cross)
from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor(penalty='l2', max_iter= 100, alpha = 0.05)
sgd.fit(X_train, y_train)
y_sgd = sgd.predict(X_test)
sgd_score = np.sqrt(mean_squared_error(y_test, y_sgd))
print(sgd_score)
score = cross_val_score(sgd, X_train, y_train, cv = 10, scoring='neg_mean_squared_error')
sgd_score_cross = np.sqrt(-score)
np.mean(sgd_score_cross), np.std(sgd_score_cross)
from sklearn.svm import SVR
svr = SVR(epsilon=15, kernel='linear')
svr.fit(X_train, y_train)
y_svr = svr.predict(X_test)
svr_score = np.sqrt(mean_squared_error(y_test, y_svr))
print(svr_score)
score = cross_val_score(svr, X_train, y_train, cv = 10, scoring='neg_mean_squared_error')
svr_score_cross = np.sqrt(-score)
np.mean(svr_score_cross), np.std(svr_score_cross)
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
y_dtr = dtr.predict(X_test)
dtr_score = np.sqrt(mean_squared_error(y_test, y_dtr))
print(dtr_score)
score = cross_val_score(dtr, X_train, y_train, cv= 10, scoring = 'neg_mean_squared_error')
dtr_score_cross = np.sqrt(-score)
np.mean(dtr_score_cross), np.std(dtr_score_cross)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)
rf_score = np.sqrt(mean_squared_error(y_test, y_rf))
print(rf_score)
score = cross_val_score(rf, X_train, y_train, cv = 10, scoring= 'neg_mean_squared_error')
rf_score_cross = np.sqrt(-score)
np.mean(rf_score_cross), np.std(rf_score_cross)
from sklearn.ensemble import BaggingRegressor

br = BaggingRegressor(max_samples = 70)
br.fit(X_train, y_train)
y_br = br.predict(X_test)
br_score = np.sqrt(mean_squared_error(y_test, y_br))
print(br_score)
score = cross_val_score(br, X_train, y_train, cv = 10, scoring='neg_mean_squared_error')
br_score_cross = np.sqrt(-score)
np.mean(br_score_cross), np.std(br_score_cross)
from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor()
ada.fit(X_train, y_train)
y_ada = ada.predict(X_test)
ada_score = np.sqrt(mean_squared_error(y_test, y_ada))
print(ada_score)
score = cross_val_score(ada, X_train, y_train, cv = 10 , scoring = 'neg_mean_squared_error')
ada_score_cross = np.sqrt(-score)
np.mean(ada_score_cross), np.std(ada_score_cross)
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
y_gbr = gbr.predict(X_test)
gbr_score = np.sqrt(mean_squared_error(y_test, y_gbr))
print(gbr_score)
score = cross_val_score(gbr, X_train, y_train, cv =10, scoring='neg_mean_squared_error')
gbr_score_cross = np.sqrt(-score)
np.mean(gbr_score_cross) , np.std(gbr_score_cross)
from xgboost import XGBRegressor

xgb = XGBRegressor()
xgb.fit(X_train, y_train)
y_xgb = xgb.predict(X_test)
xgb_score = np.sqrt(mean_squared_error(y_test, y_xgb))
print(xgb_score)
score = cross_val_score(xgb, X_train, y_train, cv = 10, scoring='neg_mean_squared_error')
xgb_score_cross = np.sqrt(-score)
np.mean(xgb_score_cross), np.std(xgb_score_cross)
name = ['Linear Regression','Linear Regression CV','Ridge Regression','Ridge Regression CV','Lasso Regression',
     'Lasso Regression CV','Elastic Net Regression','Elastic Net Regression CV','SGD Regression','SGD Regression CV',
     'SVM','SVM CV','Decision Tree','Decision Tree Regression','Random Forest','Random Forest CV','Ada Boost','Ada Boost CV',
     'Bagging','Bagging CV','Gradient Boost','Gradient Boost CV', 'XGboost', 'XGBoost CV']
model = pd.DataFrame({'RMSE': [lr_score, lr_score_cross, r_score, r_score_cross, 
                              lasso_score, lasso_score_cross, en_score, en_score_cross, 
                              sgd_score, sgd_score_cross, svr_score, svr_score_cross, 
                              dtr_score, dtr_score_cross, rf_score, rf_score_cross, 
                               ada_score, ada_score_cross, br_score, br_score_cross, 
                              gbr_score, gbr_score_cross, xgb_score, xgb_score_cross]}, index = name)
model['RMSE'] = [np.mean(x) for x in model['RMSE']]
model['RMSE'].sort_values()
from sklearn.model_selection import GridSearchCV
gb = GradientBoostingRegressor(max_depth=7, n_estimators=200, learning_rate=0.01)
param = [{'min_samples_split' : [5,9,13], 
         'max_leaf_nodes' : [3,5,7,9],
         'max_features':[8,10,15,18]}]
gs = GridSearchCV(gb, param, cv = 5, scoring= 'neg_mean_squared_error')
gs.fit(X_train, y_train)
gs.best_estimator_
gb = gs.best_estimator_
train_final.shape
X_train_final = train_final.drop('Item_Outlet_Sales', axis = 1)
y_train_final = train_final['Item_Outlet_Sales']
X_train_final.shape, y_train_final.shape
# fitting model 
gb.fit(X_train_final, y_train_final)
test_final.shape
test_predict = gb.predict(test_final)
test_predict.shape
sample_result = pd.read_csv('/kaggle/input/big-mart-sales-prediction/sample_submission_8RXa3c6.csv')
sample_result.head()
del sample_result['Item_Outlet_Sales']
sample_result['Item_Outlet_Sales'] = test_predict
sample_result
sample_result.to_csv('submission.csv', index = False)
