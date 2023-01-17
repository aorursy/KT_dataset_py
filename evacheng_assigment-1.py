import numpy as np
import pandas as pd
import os
from scipy.stats import norm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.describe()
corrmat = df_train.corr()
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
yticklabels=cols.values, xticklabels=cols.values)
plt.show()

f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()
def show_price_relation(df_train, feature_scatter, feature_box):
    for var in feature_box:
        data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
        f, ax = plt.subplots(figsize=(8, 6))
        fig = sns.boxplot(x=var, y="SalePrice", data=data)
        fig.axis(ymin=0, ymax=800000);
    for var in feature_scatter:
        data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
        data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

show_price_relation(df_train,
                    feature_scatter = ['GrLivArea', 'TotalBsmtSF', 'GarageArea'],
                    feature_box = ['OverallQual', 'GarageCars'])
df_train = df_train[df_train["GarageArea"] < 1200]
df_train = df_train[df_train["TotalBsmtSF"] < 6000]
df_train = df_train[df_train["GrLivArea"] < 4500]

show_price_relation(df_train,
                    feature_scatter = ['GrLivArea', 'TotalBsmtSF', 'GarageArea'],
                    feature_box = ['OverallQual', 'GarageCars'])
sns.distplot(df_train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
df_train['SalePrice']= np.log(df_train['SalePrice'])

sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
df_train['GrLivArea']= np.log(df_train['GrLivArea'])
df_test['GrLivArea']= np.log(df_test['GrLivArea'])

sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
sns.distplot(df_train['TotalBsmtSF'],fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'],plot=plt)
def process_Bsmt(data):
    data['HasBsmt'] = pd.Series(len(data['TotalBsmtSF']), index=data.index)
    data['HasBsmt'] = 0 
    data.loc[data['TotalBsmtSF']>0,'HasBsmt'] = 1
    data.loc[data['HasBsmt']==1,'TotalBsmtSF'] = np.log(data['TotalBsmtSF'])
process_Bsmt(df_train)
process_Bsmt(df_test)
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)

def get_na_summary(data):  
    na_count = data.isnull().sum().sort_values(ascending=False)
    na_rate = (na_count / len(data))
    na_data = pd.concat([na_count, na_rate], axis=1, keys=['count','ratio'])
    return na_data

na_data = get_na_summary(df_train)
na_data.head(20)
drop_na = na_data[na_data['count'] > 1]

#df_train = df_train.drop(drop_na.index, axis=1)
#df_test = df_test.drop(drop_na.index, axis=1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train = df_train.drop(drop_na.index,1)
df_test = df_test.drop(drop_na.index, axis=1)
get_na_summary(df_train)
get_na_summary(df_test)
def fill_na(data, category_value, number_value):  
    quantity = [attr for attr in data.columns if data.dtypes[attr] != 'object']
    quality = [attr for attr in data.columns if data.dtypes[attr] == 'object']
    
    na_data = get_na_summary(data)
    drop_na = na_data[na_data['ratio'] > 0]
    data[drop_na.index] = data[drop_na.index].fillna(number_value)
    for c in quality:
        data[c] = data[c].astype('category')
        if data[c].isnull().any():
            data[c] = data[c].cat.add_categories([category_value])
            data[c] = data[c].fillna(category_value)
    return data

df_train = fill_na(df_train, 'MISSING', 0.)
df_test = fill_na(df_test, 'MISSING', 0.)

print(get_na_summary(df_train))
print(get_na_summary(df_test))
def trans_MSSubClass(data):
    return data.replace({"MSSubClass": {20: "A", 30: "B", 40: "C", 45: "D", 50: "E", 60: "F", 70: "G", 75: "H", 80: "I", 85: "J", 90: "K", 120: "L", 150: "M", 160: "N", 180: "O", 190: "P"}})

df_train = trans_MSSubClass(df_train)
df_test = trans_MSSubClass(df_test)
print(get_na_summary(df_train))
print(get_na_summary(df_test))
df_all = pd.concat([df_train, df_test], keys=['train', 'test'])
df_all = pd.get_dummies(df_all)
df_train = df_all.loc['train']
df_test = df_all.loc['test']
train_x = df_train
train_y = df_train['SalePrice']

train_x = train_x.drop(['SalePrice', 'Id'], 1)
test_x = df_test.drop(['SalePrice', 'Id'], 1)
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
model_RandomForestRegressor = RandomForestRegressor(n_estimators=400)
model_RandomForestRegressor.fit(train_x, train_y)

imp = model_RandomForestRegressor.feature_importances_
imp = pd.DataFrame({'feature': train_x.columns, 'score': imp}).sort_values(by=['score'], ascending=False)

train_x_RandomForestRegressor = train_x.drop(imp[100:]['feature'], 1)
test_x_RandomForestRegressor = test_x.drop(imp[100:]['feature'], 1)

model_RandomForestRegressor.fit(train_x_RandomForestRegressor, train_y)
scores = cross_val_score(model_RandomForestRegressor, train_x, train_y)
scores.mean() 
import xgboost as xgb
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_xgb.fit(train_x, train_y)
imp = model_xgb.feature_importances_
imp = pd.DataFrame({'feature': train_x.columns, 'score': imp}).sort_values(by=['score'], ascending=False)
train_x_model_xgb = train_x.drop(imp[100:]['feature'], 1)
test_x_model_xgb = test_x.drop(imp[100:]['feature'], 1)
model_xgb.fit(train_x_model_xgb, train_y)
scores = cross_val_score(model_xgb, train_x_model_xgb, train_y)
print(scores.mean()) 
from sklearn import datasets, linear_model

regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)
scores = cross_val_score(regr, train_x, train_y)
scores.mean() 
#from sklearn.neural_network import 
from sklearn import metrics
predict_y = model_xgb.predict(test_x_model_xgb)
#metrics.mean_squared_error(train_y, predict_y)
submission = pd.DataFrame()
submission['Id'] = df_test['Id']
submission['SalePrice'] = np.exp(predict_y)
submission.to_csv('submission.csv', index=False)