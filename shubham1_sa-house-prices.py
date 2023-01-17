# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
import datetime
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv', keep_default_na= False) # , na_values = 'NaN'

    
print("Data shape::", df.shape)
print("Test Data shape::", df_test.shape)
print('number:', df.select_dtypes(include='number').columns)
print('bool:', df.select_dtypes(include='bool').columns)
print('category:', df.select_dtypes(include='category').columns)
print('dtype:', df.select_dtypes(include='dtype').columns)
# Dropping columns with significant null values
null_cols = df.columns[df.isnull().sum()>int(np.sqrt(len(df)))]
print('Dropping columns:\n', null_cols)
df = df.drop(list(null_cols), axis=1)
df_test = df_test.drop(list(null_cols), axis=1)
print("Data shape::", df.shape)
print("Test Data shape::", df_test.shape)
df = df.dropna(axis=0, how='any')
print("Data shape::", df.shape)
print("Test Data shape::", df_test.shape)
d = defaultdict(LabelEncoder)

df_cat = df[df.select_dtypes(include='dtype').columns]
for col in df.columns:
    if col in df_cat:
        le = LabelEncoder()
        lt = df[col].tolist()
        lt.append("Unknown")
        le.fit(lt)
        
        df[col] = le.transform(df[col])
        #df_test[col] = df_test[col].fillna(value = df_test[col].mode()[0])
        lt2 = df_test[col].tolist()
        new_lt = []
        for item in lt2:
            if item == 'NA' or item not in le.classes_:
                new_lt.append("Unknown")
            else:
                new_lt.append(item)
        df_test[col] = le.transform(new_lt)
        d[col] = le
# df_fit = df_cat.apply(lambda x: d[x.name].fit_transform(x))
print("Data shape::", df.shape)
print("Test Data shape::", df_test.shape)

df_num = df.select_dtypes(include='number').columns
for col in df.columns:
    if col in df_num and col != 'SalePrice':
        lt2 = df_test[col].tolist()
        new_lt = []
        for item in lt2:
            if item == 'NA':
                new_lt.append(np.mean(df[col]))
            else:
                new_lt.append(item)
        df_test[col] = new_lt
        
for idx,row in df_test.iterrows():
    for item in row:
        if item == 'NA':
            print(row)
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
x_cols = list(df.columns)
x_cols.remove('SalePrice')
X, y = df[x_cols], df['SalePrice']
print("Shape: X,y::",X.shape, y.shape)
# Decision Tree Regressor
reg_dt = DecisionTreeRegressor()
reg_dt.fit(X,y)

scores = cross_val_score(reg_dt, X,y, cv = 10,scoring = 'neg_mean_squared_error')
print("Scores:", np.round(scores,4))
print("Scores Mean:", scores.mean())
print("Scores std:", scores.std())
print("RMSE:", np.sqrt(-scores).mean())
# Random Forest Regressor
reg_rf = RandomForestRegressor(n_estimators=35)
reg_rf.fit(X,y)

scores = cross_val_score(reg_rf, X,y, cv = 10, scoring = 'neg_mean_squared_error')
print("Scores:", np.round(scores,4))
print("Scores Mean:", scores.mean())
print("Scores std:", scores.std())
print("RMSE:", np.sqrt(-scores).mean())
# Random Forest Regressor
reg_et = ExtraTreesRegressor()
reg_rf.fit(X,y)

scores = cross_val_score(reg_rf, X,y, cv = 10, scoring = 'neg_mean_squared_error')
print("Scores:", np.round(scores,4))
print("Scores Mean:", scores.mean())
print("Scores std:", scores.std())
print("RMSE:", np.sqrt(-scores).mean())
# feature importances
imp_ = []
for feature, importance in zip(X.columns, reg_rf.feature_importances_):
    imp_.append((feature, importance))
imp_df = pd.DataFrame(imp_, columns=['feature', 'imp'])
imp_df = imp_df.sort_values(by = ['imp'], ascending=False)
# print(imp_df)
imp_feat = list(imp_df[:-10].feature)
print('important features:\n',imp_feat)
reg_rf_imp = RandomForestRegressor(n_estimators=35)
# X_imp = X[imp_feat]
X_imp = X
reg_rf_imp.fit(X_imp,y)

scores = cross_val_score(reg_rf_imp, X_imp,y, cv = 10, scoring = 'neg_mean_squared_error')
print("Scores:", np.round(scores,4))
print("Scores Mean:", scores.mean())
print("Scores std:", scores.std())
print("RMSE:", np.sqrt(-scores).mean())
rmse_score = np.sqrt(-scores).mean()
# Test
# df_test = df_test.drop(to_delete,axis=1)
now = datetime.datetime.now()
test_pred = reg_rf_imp.predict(df_test)
res = pd.DataFrame({'Id': df_test['Id'].values, 'SalePrice': test_pred})
res['Id'] = res['Id'].astype(int)
sub_file = 'submission_'+str(rmse_score)+'_'+str(now.strftime("%Y-%m-%d-%H-%M"))+'.csv'
res.to_csv(sub_file, index=False)


