import pandas as pd

import warnings

warnings.filterwarnings('ignore')
df_train = pd.read_csv("../input/train.csv", index_col="Id")

df_test = pd.read_csv("../input/test.csv", index_col="Id")

print (df_train.shape)

print (df_test.shape)
df_train.head()
df_train.columns[df_train.isna().sum()>0]
missing = pd.DataFrame(df_train.isna().sum() / df_train.shape[0], columns = ['Percentage']).sort_values(by='Percentage', ascending=False)

missing[missing['Percentage']>0]
columns_50less = missing[missing['Percentage']>=0.5].index



df_train.drop(columns_50less, axis = 1, inplace=True)

df_test.drop(columns_50less, axis = 1, inplace=True)
from sklearn.impute import SimpleImputer



X = df_train.drop(['SalePrice'], axis =1)

y = df_train.SalePrice
numerical_cols = [nname for nname in X.columns

                           if X[nname].dtype in ['int64','float64']]



categorical_cols = [cname for cname in X.columns

                           if X[cname].dtype == 'object']



cols = numerical_cols + categorical_cols



df_train = df_train[cols]

df_test = df_test[cols]
numerical_imputer = SimpleImputer(strategy = 'median')



imputed_train_num = pd.DataFrame(numerical_imputer.fit_transform(X[numerical_cols]))

imputed_test_num = pd.DataFrame(numerical_imputer.transform(df_test[numerical_cols]))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

imputed_train_num=pd.DataFrame(scaler.fit_transform(imputed_train_num))

imputed_test_num=pd.DataFrame(scaler.transform(imputed_test_num))





imputed_train_num.columns = X[numerical_cols].columns

imputed_test_num.columns = df_test[numerical_cols].columns
categorical_imputer = SimpleImputer(strategy = 'most_frequent')



imputed_train_cat = pd.DataFrame(categorical_imputer.fit_transform(X[categorical_cols]))

imputed_test_cat = pd.DataFrame(categorical_imputer.transform(df_test[categorical_cols]))
imputed_train_num.head()
print (imputed_train_num.shape)

print (imputed_train_cat.shape)
imputed_train = pd.concat([imputed_train_num , imputed_train_cat], axis = 1)

imputed_test = pd.concat([imputed_test_num , imputed_test_cat], axis = 1)
imputed_train=pd.get_dummies(imputed_train)

imputed_test=pd.get_dummies(imputed_test)
keep_train = [cname for cname in imputed_train.columns

            if cname in imputed_test.columns]



keep_test = [cname for cname in imputed_test.columns

            if cname in imputed_train.columns]
imputed_train_X = imputed_train[keep_train]

imputed_test_X = imputed_test[keep_test]
imputed_train_X.shape
imputed_test_X.shape
from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
params = {'alpha': [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]}



params_elastic ={'alpha': [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75],

                'l1_ratio': [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]}
ridge_cv = GridSearchCV(Ridge(), params, scoring = 'neg_mean_absolute_error')



lasso_cv = GridSearchCV(Lasso(), params, scoring = 'neg_mean_absolute_error')
scores = -1 * cross_val_score(LinearRegression(),imputed_train_X,y,cv=5, 

                              scoring = "neg_mean_absolute_error")



scores_ridge = -1 * cross_val_score(ridge_cv,imputed_train_X,y,cv=5, 

                              scoring = "neg_mean_absolute_error")



scores_lasso = -1 * cross_val_score(lasso_cv,imputed_train_X,y,cv=5, 

                              scoring = "neg_mean_absolute_error")



scores_rf = -1 * cross_val_score(RandomForestRegressor(),imputed_train_X,y,cv=5, 

                              scoring = "neg_mean_absolute_error")



scores_xgb = -1 * cross_val_score(XGBRegressor(),imputed_train_X,y,cv=5, 

                              scoring = "neg_mean_absolute_error")
imputed_train_X.head()
print ("Linear Regression: %.3f" %scores.mean())

print ("Ridge Regression: %.3f" %scores_ridge.mean())

print ("Lasso Regression: %.3f" %scores_lasso.mean())

print ("RandomForest: %.3f" %scores_rf.mean())

print ("XGBoosting: %.3f" %scores_xgb.mean())
output = pd.DataFrame({'Id':  df_test.index,

                      'SalePrice': XGBRegressor().fit(imputed_train_X, y).predict(imputed_test_X)})



output.to_csv("submission.csv", index=False)