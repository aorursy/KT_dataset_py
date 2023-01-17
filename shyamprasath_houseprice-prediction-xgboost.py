import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import pandas_profiling

from datetime import date as dt

from sklearn.model_selection import train_test_split

from xgboost import XGBRFRegressor, XGBRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

import lazypredict

from lazypredict.Supervised import LazyRegressor

from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score

from sklearn.feature_selection import RFE, f_oneway, f_regression, SelectKBest

import itertools

from statsmodels.stats.outliers_influence import variance_inflation_factor
!pip install lazypredict
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df.head()
df.shape
df.info()
# report = pandas_profiling.ProfileReport(df)

# report.to_file('report.html')
df_cat = df.select_dtypes(include = 'object')

df_num = df.select_dtypes(exclude = 'object')

print(df_cat.shape, df_num.shape)
# Drop unique value

df_num.drop(['Id'], axis = 1, inplace = True)
(df_num.isnull().sum()/df_num.shape[0]*100)[df_num.isnull().sum()/df_num.shape[0]*100>0]
df_num.drop(['GarageYrBlt'], axis = 1, inplace = True)
def missing(x):

    x = x.fillna(x.median())

    return x
df_num = df_num.apply(missing)
(df_cat.isnull().sum()/df_cat.shape[0]*100)[df_cat.isnull().sum()/df_cat.shape[0]*100>0]
for i in df_cat.columns:

    if i == 'Electrical':

        continue

    df_cat[i] = df_cat[i].fillna('No')
df_cat['Electrical'].value_counts()
df_cat[df_cat['Electrical'].isnull()]
df_cat['Electrical'] = df_cat['Electrical'].fillna('Mix')
df_cat
df_num
df_num['No_years_old'] = dt.today().year - pd.to_datetime(df_num['YearBuilt'], format = '%Y').dt.year
df_num['No_years_after_remodel'] = df_num['YearRemodAdd'] - df_num['YearBuilt']
df_num.drop(['YearBuilt', 'YearRemodAdd', 'YrSold'], axis = 1, inplace = True)
df_num.describe()
df_cat.describe()
def summary(x):

    return pd.Series([x.count(), x.mean(), x.std(), x.min(), x.quantile(0.25), x.quantile(0.50), x.quantile(0.75), 

                      x.quantile(0.90), x.quantile(0.95), x.quantile(0.99), x.max()], 

                     index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', '90%', '95%', '99%', 'max'])
df_num.apply(summary)
# def outlier_capping(x):

#     x = x.clip(upper = x.quantile(0.99))

#     x = x.clip(lower = x.quantile(0.01)) 

#     return x
# df_num = df_num.apply(outlier_capping)
# df_num.apply(summary)
df_cat.columns
lst = []

for i in df_cat.columns:

    v = df_cat[i].value_counts()

    lst.append(v)

    print(v)

    print('-'*50)
df1 = pd.concat([df_num, df_cat], axis = 1)

df1.head()
cv = np.mean(df1)/np.std(df1)

cv
plt.figure(figsize=(20,15))

sns.heatmap(df.corr(), annot = True, fmt = '.2f', cmap = 'coolwarm')

plt.show()
df1 = pd.get_dummies(df1, drop_first = True)
feature = df1[df1.columns.difference(['SalePrice'])]

target = df1.SalePrice
#rfe = RFE(RandomForestRegressor(), n_features_to_select = 15).fit(feature,target)
#list(feature.columns[rfe.get_support()])
skb = SelectKBest(f_oneway, k = 15).fit(feature, target)
list(feature.columns[skb.get_support()])
f_value, p_value = f_regression(feature, target)



f_reg = [(i,v,z) for i, v,z in itertools.zip_longest(feature.columns, f_value, ['%.3f' %p for p in p_value])]
f_reg = pd.DataFrame(f_reg, columns = ['Feature', 'F_value', 'P_value'])

f_reg.sort_values(by = ['P_value'], ascending = True)['Feature'].head(15)
final_list = ['1stFlrSF',

              '2ndFlrSF',

              'BsmtFinSF1',

#              'BsmtUnfSF',

              'GarageArea',

              'GarageCars',

              'LotArea',

#              'LotFrontage',

             'No_years_old',

              'OpenPorchSF',

              'OverallCond',

              'OverallQual',

              'TotalBsmtSF',

              'WoodDeckSF',

#             'Condition2_RRAn',

#              'ExterCond_Po',

#              'Exterior1st_AsphShn',

#              'Exterior1st_ImStucc',

#              'Exterior2nd_Other',

#              'Functional_Sev',

#              'HeatingQC_Po',

#              'MiscFeature_TenC',

#              'PoolArea',

#              'RoofMatl_Membran',

#              'RoofMatl_Metal',

#              'RoofMatl_Roll',

#              'Utilities_NoSeWa',

#             'GarageQual_TA',

             'GarageType_Attchd',

#             'GarageType_BuiltIn',

             'GarageType_Detchd',

#             'GarageType_No',

             'GrLivArea',

#             'HalfBath',

#             'GarageQual_Fa',

#             'HeatingQC_Fa',

#             'HeatingQC_TA',

#             'Heating_GasA',

#             'Heating_Grav',

#             'HouseStyle_2Story'

            ]



x = feature[final_list]
vif = pd.DataFrame()

vif['vif_factor'] = [variance_inflation_factor(x.values,i) for i in range(x.shape[1])]

vif['features'] = x.columns

vif.sort_values(by = ['vif_factor'], ascending = False)
model = ExtraTreesRegressor().fit(x, target)
model.feature_importances_
pd.Series(model.feature_importances_, index = x.columns).sort_values(ascending = False).head(20).plot(kind = 'barh')

plt.show()
x_train, x_test, y_train, y_test = train_test_split(x, target, random_state = 20)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
reg = LazyRegressor(verbose=0,ignore_warnings=False)

models,predictions = reg.fit(x_train, x_test, y_train, y_test)
models
predictions
xgb = XGBRegressor(learning_rate=0.01, n_estimators=5500,

                       max_depth=3, min_child_weight=0,

                       gamma=0, subsample=0.7,

                       colsample_bytree=0.7,

                       objective='reg:squarederror', nthread=-1,

                       scale_pos_weight=1,

                       reg_alpha=0.00006)



xgb.fit(x_train, y_train)
result_train = pd.DataFrame()

result_test = pd.DataFrame()



result_train['Predicted'] = xgb.predict(x_train)

result_train['Actual'] = y_train



result_test['Predicted'] = xgb.predict(x_test)

result_test['Actual'] = y_test
print('Train Accuracy', r2_score(y_train, result_train.Predicted))

print('Test Accuracy', r2_score(y_test, result_test.Predicted))
print('Train Error', np.sqrt(mean_squared_log_error(y_train,  result_train.Predicted)))

print('Test Error', np.sqrt(mean_squared_log_error(y_test,  result_test.Predicted)))
train_error = result_train['Actual'] - result_train['Predicted']
sns.distplot(train_error)
test_error = result_test['Actual'] - result_test['Predicted']
sns.distplot(test_error)
result_train['Decile'] = pd.qcut(result_train['Predicted'], 10, labels=False)

result_train.groupby('Decile').mean()
result_test['Decile'] = pd.qcut(result_test['Predicted'], 10, labels=False)

result_test.groupby('Decile').mean()
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

df.head()
df.shape
df_cat = df.select_dtypes(include = 'object')

df_num = df.select_dtypes(exclude = 'object')



df_num.drop(['Id'], axis = 1, inplace = True)



df_num.drop(['GarageYrBlt'], axis = 1, inplace = True)



def missing(x):

    x = x.fillna(x.median())

    return x



df_num = df_num.apply(missing)



for i in df_cat.columns:

    if i == 'Electrical':

        continue

    df_cat[i] = df_cat[i].fillna('No')

    

df_cat['Electrical'] = df_cat['Electrical'].fillna('Mix')



df_num['No_years_old'] = dt.today().year - pd.to_datetime(df_num['YearBuilt'], format = '%Y').dt.year



df_num['No_years_after_remodel'] = df_num['YearRemodAdd'] - df_num['YearBuilt']



# Drop unwanted columns



df_num.drop(['YearBuilt', 'YearRemodAdd', 'YrSold'], axis = 1, inplace = True)







df1 = pd.concat([df_num, df_cat], axis = 1)



df1 = pd.get_dummies(df1, drop_first = True)
df1 = df1[final_list]
list(xgb.predict(df1))
submission = pd.DataFrame({'Id' : df['Id'], 'SalePrice': xgb.predict(df1)})
submission
submission.to_csv('submission.csv', index=False)