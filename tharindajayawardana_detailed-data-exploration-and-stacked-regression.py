import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df.set_index('Id', inplace=True)
df.head(5)
df.describe()
df.info()
df.hist(bins=20, figsize=(20,15))
plt.show()
df_empty = df.columns[df.isnull().any()]
df_empty = df[df_empty]
df_empty.info()
df_empty.describe()
from pandas.plotting import scatter_matrix
df_empty = df_empty.join(df['SalePrice'])
scatter_matrix(df_empty, figsize=(12,8), diagonal='hist')
plt.show()
#missing values strategy
#na means absence of option in house - custom transformer
na_cat = ['Alley', 'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu',
                'GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']

#for numerical empty values strategy=Median
median =['LotFrontage']

#for numerical empty values strategy=Constant
#GaragYrBlt, use oldest year = 1900 to enable adding feature - age of garage - no garage means zero age
constant = ['GarageYrBlt'] 

#for numerical and catetgorical variables stratege =most frequent
frequent_cat = ['MasVnrType','Electrical']
frequent_num = ['MasVnrArea']

#check all missing features are addressed
empty_cat_columns = na_cat + frequent_cat
all_missing_feat = empty_cat_columns + median + constant + frequent_num
len(all_missing_feat) == len(df_empty.columns) -1
#fill in empty data to facilitate data evaluation
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

df_fillempty = ColumnTransformer([('median', SimpleImputer(strategy='median'), median),
                                        ('constant', SimpleImputer(strategy='constant', fill_value=1900), constant),
                                        ('frequent', SimpleImputer(strategy='most_frequent'), frequent_num),
                                        ('na', SimpleImputer(strategy='constant', fill_value='na'),na_cat),
                                        ('cat_frequent', SimpleImputer(strategy='most_frequent'), frequent_cat)], 
                                       remainder='passthrough')
df_fill = df_fillempty.fit_transform(df)
fill_columns = ['LotFrontage','GarageYrBlt','MasVnrArea','Alley', 'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu',
                'GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature','MasVnrType','Electrical']
column_list = list(df.columns)
passthrough_columns = [i for i in column_list if i not in fill_columns]
df_fill_columns = fill_columns + passthrough_columns
df_fill = pd.DataFrame(df_fill, columns=df_fill_columns)
df_fill.index = df.index
df_fill = df_fill.apply(pd.to_numeric, errors='ignore')
df_fill.info() 
len(df.select_dtypes(include='number').columns)
import seaborn as sns
plt.figure(figsize=(20,20))
corr_matrix=df_fill.corr().abs()
sns.heatmap(corr_matrix, cmap='YlGnBu' , annot=True)
#observations

#New features to test out
#SF related features> combine to form 'total_sf' and remove the consitutents - 'TotalBsmtSF','GrLivArea', other SF related features make up these two vairables hence remove
#combine bathrooms = from one feature and check for relationship - 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'
#check whether breaking out other rooms from TotRmsAbvGrd will improve 'TotRmsAbvGrd' , 'BedroomAbvGr'
#combine out door areas to form outdoorSF by combining 'OpenPorchSF','EnclosedPorch', '3SsnPorch', 'ScreenPorch','WoodDeckSF'

#garagecars and garagearea same thing, dropping garagearea and garagecars tad bid more related to salesprice

#yearremodeled and yearbuilt are same data, removed yearbuild and remove garageyearbuilt as related to prior, not meaningfully related to sale price

#garageyrbuilt drop due to correlation with yearremodelled

#MSSubClass is an object

#Other variable based on description and muilticolinearity will drop, will focus only on num variables with strong correlation as well as fundemental relationship with saleprice

num_features_drop = ['1stFlrSF','2ndFlrSF','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','GarageCars','GarageYrBlt', 'YearBuilt','OverallCond',
                     'LowQualFinSF','PoolArea','MiscVal','KitchenAbvGr']
                
num_to_object =['MSSubClass','MoSold','YrSold']
#experiment with new features
df_cleannum = df_fill.drop(num_features_drop, axis=1)
df_cleannum['TotalSF'] = df_cleannum[['TotalBsmtSF','GrLivArea']].sum(axis=1)
df_cleannum['TotalBathrooms'] = df_cleannum[['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']].sum(axis=1)
df_cleannum['OtherRooms'] = df_cleannum['TotRmsAbvGrd'] - df_cleannum['BedroomAbvGr']
df_cleannum['OutdoorSF'] = df_cleannum[['OpenPorchSF','EnclosedPorch', '3SsnPorch', 'ScreenPorch','WoodDeckSF']].sum(axis=1)
df_cleannum['AvgRoomSF'] = df_cleannum['GrLivArea'] / df_cleannum['TotRmsAbvGrd']
corr_matrix = df_cleannum.corr().abs()
plt.figure(figsize=(20,20))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu')
#run scatter plot with a few features highly correlated with SalePrice
from pandas.plotting import scatter_matrix
df_scatter = df_cleannum[['SalePrice','GrLivArea','OverallQual','MasVnrArea','Fireplaces', 'TotalSF']]
scatter_matrix(df_scatter,figsize=(15,12))
plt.show()
#two data points seem to be clear outliers when looking SalePrice vs both GrLivArea and OverallQaul
#filter dataset based on GRLivArea > 4000

df_cleannum[df_cleannum.SalePrice>700000]
sns.boxplot(x='SalePrice', data=df_cleannum )
df_cleannum[df_cleannum.TotalSF>7000]
df_cleannum[df_cleannum['SalePrice']>700000]
sns.boxplot(x='TotalSF', data=df_cleannum)
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=df_cleannum)
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df_cleannum)
df_cleannum[df_cleannum['GrLivArea']>4000]
data_issues = df_cleannum[df_cleannum['TotalBsmtSF'] > df_cleannum['GrLivArea']]
data_issues[data_issues['TotalBsmtSF'] / data_issues['GrLivArea'] > 1.05][['TotalBsmtSF', 'GrLivArea']]
#keeping SalePrice outliers as seems justified based on features, rerun after removing to see whether there will be an improvement
#data items to drop
outliers= [524, 1299, 692, 1183, 127, 154, 333, 441, 1265]
df_cleannum.drop(outliers, inplace=True)
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df_cleannum)
df_cleannum.skew(axis=0)
df_cleannum[num_to_object] = df_cleannum[num_to_object].astype(object)
cat_column_list = df_cleannum.select_dtypes('object').columns
cat_column_list
#determinie categorical encoding strategy 

#dropiing the following;

cat_features_drop = ['BsmtExposure','GarageFinish',
         'PoolQC','Fence','MiscFeature','Exterior1st','Exterior2nd','GarageCond']

#ordinal encoding
ordinal = ['Neighborhood','BldgType','HouseStyle','ExterCond', 'ExterQual', 'BsmtQual','HeatingQC','CentralAir','FireplaceQu','PavedDrive','GarageQual','BsmtFinType1','BsmtFinType2']

#onehot encoding
one_hot = ['Foundation','BsmtCond','Heating','Electrical', 'RoofMatl','Functional','SaleType','MSZoning','Utilities','KitchenQual','MasVnrType',
           'Street', 'Alley','LandSlope', 'LandContour','Condition1', 'Condition2','LotShape', 'LotConfig','MoSold','YrSold','MSSubClass','SaleCondition','GarageType','RoofStyle']

#categoris to combine & onehot encode
Combine_cat = ['Street', 'Alley','LandSlope', 'LandContour','Condition1', 'Condition2','LotShape', 'LotConfig']
Access = ['Street', 'Alley']
LandChar = ['LandSlope', 'LandContour']
Proximitty = ['Condition1', 'Condition2']
LandProfile = ['LotShape', 'LotConfig']


len(cat_column_list) - len(ordinal + one_hot + cat_features_drop)
sns.catplot(x='Fence', y='SalePrice', data=df_cleannum, height=5, aspect=5)
df_cleannum['Fence'].value_counts()
df_test = df_cleannum.copy()
df_test['Access'] = df_test[['Street', 'Alley']].agg('_'.join, axis=1)
df_test['LandChar'] = df_test[['LandSlope', 'LandContour']].agg('_'.join, axis=1)
df_test['Proximitty'] = df_test[['Condition1', 'Condition2']].agg('_'.join, axis=1)
df_test['LandProfile'] = df_test[['LotShape', 'LotConfig']].agg('_'.join, axis=1)
df_test['BsmtQual'] = df_test[['BsmtFinType1', 'BsmtFinType2']].agg('_'.join, axis=1)
sns.catplot(x='Access', y='SalePrice', data=df_test, height=5, aspect=3)
df_test['Access'].value_counts()
sns.catplot(x='LandChar', y='SalePrice', data=df_test, height=5, aspect=3)
df_test['LandChar'].value_counts()
sns.catplot(x='Proximitty', y='SalePrice', data=df_test, height=5, aspect=5)
df_test['Proximitty'].value_counts()
sns.catplot(x='LandProfile', y='SalePrice', data=df_test, height=5, aspect=5)
df_test['LandProfile'].value_counts()
sns.catplot(x='BsmtQual', y='SalePrice', data=df_test, height=5, aspect=5)
df_test['BsmtQual'].value_counts()
df.drop(outliers, inplace=True)
X = df.drop('SalePrice', axis=1)
y = df['SalePrice'].copy()
X.shape
X[num_to_object] = X[num_to_object].astype('object')
#get num and categorical columns lists
num_feature_list = list(X.select_dtypes('number').columns)
cat_feature_list = list(set(X.columns)-set(num_feature_list))
#remove the columns to drop
num_feature_list = list(set(num_feature_list) - set(num_features_drop))
cat_feature_list = list(set(cat_feature_list) - set(cat_features_drop))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
ordinal_na = [cat for cat in na_cat if cat in ordinal]
ordinal_balance = list(set(ordinal) - set(ordinal_na))
onehot_na = [cat for cat in na_cat if cat in one_hot and Combine_cat]
onehot_frequent = [cat for cat in frequent_cat if cat in one_hot and Combine_cat]
onehot_balance = list(set(one_hot) - set(onehot_na + onehot_frequent))
one_hot_all = onehot_na + onehot_frequent + onehot_balance
len(cat_feature_list) == len(ordinal_na + ordinal_balance + onehot_na + onehot_frequent + onehot_balance)
num_frequant = [num for num in frequent_num if num in num_feature_list]
num_constant = [num for num in constant if num in num_feature_list]
num_median = list(set(num_feature_list) - set(num_frequant + num_constant))

#setup pipeline with processing as per this sequence of features
num_feature_list = num_frequant + num_constant + num_median
#custom feature selector
from sklearn.base import BaseEstimator, TransformerMixin

class CustomFeatureSelector(BaseEstimator, TransformerMixin):
  def __init__(self, feature_names):
    self._feature_names=feature_names

  def fit(self, X, y=None):
    return self
  
  def transform(self, X, y=None):
    return X[self._feature_names]
#setup custom feature addition - num features

TotalBsmtSF_ix= X_train[num_feature_list].columns.get_loc('TotalBsmtSF')
GrLivArea_ix= X_train[num_feature_list].columns.get_loc('GrLivArea')
BsmtFullBath_ix= X_train[num_feature_list].columns.get_loc('BsmtFullBath')
BsmtHalfBath_ix= X_train[num_feature_list].columns.get_loc('BsmtHalfBath')
Fullbath_ix= X_train[num_feature_list].columns.get_loc('FullBath')
Halfbath_ix= X_train[num_feature_list].columns.get_loc('HalfBath')
BedroomAbvGr_ix = X_train[num_feature_list].columns.get_loc('BedroomAbvGr')
TotRmsAbvGrd_ix = X_train[num_feature_list].columns.get_loc('TotRmsAbvGrd')
OpenPorchSF_ix = X_train[num_feature_list].columns.get_loc('OpenPorchSF')
EnclosedPorch_ix = X_train[num_feature_list].columns.get_loc('EnclosedPorch')
SsnPorch_ix = X_train[num_feature_list].columns.get_loc('3SsnPorch')
ScreenPorch_ix = X_train[num_feature_list].columns.get_loc('ScreenPorch')
WoodDeckSF_ix = X_train[num_feature_list].columns.get_loc('WoodDeckSF')


class CustomFeatureAdditionNum(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    totalSF = X[:, TotalBsmtSF_ix] + X[:,GrLivArea_ix]
    totalbaths = X[:, BsmtFullBath_ix] + X[:, BsmtHalfBath_ix] + X[:, Fullbath_ix] + X[:,Halfbath_ix]
    otherrooms = X[:, TotRmsAbvGrd_ix] - X[:, BedroomAbvGr_ix]
    outdoorSF = X[: , OpenPorchSF_ix] + X[: , EnclosedPorch_ix] + X[: , SsnPorch_ix] + X[: , ScreenPorch_ix] + X[: , WoodDeckSF_ix]
    avgroomSF = X[: ,GrLivArea_ix] / X[:, TotRmsAbvGrd_ix]
    X=np.delete(X, [TotalBsmtSF_ix, GrLivArea_ix, BsmtFullBath_ix, BsmtHalfBath_ix, Fullbath_ix, Halfbath_ix, TotRmsAbvGrd_ix, OpenPorchSF_ix,
                    EnclosedPorch_ix, SsnPorch_ix, ScreenPorch_ix, WoodDeckSF_ix] , axis=1)
    return np.c_[X, totalSF, totalbaths, otherrooms, outdoorSF, avgroomSF]

#correcting skewness

class LogTransform(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    X_log = np.log1p(X)
    return X_log
#setup custom feature addition - cat features
Access = ['Street', 'Alley']
LandChar = ['LandSlope', 'LandContour']
Proximitty = ['Condition1', 'Condition2']
LandProfile = ['LotShape', 'LotConfig']

Street_ix = X_train[one_hot_all].columns.get_loc('Street')
Alley_ix = X_train[one_hot_all].columns.get_loc('Alley')
LandSlope_ix = X_train[one_hot_all].columns.get_loc('LandSlope')
LandContour_ix = X_train[one_hot_all].columns.get_loc('LandContour')
Condition1_ix = X_train[one_hot_all].columns.get_loc('Condition1')
Condition2_ix = X_train[one_hot_all].columns.get_loc('Condition2')
LotShape_ix = X_train[one_hot_all].columns.get_loc('LotShape')
LotConfig_ix = X_train[one_hot_all].columns.get_loc('LotConfig')



class CustomFeatureAdditionCat(BaseEstimator,TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    street = X[:, Street_ix].astype('str')
    alley = X[:, Alley_ix].astype('str')
    slope = X[:, LandSlope_ix].astype('str')
    contour = X[:, LandContour_ix].astype('str')
    condition1 = X[:, Condition1_ix].astype('str')
    condition2 = X[:, Condition2_ix].astype('str')
    lotshape = X[:, LotShape_ix].astype('str')
    lotconfig = X[:, LotConfig_ix].astype('str')
    LandChar = np.char.add(slope, contour)
    Access = np.char.add(street, alley)
    Proximity = np.char.add(condition1, condition2)
    LandProfile = np.char.add(lotshape, lotconfig)
    X = np.delete(X, [Street_ix, Alley_ix, LandSlope_ix, LandContour_ix, Condition1_ix, Condition2_ix, LotShape_ix, LotConfig_ix], axis=1)
    return np.c_[X, LandChar, Access, Proximity, LandProfile]
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
#onehot_columntransformer
onehot_transformer = ColumnTransformer([('constant_onehot', SimpleImputer(strategy ='constant', fill_value='na'),onehot_na),
                                        ('frequent_onehot', SimpleImputer(strategy='most_frequent'), onehot_frequent),
                                        ('balance_onehot', SimpleImputer(strategy='most_frequent'), onehot_balance)],
                                       remainder='passthrough')
                                     
#cat pipelines
ordinal_pipeline = Pipeline([('catordinal_column_selection', CustomFeatureSelector(ordinal)),
                             ('fill_NanOrdConstant', SimpleImputer(strategy='constant', fill_value='na')),
                             ('ordinal_encode', OrdinalEncoder()),
                             ('std_cat', StandardScaler())])

onehot_pipeline = Pipeline([('catonehot_column_selection', CustomFeatureSelector(one_hot_all)),
                            ('onehot_transformer', onehot_transformer),
                            ('catcustom_feature', CustomFeatureAdditionCat()),
                            ('onehot_encoder', OneHotEncoder(handle_unknown='ignore'))])

#num transformer
num_transformer = ColumnTransformer([('frequent_impute', SimpleImputer(strategy='most_frequent'), num_frequant),
                            ('constant_num', SimpleImputer(strategy='constant', fill_value=1900), num_constant),
                            ('median_num', SimpleImputer(strategy='median'), num_median)])

#num pipeline
num_pipeline = Pipeline([('num_column_selection', CustomFeatureSelector(num_feature_list)),
                        ('num_tranformer', num_transformer),
                        ('custom_features', CustomFeatureAdditionNum()),
                        ('log_transform', LogTransform()),
                        ('std_num', StandardScaler())])

#combining pipeline
pre_process = FeatureUnion([('ordinal_pipeline', ordinal_pipeline),
                            ('onehot_pipeline', onehot_pipeline),
                            ('num_pipeline', num_pipeline)])
#fitting to full training - missing categorical vairable, can't initiate 'ignore' as dropping first columsn in onehot encoding to avoid multicolenearity
X_prepared_train = pre_process.fit_transform(X_train)
X_prepared_test = pre_process.transform(X_test)

# Checking to ensure shapes of processed train and test set is equal due to uneven spread of cat data between test and training set
print(X_prepared_train.shape[1] == X_prepared_test.shape[1])

#log_transform y(SalePrice)
y_train_log = np.log1p(y_train)
X_prepared_train.shape
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error
np.set_printoptions(precision=3)
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
def rmse_model(y_train, y_pred):
  mse = mean_squared_error(y_train, y_pred)
  rmse = np.sqrt(mse)
  return rmse
def crossvalscores(model, X, y, cv):
  scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error',cv=cv)
  # scores = np.expm1(scores)
  scores = np.sqrt(-scores)
  print('Scores : ', scores)
  print('Mean : ', scores.mean())
  print('StdD : ', scores.std())
def plot_learning_curves(model, X, y):
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
  train_errors, val_errors = [], []
  for m in range(1, len(X_train)):
    model.fit(X_train[:m], y_train[:m])
    y_train_predict = model.predict(X_train[:m])
    y_val_predict = model.predict(X_val)
    train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
    val_errors.append(mean_squared_error(y_val, y_val_predict))
  plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label='train')
  plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
from sklearn.linear_model import ElasticNet

enet_reg = ElasticNet(l1_ratio=0)
enet_reg.fit(X_prepared_train, y_train_log)
y_pred = np.expm1(enet_reg.predict(X_prepared_train))
rmse_model(y_train, y_pred)
crossvalscores(enet_reg, X_prepared_train, y_train, 20)
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(loss='huber')
sgd_reg.fit(X_prepared_train, y_train_log)
y_pred=sgd_reg.predict(X_prepared_train)
rmse_model(y_train_log, y_pred)
crossvalscores(sgd_reg, X_prepared_train, y_train_log, cv=10)
from sklearn.svm import SVR
svr_reg = SVR(kernel='linear')
svr_reg.fit(X_prepared_train, y_train_log)
y_pred= svr_reg.predict(X_prepared_train)
# y_pred= np.expm1(y_pred)
rmse_model(y_train_log, y_pred)
crossvalscores(svr_reg, X_prepared_train, y_train_log, 10)
y_pred = svr_reg.predict(X_prepared_test)
rmse_model(y_test, np.expm1(y_pred))
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth=8)
tree_reg.fit(X_prepared_train, y_train_log)
tree_rmse = np.sqrt(mean_squared_error(y_train_log, tree_reg.predict(X_prepared_train)))
tree_rmse
crossvalscores(tree_reg, X_prepared_train, y_train_log, 10)
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(X_prepared_train, y_train_log)
y_pred = forest_reg.predict(X_prepared_train)
rmse_model(y_train_log, y_pred)
crossvalscores(forest_reg, X_prepared_train, y_train_log, 10)
from xgboost import XGBRFRegressor
xgb_reg = XGBRFRegressor(n_estimators=10, max_depth=5)
xgb_reg.fit(X_prepared_train, y_train_log)
y_pred = xgb_reg.predict(X_prepared_train)
rmse_model(y_train_log, y_pred)
crossvalscores(xgb_reg, X_prepared_train, y_train_log, 10)
from sklearn.ensemble import GradientBoostingRegressor
gbr_reg = GradientBoostingRegressor(random_state=0)
gbr_reg.fit(X_prepared_train, y_train_log)
y_pred = gbr_reg.predict(X_prepared_train)
rmse_model(y_train_log, y_pred)
crossvalscores(gbr_reg, X_prepared_train, y_train_log, 10)
y_pred = np.expm1(y_pred)
rmse_model(y_train, y_pred)
y_test_pred = gbr_reg.predict(X_prepared_test)
y_test_pred = np.expm1(y_test_pred)
rmse_model(y_test, y_test_pred)
from sklearn.model_selection import GridSearchCV
param_grid = [{'n_estimators':[141,142,143], 'max_depth':[2,3,4], 'min_samples_split':[2,3], 'min_samples_leaf':[4,5,6]}]
grid_search = GridSearchCV(gbr_reg, param_grid, cv=5, 
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_prepared_train, y_train_log)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres["params"]):
  print(np.sqrt(-mean_score), params)
grid_search.best_params_
# feature_importances = grid_search.best_estimator_.feature_importances_
# sorted(zip(feature_importances, final_feature_list), reverse=True)
GBR_tuned = grid_search.best_estimator_
y_pred = GBR_tuned.predict(X_prepared_train)
rmse_model(y_train_log, y_pred)
crossvalscores(GBR_tuned, X_prepared_train, y_train_log, cv=10)
y_test_pred = GBR_tuned.predict(X_prepared_test)
rmse_model(y_test, np.expm1(y_test_pred))
# param_grid = [{'C':[0.3,0.4,0.5], 'epsilon':[.03, .035, 0.04], 'gamma':[.015, 0.02, 0.025]}] 'epsilon':[.037, .04, 0.045]
param_grid = [{'C':[0.075, 0.1, 0.15], 'epsilon':[.035, .040, 0.045], 'gamma':[.006, 0.008, 0.01]}]
grid_search = GridSearchCV(svr_reg, param_grid, scoring='neg_mean_squared_error', cv=10, return_train_score=True)
grid_search.fit(X_prepared_train, y_train_log)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
  print(np.sqrt(-mean_score), params)
grid_search.best_params_
SVR_tuned = grid_search.best_estimator_
SVR_tuned.fit(X_prepared_train, y_train_log)
y_pred = SVR_tuned.predict(X_prepared_train)
rmse_model(y_train_log, y_pred)
crossvalscores(SVR_tuned, X_prepared_train, y_train_log, cv=10)
y_pred = SVR_tuned.predict(X_prepared_test)
rmse_model(y_test, np.expm1(y_pred))
from sklearn.ensemble import StackingRegressor
# models = [('svr', SVR(kernel='linear', C=0.5, epsilon=0.03, gamma=0.015)), ('gbr', GradientBoostingRegressor(max_depth=2, n_estimators=140, min_samples_split=2, min_samples_leaf=5))]
# stacked_regressor = StackingRegressor(estimators=models,cv=10)
models = [('svr', SVR_tuned), ('gbr', GBR_tuned)]
stacked_regressor = StackingRegressor(estimators=models,cv=10)
stacked_regressor.fit(X_prepared_train, y_train_log)
y_pred = stacked_regressor.predict(X_prepared_train)
rmse_model(y_train_log, y_pred)
crossvalscores(stacked_regressor, X_prepared_train, y_train_log, 10)
y_test_pred = np.expm1(stacked_regressor.predict(X_prepared_test))
rmse_model(y_test, y_test_pred)
#read test set for final prediction
X_testset = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
X_testset[num_to_object] = X_testset[num_to_object].astype('object')
X_testset.shape

# #drop redundent columns set index to ID
# X_testset.set_index('Id', inplace=True)

# #preprocess
# X_prepared_testset = pre_process.transform(X_testset)

# #predict
# y_testset_pred = np.expm1(stacked_regressor.predict(X_prepared_testset))

# #prepare result and write to CSV
# y_testset_predreshape = np.reshape(y_testset_pred, (len(y_testset_pred),1))
# y_testset_preddf = pd.DataFrame(y_testset_predreshape, columns=['SalePrice'])
# y_testset_preddf.set_index(X_testset.index,inplace=True)
# y_testset_preddf.to_csv('test_submission.csv')