import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from scipy.stats import skew
from math import sqrt

warnings.filterwarnings('ignore')
%matplotlib inline
pd.set_option('display.max_columns', 300)
df_train = pd.read_csv('../input/train.csv', index_col="Id")
df_test = pd.read_csv('../input/test.csv', index_col="Id")
print('Columnas de entrenamiento : '+str(len(df_train.columns)))
print('Columnas de test : '+str(len(df_test.columns)))
y_train = df_train.SalePrice
#Eliminamos la columma del precio en el entrenamiento
#df_train = df_train.drop(columns='SalePrice')
print('Columnas de entrenamiento : '+str(len(df_train.columns)))

print('Columnas de entrenamiento num: '+str(len(df_train._get_numeric_data().columns)))
print('Columnas de entrenamiento obj: '+str(len(df_train.select_dtypes(include='object').columns)))

#check the decoration

df_test.columns

#descriptive statistics summary
#df_train['SalePrice'].describe()
#histogram
#sns.distplot(df_train['SalePrice']);
df_train.shape , df_test.shape
#Visualizacion de los primeros datos. 

df_train.head()
#descriptive statistics summary
df_train['SalePrice'].describe()

#histogram
sns.distplot(df_train['SalePrice']);
df_train.skew(), df_train.kurt()
numeric_features = df_train.select_dtypes(include=[np.number])
numeric_features.columns
correlation = numeric_features.corr()
print(correlation['SalePrice'].sort_values(ascending = False),'\n')

f , ax = plt.subplots(figsize = (14,12))

plt.title('Correlation of Numeric Features with Sale Price',y=1,size=16)

sns.heatmap(correlation,square = True,  vmax=0.8)
   
sns.set()
columns = ['GrLivArea', 'GarageCars','GarageArea','TotalBsmtSF' ,'1stFlrSF' ,'FullBath','TotRmsAbvGrd','YearBuilt']
sns.pairplot(df_train[columns],size = 2 ,kind ='scatter',diag_kind='kde')
plt.show()
df_train_v2=df_train[columns]
columns2 = ['GrLivArea', 'GarageCars','GarageArea','TotalBsmtSF' ,'1stFlrSF' ,'FullBath','TotRmsAbvGrd','YearBuilt']

df_test_v2=df_test[columns2]
#df_train_v2=df_test[columns]
#df_train_v2
total = df_train_v2.isnull().sum().sort_values(ascending=False)
percent = (df_train_v2.isnull().sum()/df_train_v2.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1,join='outer', keys=['Total Missing Count', '% of Total Observations'])
missing_data.index.name =' Numeric Feature'

missing_data.head(20)
columns = ['GrLivArea', 'GarageCars','GarageArea','TotalBsmtSF' ,'1stFlrSF' ,'FullBath','TotRmsAbvGrd','YearBuilt']
df_train_v2=df_train_v2[columns]
columns2 = ['GrLivArea', 'GarageCars','GarageArea','TotalBsmtSF' ,'1stFlrSF' ,'FullBath','TotRmsAbvGrd','YearBuilt']

df_test_v2=df_test_v2[columns2]
def cat_imputation(column, value):
    df_train_v2.loc[df_train_v2[column].isnull(),column] = value
def cat_imputation_test(column, value):
    df_test_v2.loc[df_test_v2[column].isnull(),column] = value    
    
#cat_imputation('MasVnrArea', 0.0)
#cat_imputation_test('MasVnrArea', 0.0)
cat_imputation_test('GarageCars', 0.0)
cat_imputation_test('GarageArea', 0.0)
cat_imputation_test('TotalBsmtSF', 0.0)

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
df_train=df_train_v2
df_test=df_test_v2

#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(df_train.values)
    rmse= np.sqrt(-cross_val_score(model, df_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#GB_model = ENet.fit(df_train, y_train)
# Defining two rmse_cv functions
def rmse_cv2(model):
    rmse = np.sqrt(-cross_val_score(model, df_train, y_train, scoring="neg_mean_squared_error", cv = 10))
    return(rmse)
#model_elastic = make_pipeline(RobustScaler(), ElasticNet(alpha = 0.0005))
model_elastic = Ridge(alpha = 5)
#cv_elastic = rmse_cv2(model_elastic).mean()

model_elastic.fit(df_train, y_train)
krr_pred = model_elastic.predict(df_test)
#krr_pred = model_elastic.predict(df_test)



# Setting up competition submission
sub = pd.DataFrame()
#sub['Id'] = df_test.index
sub['Id'] = df_test.index
sub['SalePrice'] = krr_pred
sub.to_csv('submission.csv',index=False)
print("Entrenameinto concluido")
krr_pred
## Getting our SalePrice estimation
#Final_labels = (np.exp(GB_model.predict(df_test))) 
## Saving to CSV
#pd.DataFrame({'Id': df_test.index, 'SalePrice': Final_labels}).to_csv('submission.csv', index =False)
#print("Fichero creado")