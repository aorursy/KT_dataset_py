#Importing libraries
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
import pandas as pd
import numpy as np
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv',index_col = 'Id')
test = pd.read_csv('../input/test.csv',index_col = 'Id')
test.shape
train.shape
#train.info()
#test.info()
train.head()
train.columns
train.describe()
num_feat=train.select_dtypes(include=[np.number])
cat_feat=train.select_dtypes(include=[np.object])
print('Numerical Features:\n',num_feat.dtypes,'\n')
print('Categorical Features:\n',cat_feat.dtypes)
plt.figure(figsize=(12,7))
sns.distplot(train['SalePrice'],fit = norm);
train['SalePrice'].describe()
#Skewness and Kurtosis for Target Variable
print('Skewness :',train['SalePrice'].skew())
print('Kurtosis :',train['SalePrice'].kurt())
plt.figure(figsize = (12,7))
sns.distplot(np.log(train.SalePrice),fit = norm);
print('Skewness = ',np.log(train.SalePrice).skew())
train.corr()['SalePrice'].sort_values(ascending=False)
features = ['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF',
                    'GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea']
plt.figure(figsize = (15,15))
sns.heatmap(train[features].corr(),annot = True,linewidths = 0.5,cmap='cubehelix_r');
plt.savefig('Correlation Heatmap.png')
#Plotting regression plot for GrLivArea
plt.figure(figsize = (10,7))
sns.regplot('GrLivArea','SalePrice',data=train,color = 'red');

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
plt.figure(figsize = (10,7))
sns.regplot('GrLivArea','SalePrice',data=train,color = 'red');
#Visualizing Garage Area
plt.figure(figsize=(10,7))
sns.regplot('GarageArea','SalePrice',data=train,color='green');
#Removing Outliers from the GarageArea
train = train[train['GarageArea']<1200]
plt.figure(figsize=(10,7))
sns.regplot('GarageArea','SalePrice',data=train,color='green');
#Visualizing TotalBsmtSF
plt.figure(figsize=(10,7))
sns.regplot('TotalBsmtSF','SalePrice',data=train,color='Red');
plt.figure(figsize=(15,7))
plt.subplot(221)
sns.regplot('1stFlrSF','SalePrice',data=train,color = 'Brown');
plt.subplot(222)
sns.regplot('2ndFlrSF','SalePrice',data=train,color = 'Brown');
plt.subplot(223)
sns.regplot(train['1stFlrSF'] + train['2ndFlrSF'],train['SalePrice']);

plt.figure(figsize=(10,7))
sns.regplot('LotFrontage','SalePrice',data=train);  #we can see the outliers here
train = train[train['LotFrontage']<200]   ##Removing Outliers
plt.figure(figsize=(10,7))
sns.regplot('LotFrontage','SalePrice',data=train);
plt.figure(figsize= (15,7))
plt.subplot(121)
sns.boxplot(train['OverallQual'], train['SalePrice']);
plt.subplot(122)
train['OverallQual'].value_counts().plot(kind="bar");
plt.savefig('OverallQual Vs SalePrice.png')
plt.figure(figsize= (20,8))
plt.subplot(121)
sns.boxplot(train['TotRmsAbvGrd'], train['SalePrice']);
sns.stripplot(train["TotRmsAbvGrd"],train["SalePrice"], jitter=True, edgecolor="gray")
plt.subplot(122)
train['TotRmsAbvGrd'].value_counts().plot(kind="bar");
plt.savefig('TotRmsAbvGrd Vs SalePrice.png')

#Sample size is decreasing after Total rooms above grade reaches to 10.
plt.figure(figsize= (15,8))
plt.subplot(121)
sns.boxplot(train['GarageCars'], train['SalePrice']);
sns.stripplot(train["GarageCars"],train["SalePrice"], jitter=True, edgecolor="gray")
plt.subplot(122)
train['GarageCars'].value_counts().plot(kind="bar");
plt.savefig('GarageCars Vs SalePrice.png')
#Median Sale Price going down after 4 Garagecars is undestandable after plotting the points on boxes.
plt.figure(figsize= (15,8))
plt.subplot(121)
sns.boxplot(train['FullBath'], train['SalePrice']);
plt.subplot(122)
train['FullBath'].value_counts().plot(kind="bar");
plt.savefig('FullBath Vs SalePrice.png')
train['log_SalePrice']=np.log(train['SalePrice']+1)
saleprices=train[['SalePrice','log_SalePrice']]

saleprices.head(5)
train=train.drop(columns=['SalePrice','log_SalePrice'])
print(test.shape)
print(train.shape)
all_data = pd.concat((train, test))
print(all_data.shape)
all_data.head()
null_data = pd.DataFrame(all_data.isnull().sum().sort_values(ascending=False))

null_data.columns = ['Null Count']
null_data.index.name = 'Feature'
null_data

# Percentage of Null Data in each Feature

(null_data/len(all_data)) * 100
# Visualising missing data
f, ax = plt.subplots(figsize=(20, 7));
plt.xticks(rotation='90');
sns.barplot(x=null_data.index, y=null_data['Null Count']);
plt.xlabel('Features', fontsize=15);
plt.ylabel('Percent of missing values', fontsize=15);
plt.title('Percent missing data by feature', fontsize=15);
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass'):
    
    all_data[col] = all_data[col].fillna('None')
#Impute the numerical features and replace with a value of zero

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath',
            'BsmtHalfBath', 'MasVnrArea'):
    
    all_data[col] = all_data[col].fillna(0)
for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional', 'Utilities'):
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))

all_data['TotalSF']=all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['No2ndFlr']=(all_data['2ndFlrSF']==0)
all_data['NoBsmt']=(all_data['TotalBsmtSF']==0)

sns.regplot(train['TotalBsmtSF']+train['1stFlrSF']+train['2ndFlrSF'],saleprices['SalePrice'],color='red');
plt.figure(figsize = (12,7))
sns.barplot(train['BsmtFullBath'] + train['FullBath'] + train['BsmtHalfBath'] + train['HalfBath'], saleprices['SalePrice']);

all_data['TotalBath']=all_data['BsmtFullBath'] + all_data['FullBath'] + all_data['BsmtHalfBath'] + all_data['HalfBath']
plt.figure(figsize=(10,7))
sns.regplot((train['YearBuilt']+train['YearRemodAdd']), saleprices['SalePrice']);

all_data['YrBltAndRemod']=all_data['YearBuilt']+all_data['YearRemodAdd']
all_data=all_data.drop(columns=['Street','Utilities','Condition2','RoofMatl',
                                'Heating','PoolArea','PoolQC','MiscVal','MiscFeature'])
# treat some numeric values as str which are infact a categorical variables
all_data['MSSubClass']=all_data['MSSubClass'].astype(str)
all_data['MoSold']=all_data['MoSold'].astype(str)
all_data['YrSold']=all_data['YrSold'].astype(str)
all_data['NoLowQual']=(all_data['LowQualFinSF']==0)
all_data['NoOpenPorch']=(all_data['OpenPorchSF']==0)
all_data['NoWoodDeck']=(all_data['WoodDeckSF']==0)
all_data['NoGarage']=(all_data['GarageArea']==0)
Basement = ['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1',
            'BsmtFinType2', 'BsmtQual', 'BsmtUnfSF','TotalBsmtSF']

Bsmt=all_data[Basement]
Bsmt.head()
Bsmt['BsmtCond'].unique()
Bsmt=Bsmt.replace(to_replace='Po', value=1)
Bsmt=Bsmt.replace(to_replace='Fa', value=2)
Bsmt=Bsmt.replace(to_replace='TA', value=3)
Bsmt=Bsmt.replace(to_replace='Gd', value=4)
Bsmt=Bsmt.replace(to_replace='Ex', value=5)
Bsmt=Bsmt.replace(to_replace='None', value=0)

Bsmt=Bsmt.replace(to_replace='No', value=1)
Bsmt=Bsmt.replace(to_replace='Mn', value=2)
Bsmt=Bsmt.replace(to_replace='Av', value=3)
Bsmt=Bsmt.replace(to_replace='Gd', value=4)

Bsmt=Bsmt.replace(to_replace='Unf', value=1)
Bsmt=Bsmt.replace(to_replace='LwQ', value=2)
Bsmt=Bsmt.replace(to_replace='Rec', value=3)
Bsmt=Bsmt.replace(to_replace='BLQ', value=4)
Bsmt=Bsmt.replace(to_replace='ALQ', value=5)
Bsmt=Bsmt.replace(to_replace='GLQ', value=6)
Bsmt.head()
Bsmt['BsmtScore']= Bsmt['BsmtQual']  * Bsmt['BsmtCond'] * Bsmt['TotalBsmtSF']
all_data['BsmtScore']=Bsmt['BsmtScore']
Bsmt['BsmtFin'] = (Bsmt['BsmtFinSF1'] * Bsmt['BsmtFinType1']) + (Bsmt['BsmtFinSF2'] * Bsmt['BsmtFinType2'])
all_data['BsmtFinScore']=Bsmt['BsmtFin']
all_data['BsmtDNF']=(all_data['BsmtFinScore']==0)
lot=['LotFrontage', 'LotArea','LotConfig','LotShape']
Lot=all_data[lot]
Lot.head()
garage=['GarageArea','GarageCars','GarageCond','GarageFinish','GarageQual','GarageType','GarageYrBlt']
Garage=all_data[garage]

Garage=Garage.replace(to_replace='Po', value=1)
Garage=Garage.replace(to_replace='Fa', value=2)
Garage=Garage.replace(to_replace='TA', value=3)
Garage=Garage.replace(to_replace='Gd', value=4)
Garage=Garage.replace(to_replace='Ex', value=5)
Garage=Garage.replace(to_replace='None', value=0)

Garage=Garage.replace(to_replace='Unf', value=1)
Garage=Garage.replace(to_replace='RFn', value=2)
Garage=Garage.replace(to_replace='Fin', value=3)

Garage=Garage.replace(to_replace='CarPort', value=1)
Garage=Garage.replace(to_replace='Basment', value=4)
Garage=Garage.replace(to_replace='Detchd', value=2)
Garage=Garage.replace(to_replace='2Types', value=3)
Garage=Garage.replace(to_replace='Basement', value=5)
Garage=Garage.replace(to_replace='Attchd', value=6)
Garage=Garage.replace(to_replace='BuiltIn', value=7)

Garage.head()
all_data.head()
non_numeric=all_data.select_dtypes(exclude=[np.number, bool])
non_numeric.head()
def onehot(col_list):
    global all_data
    while len(col_list) !=0:
        col=col_list.pop(0)
        data_encoded=pd.get_dummies(all_data[col], prefix=col)
        all_data=pd.merge(all_data, data_encoded, on='Id')
        all_data=all_data.drop(columns=col)
    print(all_data.shape)
onehot(list(non_numeric))
def log_transform(col_list):
    transformed_col=[]
    while len(col_list)!=0:
        col=col_list.pop(0)
        if all_data[col].skew() > 0.5:
            all_data[col]=np.log(all_data[col]+1)
            transformed_col.append(col)
        else:
            pass
    print(f"{len(transformed_col)} features had been tranformed")
    print(all_data.shape)
numeric=all_data.select_dtypes(include=np.number)
log_transform(list(numeric))
print(train.shape)
print(test.shape)
train=all_data[:len(train)]
test=all_data[len(train):]
print(train.shape)
print(test.shape)
# loading pakages for model. 
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

from sklearn import linear_model, model_selection, ensemble, preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet,SGDRegressor
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
def rmse(predict, actual):
    predict = np.array(predict)
    actual = np.array(actual)
    distance = predict - actual
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    score = np.sqrt(mean_square_distance)
    return score
rmse_score = make_scorer(rmse)
rmse_score
feature_names=list(all_data)
X_train = train[feature_names]
X_test = test[feature_names]
y_train = saleprices['log_SalePrice']
def score(model):
    score = cross_val_score(model, X_train, y_train, cv=5, scoring=rmse_score).mean()
    return score
from IPython.display import YouTubeVideo
#Video tutorial on Bias-Variance Tradeoff

YouTubeVideo('EuBBz3bI-aA',width=700, height=350)
#Video tutorial on ridge regression

YouTubeVideo('Q81RR3yKn30',width=700, height=350)
#Tutorial on Lasso
YouTubeVideo('NGf0voTMlcs',width=700, height=350)
#Elastic Net Regression
YouTubeVideo('1dKRdX9bfIo',width=700, height=350)
#Decision Tree
YouTubeVideo('7VeUPuFGJHk',width=700, height=350)
model_Lasso= make_pipeline(RobustScaler(), Lasso(alpha =0.000327, random_state=18))

model_ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.00052, l1_ratio=0.70654, random_state=18))


model_GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =18)

model_XGB=xgb.XGBRegressor(n_jobs=-1, n_estimators=849, learning_rate=0.015876, 
                           max_depth=58, colsample_bytree=0.599653, colsample_bylevel=0.287441, subsample=0.154134, seed=18)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

forest_reg = RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=None,
           max_features=60, max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=70, n_jobs=1, oob_score=False, random_state=42,
           verbose=0, warm_start=False)

model_Lasso.fit(X_train, y_train)
Lasso_Predictions=np.exp(model_Lasso.predict(X_test))-1

model_ENet.fit(X_train, y_train)
ENet_Predictions=np.exp(model_ENet.predict(X_test))-1

model_XGB.fit(X_train, y_train)
XGB_Predictions=np.exp(model_XGB.predict(X_test))-1

model_GBoost.fit(X_train, y_train)
GBoost_Predictions=np.exp(model_GBoost.predict(X_test))-1

model_lgb.fit(X_train, y_train)
lgb_Predictions=np.exp(model_lgb.predict(X_test))-1

forest_reg.fit(X_train, y_train)
forest_reg_Predictions=np.exp(forest_reg.predict(X_test))-1

scores ={}
scores.update({'Lasso':score(model_Lasso)})
scores.update({"Elastic Net":score(model_ENet)})

scores.update({"XGB":score(model_XGB)})
scores.update({"Gradient Boost":score(model_GBoost)})
scores.update({"lgb":score(model_lgb)})
scores.update({"Random Forest":score(forest_reg)})
scores
scores_df =pd.DataFrame(list(scores.items()),columns=['Model','Score'])
scores_df.sort_values(['Score'])
plt.figure(figsize=(15,7))
sns.barplot(scores_df['Model'],scores_df['Score']);
from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://blogs.sas.com/content/subconsciousmusings/files/2017/05/weighted-unweighted.png")
ensemble = (Lasso_Predictions + XGB_Predictions + lgb_Predictions + ENet_Predictions +
           forest_reg_Predictions)/5

ensemble
submission=pd.read_csv('../input/sample_submission.csv')
submission['SalePrice']= ensemble
submission.head()
submission.to_csv('submission.csv',index=False)
