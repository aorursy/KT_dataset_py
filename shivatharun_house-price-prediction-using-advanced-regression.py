import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm,skew
train= pd.read_csv("../input/train.csv")
test =pd.read_csv("../input/test.csv")
train.columns
train.describe()
test.describe()
print("Train : "+str(train.shape))

#checking for duplicates
idUn = len(set(train.Id))
idTo = train.shape[0]
idDup = idTo - idUn
print(str(idDup)+" duplicates available in this dataset")
train_ID = train['Id']
test_ID = test['Id']

#Delete the ID Column
train.drop('Id',axis=1,inplace = True)
test.drop('Id', axis=1, inplace = True)

#After dropping Id Column
print("Train Data: "+str(train.shape))
print("Test Data: "+str(test.shape))
#Select the Numerical & Categorical Features

numerical_features = train.select_dtypes(exclude = ['object']).columns
categorical_features = train.select_dtypes(include = ['object']).columns
# Plotting the numerical columns
fig = plt.figure(figsize = (15,15))
ax = fig.gca()
train[numerical_features].hist(ax=ax)
fig.tight_layout()
fig.show()

#plot the Numeric columns against SalePrice Using ScatterPlot

fig = plt.figure(figsize=(15,30))
for i,col in enumerate(numerical_features[1:]):
    fig.add_subplot(12,3,1+i)
    plt.scatter(train[col], train['SalePrice'])
    plt.xlabel(col)
    plt.ylabel('SalePrice')
fig.tight_layout()
fig.show()
fig = plt.figure(figsize=(15,50))
for i, col in enumerate(categorical_features):
    fig.add_subplot(11,4,1+i)
    train.groupby(col).mean()['SalePrice'].plot.bar(yerr = train.groupby(col).std())
fig.tight_layout()
fig.show()
sns.set_style('darkgrid')
fig, ax = plt.subplots()
sns.regplot(train['GrLivArea'], train['SalePrice'])
#ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

fig, ax = plt.subplots()
sns.regplot(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
train.SalePrice.describe()
#PLot Histogram for 'SalePrice'
sns.distplot(train['SalePrice'])
#Skewness & Kurtosis
print("Skewness : %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())
#PLot Histogram for 'SalePrice'
stats.probplot(train['SalePrice'], plot=plt)
train['SalePrice'] = np.log1p(train['SalePrice'])

#Normal Distribution of New Sales Price
mu, sigma = norm.fit(train['SalePrice'])
print("Mu : {:.2f}\nSigma : {:.2f}".format(mu,sigma))

#Visualization
sns.distplot(train['SalePrice'],fit=norm);
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\Sigma=$ {:.2f})'.format(mu,sigma)],loc = 'best')
plt.xlabel('SalePrice Distribution')
plt.ylabel('Frequency')

fig = plt.figure()
res = stats.probplot(train['SalePrice'],plot=plt)
plt.show()
train_n = train.shape[0]
test_n = test.shape[0]
y_train = train.SalePrice.values
y_test = train['SalePrice']
all_data = pd.concat((train,test),sort=False).reset_index(drop = True)
all_data.drop(['SalePrice'], axis=1, inplace = True)
print("all_data size is : {}".format(all_data.shape))
all_data.isnull().sum().sort_values(ascending=False)


all_data_na_values = all_data.isnull().sum()
all_data_na_values = all_data_na_values.drop(all_data_na_values[all_data_na_values == 0].index).sort_values(ascending=False)[:30]
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na,'Missing Values' :all_data_na_values})
missing_data.head(20)
plt.subplots(figsize = (15,12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index,y=all_data_na)
plt.xlabel('Features',fontsize=15)
plt.ylabel('Percent of Missing Values', fontsize=15)
plt.title('% of Misssing data by Features', fontsize=15)


#Correlation map to see how features are correlated with SalePrice
corrmat = train.corr()
plt.subplots(figsize=(25,15))
sns.heatmap(corrmat, vmax=0.9, square=True, annot=True, fmt=".2f")
# Fill the Missing Values
all_data['PoolQC'] = all_data['PoolQC'].fillna("None")
all_data['MiscFeature'] = all_data['MiscFeature'].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
    
for col in ('BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1'):
    all_data[col] = all_data[col].fillna('None')

for col in ('BsmtHalfBath', 'BsmtFullBath', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1'):
    all_data[col] = all_data[col].fillna(0)

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data_na_values = all_data.isnull().sum()
all_data_na_values = all_data_na_values.drop(all_data_na_values[all_data_na_values == 0].index).sort_values(ascending=False)[:30]
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na,'Missing Values' :all_data_na_values,'Data_type':all_data_na.dtype})
missing_data.head()

#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))
#Adding Total sqfoot feature
all_data['TotalSF'] = all_data['TotalBsmtSF']+all_data['1stFlrSF']+all_data['2ndFlrSF']

from scipy.stats import skew
num = all_data.dtypes[all_data.dtypes != 'object'].index

#Skew all the Numerical Features
skew_feat = all_data[num].apply(lambda x: skew(x.dropna())).sort_values(ascending = False)

sk = pd.DataFrame({'Skewness' :skew_feat})
sk.head(10)
#should/need to define categorical columns list
all_data = pd.get_dummies(all_data)
print(all_data.shape)
train_new = all_data[:train_n]
test_new = all_data[train_n:]
print(train_new.shape)
print(test_new.shape)
 
import xgboost as xgb

regr = xgb.XGBRegressor(colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.1,
                 max_depth=4,
                 min_child_weight=1.5,
                 n_estimators=7200,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)

regr.fit(train_new,y_train)

y_pred = regr.predict(train_new)
y_test = train['SalePrice']

from sklearn.metrics import mean_squared_error
print("XGB Score :",(np.sqrt(mean_squared_error(y_test, y_pred))))

y_pred_xgb = regr.predict(test_new)

y_pred_xgb = np.exp(y_pred_xgb)

pred_df = pd.DataFrame(y_pred_xgb, index=test_ID, columns=["SalePrice"])
pred_df.to_csv('submission1.csv', header=True, index_label='Id')
from sklearn.linear_model import ElasticNet, Lasso,Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
model_lasso =Lasso(alpha=0.0005,normalize=True, max_iter=1e5)
model_lasso.fit(train_new,y_train)
y_pred_lasso = model_lasso.predict(train_new)
score_lasso = np.sqrt(mean_squared_error(y_train, y_pred_lasso))
print("Lasso Score :",score_lasso)

y_pred_lasso_test = model_lasso.predict(test_new)
y_pred_lasso_test = np.exp(y_pred_lasso_test)
model_rd = Ridge(alpha = 4.84)
model_rd.fit(train_new,y_train)
y_pred_rd = model_rd.predict(train_new)
score_rd = np.sqrt(mean_squared_error(y_train, y_pred_rd))
print("Ridge Score :",score_rd)

y_pred_rd_test = model_rd.predict(test_new)
y_pred_rd_test = np.exp(y_pred_rd_test)
model_enet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)
model_enet.fit(train_new,y_train)
y_pred_enet = model_enet.predict(train_new)
score_enet = np.sqrt(mean_squared_error(y_train, y_pred_enet))
print("ElasticNet Score :",score_enet)

y_pred_enet_test = model_enet.predict(test_new)
y_pred_enet_test = np.exp(y_pred_enet_test)
model_rf = RandomForestRegressor(n_estimators = 12,max_depth = 3,n_jobs = -1)
model_rf.fit(train_new,y_train)
y_pred_rf = model_rf.predict(train_new)
score_rf = np.sqrt(mean_squared_error(y_train, y_pred_rf))
print("RandomForest Score :",score_rf)

y_pred_rf_test = model_rf.predict(test_new)
y_pred_rf_test = np.exp(y_pred_rf_test)
model_gb = GradientBoostingRegressor(n_estimators = 40,max_depth = 2)
model_gb.fit(train_new,y_train)
y_pred_gb = model_gb.predict(train_new)
score_gb = np.sqrt(mean_squared_error(y_train, y_pred_gb))
print("GradientBoosting Score :",score_gb)

y_pred_gb_test = model_gb.predict(test_new)
y_pred_gb_test = np.exp(y_pred_gb_test)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=4)
model_ad = AdaBoostRegressor(dt, learning_rate = 0.1, n_estimators=300,random_state= None)
model_ad.fit(train_new,y_train)
y_pred_ad = model_ad.predict(train_new)
score_ad = np.sqrt(mean_squared_error(y_train, y_pred_ad))
print("AdaBoost Model :",score_ad)

y_pred_ad_test = model_ad.predict(test_new)
y_pred_ad_test = np.exp(y_pred_ad_test)