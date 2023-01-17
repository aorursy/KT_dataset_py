# EDA
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt  
import seaborn as sns; color = sns.color_palette(); sns.set_style('darkgrid')
from scipy import stats
from scipy.stats import norm, skew 
# Data Preprocessing
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox1p
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
# Models
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.shape
test.shape
# Store the 'ID' column and drop it from the train and test datasets
train_ID = train['Id']
test_ID = test['Id']

train.drop('Id', axis = 1, inplace = True)
test.drop('Id', axis = 1, inplace = True)
# Check for skewness in target data
sns.distplot(train['SalePrice'] , fit=norm);
# Check for Correlation between target and numeric features
numeric_data = train.select_dtypes(include=[np.number])

plt.figure(figsize=(15,7))
corr = numeric_data.corr()
top_corr_features = corr.index[abs(corr["SalePrice"])>0.3]
sns.heatmap(numeric_data[top_corr_features].corr(), annot=True,cmap="RdYlGn")

# Top features found: 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF'
top_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF']
n = 1
plt.figure(figsize=(15,8))
for each in top_features:
    plt.subplot(2, 3, n)
    plt.scatter(x = train[each], y = train['SalePrice'])
    plt.title(each)
    n = n + 1
plt.tight_layout()
plt.show()
### Outlier treatment

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
# Correct skewness in target
train["SalePrice"] = np.log1p(train["SalePrice"])

# Check target again
sns.distplot(train['SalePrice'] , fit=norm)
# Store SalePrice in y_train
y_train = train.SalePrice.values
ntrain = train.shape[0]
ntest = test.shape[0]

all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

all_data["Alley"] = all_data["Alley"].fillna("None")

all_data["Fence"] = all_data["Fence"].fillna("None")

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    all_data[col] = all_data[col].fillna('None')
    
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
    
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
    
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
    
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data = all_data.drop(['Utilities'], axis=1)

all_data["Functional"] = all_data["Functional"].fillna("Typ")

mode_col = ['Electrical','KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']
for col in mode_col:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
# Label encode categorical features

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))
   
all_data.shape
# Create new feature which is the total of all square footages

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
# Calculate the skew of all numerical features

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(5)
# Filter features with a skewness above 0.75
skewness = skewness[abs(skewness) > 0.75]

skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
# Dummy creation
all_data = pd.get_dummies(all_data)
all_data.shape
X_train = all_data[:ntrain]

test = all_data[ntrain:]
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
lasso.fit(X_train,y_train)
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
ENet.fit(X_train,y_train)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
GBoost.fit(X_train,y_train)
ridge = make_pipeline(RobustScaler(), Ridge(alpha=0.0005, normalize=True))
ridge.fit(X_train,y_train)
predictions = (np.expm1(lasso.predict(test.values)) + np.expm1(ENet.predict(test.values)) + np.expm1(ridge.predict(test.values)) + np.expm1(GBoost.predict(test.values)) ) / 4
predictions
preds = pd.DataFrame()
preds['Id'] = test_ID
preds['SalePrice'] = predictions
preds.to_csv('predictions.csv',index=False)