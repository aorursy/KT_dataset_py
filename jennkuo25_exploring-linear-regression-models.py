import numpy as np # linear algebra
import pandas as pd # data processing

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
testID = test['Id']

data = pd.concat([train.drop('SalePrice', axis=1), test], keys=['train', 'test'])
data.drop(['Id'], axis=1, inplace=True)
data.head()
# Create list of years and metrics
years = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
metrics = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
         '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 
         'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
data[years].max()
mask = (data[years] > 2018).any(axis=1) # take any index with illogical year value
data[mask]['GarageYrBlt']
data.loc[mask, 'GarageYrBlt'] = data[mask]['YearBuilt']
data[mask]['GarageYrBlt']
# Categorize features
numerical_feats = data.select_dtypes(include = ['float', 'integer']).columns
categorical_feats = data.select_dtypes(include = 'object').columns
numerical_feats
# List all columns indicating a grade (refer to data dictionary)
grades = ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
          'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
data[grades].head()
# By looking at the data dictionary, we can assign the literal grades to a numerical value that will be more informative
literal = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
num = [9, 7, 5, 3, 2]
# Create a dictionary
G = dict(zip(literal, num))
data[grades] = data[grades].replace(G)
data[grades].head()
# Create a list of column names from documentation that are *meant* to be categorical
nominal_features = ["MSSubClass", "MSZoning", "Street", "Alley", "LandContour", "LotConfig", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "Heating", "CentralAir", "GarageType", "MiscFeature", "SaleType", "SaleCondition"]
# Explore which numerical columns should be converted to categorical
numerical_df = data.select_dtypes(include = ['float', 'integer'])
cat_cols = []
for col in numerical_df.columns:
    if col in nominal_features:
        cat_cols.append(col)
cat_cols
# Adjust data type of variable MSSubClass to categorical
data['MSSubClass'] = data['MSSubClass'].astype('object', copy=False)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import norm, skew

sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
# Use the numpy fuction log1p which  applies log(1+x) to transform the target
price = np.log1p(train['SalePrice'])

# Check the new distribution 
sns.distplot(price , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(price)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(price, plot=plt)
plt.show()
# Identify skewed continuous numerical features:
skewed_feats = data.loc['train'][metrics].apply(lambda x: x.skew(skipna=True)) #compute skewness
skewed_feats
# Get index of skewed features
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
# Log transform skewed features
data[skewed_feats] = np.log1p(data[skewed_feats])
train['SalePrice'].head()
# Add log-transformed SalePrice to train dataset
train['SalePrice'] = np.log1p(train['SalePrice'])
train['SalePrice'].head()
# Observe correlation to determine which features to look at
print("Find most important features relative to target")
abs_corr_eff = train.corr()['SalePrice'].abs().sort_values(ascending = False)
print(abs_corr_eff)
# Show columns with a correlation coefficient of larger than 0.5
abs_corr_eff[abs_corr_eff > 0.5]
# Look into OverallQual column since highly correlated
sns.set_style("darkgrid")
fig, ax = plt.subplots()
ax.scatter(x = train['OverallQual'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('OverallQuality', fontsize=13)
plt.show()
# Look into OverallQual column since highly correlated
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
# Look into GarageCars column since highly correlated
fig, ax = plt.subplots()
ax.scatter(x = train['GarageCars'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GarageCars', fontsize=13)
plt.show()
# Check null values of numerical features
nulls_num = pd.DataFrame(data[numerical_feats].isnull().sum().sort_values(ascending=False))
nulls_num.columns = ['Null Count']
nulls_num = nulls_num[nulls_num['Null Count'] > 0]
nulls_num
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
# Fill in missing values of numerical columns with 0
data[numerical_feats] = data[numerical_feats].fillna(0)
nulls_num = pd.DataFrame(data[numerical_feats].isnull().sum().sort_values(ascending=False))
nulls_num.columns = ['Null Count']
nulls_num = nulls_num[nulls_num['Null Count'] > 0]
nulls_num
# Let's look at PoolQC values to check distribution
data['PoolQC'].value_counts()
# Check null values of categorical columns
nulls_cat = pd.DataFrame(data[categorical_feats].isnull().sum().sort_values(ascending=False))
nulls_cat.columns = ['Null Count']
nulls_cat = nulls_cat[nulls_cat['Null Count'] > 0]
nulls_cat
# Fill in missing value with NA = typical
data["Functional"] = data["Functional"].fillna("Typ")

# Fill in missing value with most common value
data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])

# Fill in missing value with most common value
data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])

# Fill in missing value with most common value
data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])

# Substitute most common string for missing values
data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])

# Fill in most common sale type for missing value
data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
# Check null values of categorical columns
nulls_cat = pd.DataFrame(data[categorical_feats].isnull().sum().sort_values(ascending=False))
nulls_cat.columns = ['Null Count']
nulls_cat = nulls_cat[nulls_cat['Null Count'] > 0]
nulls_cat
# Fill in missing values of categorical columns with '0'
data[categorical_feats] = data[categorical_feats].fillna('0')
nulls_cat = pd.DataFrame(data[categorical_feats].isnull().sum().sort_values(ascending=False))
nulls_cat.columns = ['Null Count']
nulls_cat = nulls_cat[nulls_cat['Null Count'] > 0]
nulls_cat
# Adding total sqfootage feature 
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
# Engineer new features for years before sale, years since remodel
years_sold = data['YrSold'] - data['YearBuilt']  
years_remodeled = data['YrSold'] - data['YearRemodAdd'] 
data['Years Before Sale'] = years_sold
data['Years Since Remodel'] = years_remodeled
data.head()
train[categorical_feats].describe()
# Drop columns with less than 0.6 correlation with SalePrice
feats_to_drop = abs_corr_eff[abs_corr_eff < 0.6].index
for i in feats_to_drop:
    if i in data.columns:
        data = data.drop(i, axis=1)
    else:
        pass
# Drop utilities column since no information added for predictive modeling
feats_to_drop_1 = ['Utilities', 'PoolQC', 'Alley', 'Street']
for i in feats_to_drop_1:
    if i in data.columns:
        data = data.drop(i, axis=1)
    else:
        pass
# Drop old year columns that were replaced with 'Years Before Sale' and 'Years Since Remodel' feature
feats_to_drop_2 = ['YearBuilt', 'YearRemodAdd', 'YrSold']
for i in feats_to_drop_2:
    if i in data.columns:
        data = data.drop(i, axis=1)
    else:
        pass
# Drop old year columns that were replaced with 'TotalSF' feature
feats_to_drop_3 = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
for i in feats_to_drop_3:
    if i in data.columns:
        data = data.drop(i, axis=1)
    else:
        pass
# Drop Neighborhood column because too many unique values
feats_to_drop_4 = ['Neighborhood']
for i in feats_to_drop_4:
    if i in data.columns:
        data = data.drop(i, axis=1)
    else:
        pass
data.columns
cat_cols = data.select_dtypes(include = 'object').columns
cat_cols
data.shape
finaldata = pd.get_dummies(data)
finaldata.shape
bsmt = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
        'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath',
        'BsmtHalfBath', 
        'TotalBsmtSF']
fire = ['Fireplaces', 'FireplaceQu']
garage = ['GarageQual', 'GarageCond', 'GarageType', 'GarageFinish', 'GarageCars', 
          'GarageArea', 'GarageYrBlt']
masn = ['MasVnrType', 'MasVnrArea']
others = ['Alley', 'Fence', 'PoolQC', 'MiscFeature']

black_list = bsmt + fire + garage + masn + others
for feat in finaldata.columns:
    if ('_0' in feat) and (feat.split("_")[0] in black_list):
        finaldata.drop(feat, axis=1, inplace=True)
finaldata.shape
# Training/testing sets
X_test = finaldata.loc['test']
X_train = finaldata.loc['train']

y_train = price
print(X_train.shape)
print(X_test.shape)
from sklearn.linear_model import LinearRegression

# Create linear regression object
LR = LinearRegression()

# Train the model using the training sets
LR.fit(X_train, y_train)
# Top influencers
maxcoef = np.argsort(-np.abs(LR.coef_))
coef = LR.coef_[maxcoef]
for i in range(0, 5):
    print("{:.<025} {:< 010.4e}".format(finaldata.columns[maxcoef[i]], coef[i]))
from sklearn.linear_model import LassoCV

# Create linear regression object
Ls = LassoCV()

# Train the model using the training sets
Ls.fit(X_train, y_train)
maxcoef = np.argsort(-np.abs(Ls.coef_))
coef = Ls.coef_[maxcoef]
for i in range(0, 5):
    print("{:.<025} {:< 010.4e}".format(finaldata.columns[maxcoef[i]], coef[i]))
from sklearn.linear_model import RidgeCV

# Create linear regression object
Rr = RidgeCV()

# Train the model using the training sets
Rr.fit(X_train, y_train)
maxcoef = np.argsort(-np.abs(Rr.coef_))
coef = Rr.coef_[maxcoef]
for i in range(0, 5):
    print("{:.<025} {:< 010.4e}".format(finaldata.columns[maxcoef[i]], coef[i]))
from sklearn.linear_model import ElasticNetCV

# Create linear regression object
EN = ElasticNetCV(l1_ratio=np.linspace(0.1, 1.0, 5)) # we are essentially smashing most of the Rr model here

# Train the model using the training sets
train_EN = EN.fit(X_train, y_train)
maxcoef = np.argsort(-np.abs(EN.coef_))
coef = EN.coef_[maxcoef]
for i in range(0, 5):
    print("{:.<025} {:< 010.4e}".format(finaldata.columns[maxcoef[i]], coef[i]))
from sklearn.model_selection import cross_val_score

model = [Ls, Rr, EN]
M = len(model)
score = np.empty((M, 5))
for i in range(0, M):
    score[i, :] = cross_val_score(model[i], X_train, y_train, cv=5)
# print out all scores
score
# get the average of scores for each model
print(score.mean(axis=1))
# My model based on ridge regression performed the best through cross-validation
submit = pd.DataFrame({'Id': testID, 'SalePrice': np.exp(Rr.predict(X_test))})
submit.to_csv('submission.csv', index=False)