#Import Libraries
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from scipy.stats import skew, skewtest
import warnings
warnings.filterwarnings('ignore')
#Load data
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# train Data shape
print('train Data Shape',train.shape)
# Test data shape
print('test data shape',test.shape)
datasetHasNan = False
if train.count().min() == train.shape[0] and test.count().min() == test.shape[0] :
    print('We do not need to worry about missing values.') 
else:
    datasetHasNan = True
    print('yes, we have missing values')

# now list items    
print('--'*40) 
if datasetHasNan == True:
    nas = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['Train Dataset', 'Test Dataset']) 
    print('Nan in the data sets')
    print(nas[nas.sum(axis=1) > 0])
# Check for percentages they are missing

nap= pd.concat([round(train.isnull().sum().sort_values(ascending = False)/len(train)*100,2),
                (round(test.isnull().sum().sort_values(ascending = False)/len(train)*100,2))],
                axis=1, keys=['Train Dataset', 'Test Dataset']) 
print('--'*40) 
print('Missing Data Percentage')
print('--'*40) 
print(nap)
# Explore features
def feat_explore(column):
    return train[column].value_counts()

# Function to impute missing values
def feat_impute(column, value):
    train.loc[train[column].isnull(),column] = value
    test.loc[test[column].isnull(),column] = value
#PoolQC, MiscFeature, Alley, Fence will all be removed as they are missing over half of their observations.(Over 50% Missing)

features_drop = ['PoolQC','MiscFeature','Alley','Fence']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)   
print('TRAIN: FireplaceQu Missing Before:', train['FireplaceQu'].isnull().sum(),'\n',
      'TEST: FireplaceQu Missing Before:', test['FireplaceQu'].isnull().sum())
#TRAIN: missing 690 observations. 
#TEST: missing 730 observations. 
#However, these nulls may be attributed to homes that do not have fireplaces at all. 
#If this assumption proves to be true, we can impute these nulls with '0' as they do not have a fireplace.

# checking this assumption
assumption1 = pd.concat([(train[train['FireplaceQu'].isnull()][['Fireplaces','FireplaceQu']]),
                (test[test['FireplaceQu'].isnull()][['Fireplaces','FireplaceQu']])], 
                axis=1, keys=['Train Dataset',' Test Dataset']) 
print(assumption1)
print('TRAIN: shape :', (train[train['FireplaceQu'].isnull()][['Fireplaces','FireplaceQu']].shape),
        '\n','TEST: shape :',  (train[train['FireplaceQu'].isnull()][['Fireplaces','FireplaceQu']].shape))

# Impute the nulls with None 
train['FireplaceQu'] = train['FireplaceQu'].fillna('None')
test['FireplaceQu'] = test['FireplaceQu'].fillna('None')

# Cross check columns
print('Confirm Imputation for train')
print(pd.crosstab(train.FireplaceQu,train.Fireplaces,))
print('--'*40)
print('Confirm Imputation for test')
print(pd.crosstab(test.FireplaceQu,test.Fireplaces,))
print('TRAIN: LotFrontage Missing Before:', train['LotFrontage'].isnull().sum(),'\n','TEST: LotFrontage Missing Before:', test['LotFrontage'].isnull().sum())

#TRAIN: missing 259 observations. 
#TEST: missing 227 observations. 
#First check if there are other variables that are strongly correlated with LotFrontage can be used for imputation. 
#Otherwise impute with the median LotFrontage value.

# Check above mentioned assumption
corr_lf = train.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()
cor_dict_lf = corr_lf['LotFrontage'].to_dict()
del cor_dict_lf['LotFrontage']
print("Numeric features by Correlation with LotFrontage:\n")
for ele in sorted(cor_dict_lf.items(), key = lambda x: -abs(x[1])):
    print("{0}: \t{1}".format(*ele))

# Nothing highly correlated to LotFrontage so we will impute with the mean

train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].median())
test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].median())

print('Garage Features Missing Before')
print('--'*40)
print(pd.concat([(train[['GarageYrBlt', 'GarageType', 'GarageFinish','GarageQual','GarageCond']].isnull().sum()),
                (train[['GarageYrBlt', 'GarageType', 'GarageFinish','GarageQual','GarageCond']].isnull().sum())], 
                axis=1, keys=['Train Dataset',' Test Dataset']) )

#TRAIN: missing 259 observations. 
#TEST: missing 227 observations. 
#These null values are assumed to be in the same rows for each column and associated with homes 
#that do not have garages at all. 
#If these assumptions are correct, the nulls can be inputed with zero as these are properties without garages.

# Assumptions check
print('--'*40)
print('Assumption Check TRAIN DATASET')
null_garage = ['GarageYrBlt','GarageType','GarageQual','GarageCond','GarageFinish']
print(train[(train['GarageYrBlt'].isnull())|
                 (train['GarageType'].isnull())|
                 (train['GarageQual'].isnull())|
                 (train['GarageCond'].isnull())|
                 (train['GarageFinish'].isnull())]
                 [['GarageCars','GarageYrBlt','GarageType','GarageQual','GarageCond','GarageFinish']])
print('--'*40)
print('Assumption Check TEST DATASET')
print(test[(test['GarageYrBlt'].isnull())|
                 (test['GarageCond'].isnull())|
                 (test['GarageQual'].isnull())|
                (test['GarageFinish'].isnull())|
                (test['GarageType'].isnull())|
                (test['GarageCars'].isnull())|
                (test['GarageArea'].isnull())]
                 [['GarageYrBlt','GarageCond','GarageQual', 'GarageFinish','GarageType','GarageCars','GarageArea']])


# Impute nulls at index 666 that have a garage with most common value in each column for categorical variables 
test.iloc[666, test.columns.get_loc('GarageYrBlt')] = test['GarageYrBlt'].mode()[0]
test.iloc[666, test.columns.get_loc('GarageCond')] = test['GarageCond'].mode()[0]
test.iloc[666, test.columns.get_loc('GarageFinish')] = test['GarageFinish'].mode()[0]
test.iloc[666, test.columns.get_loc('GarageQual')] = test['GarageQual'].mode()[0]
test.iloc[666, test.columns.get_loc('GarageType')] = test['GarageType'].mode()[0]

# Impute nulls at index 1116 that have a garage with most common value in each column for categorical variables 
test.iloc[1116, test.columns.get_loc('GarageYrBlt')] = test['GarageYrBlt'].mode()[0]
test.iloc[1116, test.columns.get_loc('GarageCond')] = test['GarageCond'].mode()[0]
test.iloc[1116, test.columns.get_loc('GarageFinish')] = test['GarageFinish'].mode()[0]
test.iloc[1116, test.columns.get_loc('GarageQual')] = test['GarageQual'].mode()[0]
test.iloc[1116, test.columns.get_loc('GarageType')] = test['GarageType'].mode()[0]

# Impute nulls at index 1116 that have a garage with median value in each column for continuous variables 
test.iloc[1116, test.columns.get_loc('GarageCars')] = test['GarageCars'].median()
test.iloc[1116, test.columns.get_loc('GarageArea')] = test['GarageArea'].median()
# Impute the remaining nulls as None
null_garage2 = ['GarageYrBlt','GarageCond','GarageFinish','GarageQual', 'GarageType','GarageCars','GarageArea']

for cols in null_garage2:
    if(train[cols].dtype ==np.object)&(test[cols].dtype ==np.object) :
         feat_impute(cols, 'None')
    else:
         feat_impute(cols, 0)

# Cross check columns

print('Confirm Imputation')
for cols in null_garage:
    print(pd.crosstab(train[cols],train.GarageCars))
    print(pd.crosstab(test[cols],test.GarageCars))

null_bsmt = ['BsmtFullBath','BsmtHalfBath','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
             'TotalBsmtSF','BsmtCond','BsmtExposure','BsmtQual','BsmtFinType1','BsmtFinType2']
print('Missing Data Before','\n')
for cols in null_bsmt:
    print('TRAIN:',cols,train[cols].isnull().sum())
    print('TEST:',cols,test[cols].isnull().sum())
    print('--'*40)
train['BsmtQual'].fillna('NA', inplace = True)
test['BsmtQual'].fillna('NA', inplace = True)

train['BsmtCond'].fillna('NA', inplace = True)
test['BsmtCond'].fillna('NA', inplace = True)

train['BsmtExposure'].fillna('NA', inplace = True)
test['BsmtExposure'].fillna('NA', inplace = True)

train['BsmtFinType1'].fillna('NA', inplace = True)
test['BsmtFinType1'].fillna('NA', inplace = True)

train['BsmtFinType2'].fillna('NA', inplace = True)
test['BsmtFinType2'].fillna('NA', inplace = True)

test['BsmtFullBath'].fillna('NA', inplace = True)
test['BsmtHalfBath'].fillna('NA', inplace = True)
test['BsmtFinSF1'].fillna('NA', inplace = True)
test['BsmtFinSF2'].fillna('NA', inplace = True)
print('Masonry Features Missing Before')
print(pd.concat([(train[['MasVnrArea', 'MasVnrType']].isnull().sum()),
                (test[['MasVnrArea', 'MasVnrType']].isnull().sum())], 
                axis=1, keys=['Train Dataset',' Test Dataset']) )
# MasVnrArea and MasVnrType are each missing 8 observations
# Confirm that the missing values in these columns are the same rows
print('Check Assumptions FOR TRAIN SET')
print(train[(train['MasVnrArea'].isnull())|(train['MasVnrType'].isnull())]
                 [['MasVnrArea','MasVnrType']])

print(train[(train['MasVnrArea'].isnull())|(train['MasVnrType'].isnull())]
                 [['MasVnrArea','MasVnrType']].shape)

# View nulls in masonry features in Test set now
print('--'*40,'\nAssumption Check FOR TEST SET')
print(test[(test['MasVnrType'].isnull())|(test['MasVnrType'].isnull())|
                (test['MasVnrArea'].isnull())|(test['MasVnrArea'].isnull())]
                 [['MasVnrType','MasVnrArea']])
# Impute `MasVnrArea` with the most frequent values
# feat_explore('MasVnrArea')
# feat_impute('MasVnrArea','None')
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mode()[0])

# Impute `MasVnrType` with the most frequent values
# feat_explore('MasVnrType')
# feat_impute('MasVnrType',0.0)

# Impute exceptions to assumption that nulls correspond to homes with no exposure
test.iloc[1150, test.columns.get_loc('MasVnrType')] = test['MasVnrType'].mode()[0]

train['MasVnrType'] = train['MasVnrType'].fillna(train['MasVnrType'].mode()[0])

# create list 
null_masonry = ['MasVnrType','MasVnrArea']

for cols in null_masonry:
    if((train[cols].dtype ==np.object)&(test[cols].dtype ==np.object)):
        feat_impute(cols, 'None')
    else:
        feat_impute(cols, 0)
# Electrical is only missing one value

print('Electrical Feature Missing Before')
print(train[['Electrical']].isnull().sum())


# Impute Electrical with the most frequent value, 'SBrkr'
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])

#check now
print('Electrical Feature Missing After')
print(train[['Electrical']].isnull().sum())
print('--'*40)
null_others = ['MSZoning', 'Utilities','Functional','Exterior2nd','Exterior1st','SaleType','KitchenQual'] 

print('REMAINING Missing Data TEST SET')

for cols in null_others:
    print(cols,test[cols].isnull().sum())
    
print('--'*30,'\n','REMAINING Missing Data TRAIN SET')

for cols in null_others:
    print(cols,train[cols].isnull().sum())

for cols in null_others:
    test[cols] = test[cols].mode()[0]

print('--'*40)
print('TEST SET : Missing Data After Imputation')
for cols in null_others:
    print(cols,test[cols].isnull().sum())
#Proposed feature: '1stFlrSF' + '2ndFlrSF' to give us combined Floor Square Footage

try_feature = (train['1stFlrSF'] + train['2ndFlrSF']).copy()
print("Skewness of the original intended feature:",skew(try_feature))
print("Skewness of transformed feature", skew(np.log1p(try_feature)))

# we'll use the transformed feature:)
try_feature = np.log1p(try_feature)
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

# seaborn's regression plot
sns.regplot(x=(try_feature), y=np.log1p(train['SalePrice']), data=train, order=1);

# lets create the feature then
train['1stFlr_2ndFlr_Sf'] = np.log1p(train['1stFlrSF'] + train['2ndFlrSF'])
test['1stFlr_2ndFlr_Sf'] = np.log1p(test['1stFlrSF'] + test['2ndFlrSF'])

#Feature number 2 -> 1stflr+2ndflr+lowqualsf+GrLivArea = All_Liv_Area
try_feature = (train['1stFlr_2ndFlr_Sf'] + train['LowQualFinSF'] + train['GrLivArea']).copy()
print("Skewness of the original intended feature:",skew(try_feature))
print("Skewness of transformed feature", skew(np.log1p(try_feature)))

# hence, we'll use the transformed feature
try_feature = np.log1p(try_feature)
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

# seaborn's regression plot 
sns.regplot(x=(try_feature), y=np.log1p(train['SalePrice']), data=train, order=1);


train['All_Liv_SF'] = np.log1p(train['1stFlr_2ndFlr_Sf'] + train['LowQualFinSF'] + train['GrLivArea'])
test['All_Liv_SF'] = np.log1p(test['1stFlr_2ndFlr_Sf'] + test['LowQualFinSF'] + test['GrLivArea'])
# get all features except Id and SalePrice
feats = train.columns.difference(['Id','SalePrice'])

# the most hassle free way of working with data is to concatenate them
data_combo = pd.concat((train.loc[:,feats],
                      test.loc[:,feats]))

# But first, we log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])


numeric_feats = data_combo.dtypes[data_combo.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

data_combo[skewed_feats] = np.log1p(data_combo[skewed_feats])

# getting dummies for all features. 

data_combo = pd.get_dummies(data_combo)

print(data_combo.shape)
# creating matrices for sklearn:

X_train = data_combo[:train.shape[0]]
X_test = data_combo[train.shape[0]:]
y = train.SalePrice
#But first, Let's devise a cross-validation methodology once and for all

from sklearn.cross_validation import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv=5))
    return(rmse)

# first import library
from sklearn.linear_model import LassoCV

#now create our object
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005], selection='random', max_iter=15000).fit(X_train, y)
res = rmse_cv(model_lasso)
print("Mean:",res.mean())
print("Min: ",res.min())


coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

# plotting feature importances!
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")

