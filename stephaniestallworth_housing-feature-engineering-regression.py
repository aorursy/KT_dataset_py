# Data Pre-Processing

# Imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import sklearn

import seaborn as sns

import matplotlib.mlab as mlab

import warnings

warnings.filterwarnings('ignore')



# Read data

train_data = pd.read_csv('../input/train.csv')



# Settings

pd.set_option('display.height', 1000)

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', None)

pd.set_option('display.width', 2000)



# Data shape

print('Data Shape',train_data.shape)

print(train_data.info())

train_data.head(5)
# Missing Value Count Function

def show_missing():

    missing = train_data.columns[train_data.isnull().any()].tolist()

    return missing



# Missing data counts and percentage

print('Missing Data Count')

print(train_data[show_missing()].isnull().sum().sort_values(ascending = False))

print('--'*40)

print('Missing Data Percentage')

print(round(train_data[show_missing()].isnull().sum().sort_values(ascending = False)/len(train_data)*100,2))
# Functions to address missing data



# Explore features

def feat_explore(column):

    return train_data[column].value_counts()



# Function to impute missing values

def feat_impute(column, value):

    train_data.loc[train_data[column].isnull(),column] = value
# Features with over 50% of its observations missings will be removed

train_data = train_data.drop(['PoolQC','MiscFeature','Alley','Fence'],axis = 1)
# FireplaceQu missing data

print('FireplaceQu Missing Before:', train_data['FireplaceQu'].isnull().sum())

print('--'*40)



# The null values may be homes that do not have fireplaces at all. Need to check this assumption

print(train_data[train_data['FireplaceQu'].isnull()][['Fireplaces','FireplaceQu']])

print(train_data[train_data['FireplaceQu'].isnull()][['Fireplaces','FireplaceQu']].shape)

print('--'*40)



# Impute the nulls with None 

train_data['FireplaceQu'] = train_data['FireplaceQu'].fillna('None')

print('FireplaceQu Missing After:', train_data['FireplaceQu'].isnull().sum())



print('--'*40)

# Cross check columns

print('Confirm Imputation')

print(pd.crosstab(train_data.FireplaceQu,train_data.Fireplaces,))
# Lot Frontage

print('LotFrontage Missing Before:', train_data['LotFrontage'].isnull().sum())



# Check to see if there is a strong correlation with other variables we can use to impute

corr_lf = train_data.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()

cor_dict_lf = corr_lf['LotFrontage'].to_dict()

del cor_dict_lf['LotFrontage']

print("Numeric features by Correlation with LotFrontage:\n")

for ele in sorted(cor_dict_lf.items(), key = lambda x: -abs(x[1])):

    print("{0}: \t{1}".format(*ele))



# Nothing highly correlated to LotFrontage so will impute with the mean

train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].median())

print('LotFrontage Missing After:', train_data['LotFrontage'].isnull().sum())
# Garage Features

print('Garage Features Missing Before')

print(train_data[['GarageYrBlt', 'GarageType', 'GarageFinish','GarageQual','GarageCond']].isnull().sum())



# Assumptions check

print('--'*40)

print('Assumption Check')

null_garage = ['GarageYrBlt','GarageType','GarageQual','GarageCond','GarageFinish']

print(train_data[(train_data['GarageYrBlt'].isnull())|

                 (train_data['GarageType'].isnull())|

                 (train_data['GarageQual'].isnull())|

                 (train_data['GarageCond'].isnull())|

                 (train_data['GarageFinish'].isnull())]

                 [['GarageCars','GarageYrBlt','GarageType','GarageQual','GarageCond','GarageFinish']])



# Impute null garage features

for cols in null_garage:

   if train_data[cols].dtype ==np.object:

         feat_impute(cols, 'None')

   else:

         feat_impute(cols, 0)

        

# Garage Features After

print('--'*40)

print('Garage Features Missing After')

print(train_data[['GarageYrBlt', 'GarageType', 'GarageFinish','GarageQual','GarageCond']].isnull().sum())

print('--'*40)

# Cross check columns

print('Confirm Imputation')

for cols in null_garage:

    print(pd.crosstab(train_data[cols],train_data.GarageCars))
# Basement Features

print('Basement Features Missing Before')

print(train_data[['BsmtFinType2', 'BsmtExposure']].isnull().sum())

print('--'*40)



# BsmtFinType2 and BsmtExposure are both missing 38 observations

# Check that data is missing in the same rows

# Confirm if all nulls correspond to homes without basements

print('Assumption Check')

null_basement = ['BsmtFinType2','BsmtExposure']

print(train_data[train_data['BsmtFinType2'].isnull()|(train_data['BsmtExposure'].isnull())][['TotalBsmtSF','BsmtFinType2','BsmtExposure']])

print('entries',train_data[train_data['BsmtFinType2'].isnull()|(train_data['BsmtExposure'].isnull())][['TotalBsmtSF','BsmtFinType2','BsmtExposure']].shape)
# Impute the only null BsmtFinType2 with a basement at index 332 with most frequent value

train_data.iloc[332, train_data.columns.get_loc('BsmtFinType2')] = train_data['BsmtFinType2'].mode()[0]

#train_data.set_value(332,'BsmtFinType2',train_data['BsmtFinType2'].mode()[0])



# Impute the only null BsmtExposure with a basement at index 948 with most frequent value

train_data.iloc[948, train_data.columns.get_loc('BsmtExposure')] = train_data['BsmtExposure'].mode()[0]



# Impute the remaining nulls as None

for cols in null_basement:

   if train_data[cols].dtype ==np.object:

         feat_impute(cols, 'None')

   else:

         feat_impute(cols, 0)



# Basement Features After

print('--'*40)

print('Basement Features Missing After')

print(train_data[['BsmtFinType2', 'BsmtExposure']].isnull().sum())

print('--'*40)

# Cross check columns

print('Confirm Imputation')

for cols in null_basement:

    print(pd.crosstab(train_data.TotalBsmtSF,train_data[cols]))
# Basement Features Part 2

print('Basement Features Part 2 Missing Before')

print(train_data[['BsmtFinType1', 'BsmtCond','BsmtQual']].isnull().sum())

print('--'*40)



# Check assumptions

null_basement2 = ['BsmtFinType1', 'BsmtCond','BsmtQual']

print('Assumption Check')

print(train_data[(train_data['BsmtFinType1'].isnull())|

                 (train_data['BsmtCond'].isnull())|

                 (train_data['BsmtQual'].isnull())]

                 [['TotalBsmtSF','BsmtFinType1', 'BsmtCond','BsmtQual']])

print('entries',train_data[(train_data['BsmtFinType1'].isnull())|

                 (train_data['BsmtCond'].isnull())|

                 (train_data['BsmtQual'].isnull())]

                 [['TotalBsmtSF','BsmtFinType1', 'BsmtCond','BsmtQual']].shape)



# NA in all. NA means No basement

# for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

#   features[col] = features[col].fillna('None')



# Impute nulls to None or 0

for cols in null_basement2:

    if train_data[cols].dtype ==np.object:

        cols = feat_impute(cols, 'None')

    else:

        cols = feat_impute(cols, 0)    



print('--'*40)

print('Basement Features Part 2 Missing After')

print(train_data[['BsmtFinType1', 'BsmtCond','BsmtQual']].isnull().sum())

# MasVnrArea and MasVnrType are each missing 8 observations

print('Masonry Features Missing Before')

print(train_data[['MasVnrArea', 'MasVnrType']].isnull().sum())

print('--'*40)



# Confirm that the missing values in these columns are the same rows

print('Check Assumptions')

print(train_data[(train_data['MasVnrArea'].isnull())|

                 (train_data['MasVnrType'].isnull())]

                 [['MasVnrArea','MasVnrType']])



print(train_data[(train_data['MasVnrArea'].isnull())|

                 (train_data['MasVnrType'].isnull())]

                 [['MasVnrArea','MasVnrType']].shape)



# Impute MasVnrArea with the most frequent values

# feat_explore('MasVnrArea')

# feat_impute('MasVnrArea','None')

train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(train_data['MasVnrArea'].mode()[0])



# Impute MasVnrType with the most frequent values

# feat_explore('MasVnrType')

# feat_impute('MasVnrType',0.0)

train_data['MasVnrType'] = train_data['MasVnrType'].fillna(train_data['MasVnrType'].mode()[0])



print('Masonry Features Missing After')

print(train_data[['MasVnrArea', 'MasVnrType']].isnull().sum())

print('--'*40)



print('Confirm Imputation')

print(train_data[(train_data['MasVnrArea'].isnull())|

                 (train_data['MasVnrType'].isnull())]

                 [['MasVnrArea','MasVnrType']])
# Electrical is only missing one value

print('Electrical Feature Missing Before')

print(train_data[['Electrical']].isnull().sum())

print('--'*40)



# Impute Electrical with the most frequent value, 'SBrkr'

# feat_explore('Electrical')

# feat_impute('Electrical','SBrkr')

train_data['Electrical'] = train_data['Electrical'].fillna(train_data['Electrical'].mode()[0])

print('Electrical Feature Missing After')

print(train_data[['Electrical']].isnull().sum())

print('--'*40)
# Confirm all changes

print('Missing Data Count')

print(train_data[show_missing()].isnull().sum().sort_values(ascending = False))

print('No Missing Values')
train_data.info()
# Data Types

# Categorical Features

print('Categorical Features:\n ', train_data.select_dtypes(include=['object']).columns)

print('--'*40)



# Numeric Features

print('Numeric Features:\n ', train_data.select_dtypes(exclude=['object']).columns)
catcols = train_data.select_dtypes(['object'])

for cat in catcols:

    print('--'*40)

    print(cat)

    print(train_data[cat].value_counts())
# Encode ordinal data

train_data['LotShape'] = train_data['LotShape'].map({'Reg':0,'IR1':1,'IR2':2,'IR3':3})

train_data['LandContour'] = train_data['LandContour'].map({'Low':0,'HLS':1,'Bnk':2,'Lvl':3})

train_data['Utilities'] = train_data['Utilities'].map({'NoSeWa':0,'NoSeWa':1,'AllPub':2})

train_data['BldgType'] = train_data['BldgType'].map({'Twnhs':0,'TwnhsE':1,'Duplex':2,'2fmCon':3,'1Fam':4})

train_data['HouseStyle'] = train_data['HouseStyle'].map({'1Story':0,'1.5Fin':1,'1.5Unf':2,'2Story':3,'2.5Fin':4,'2.5Unf':5,'SFoyer':6,'SLvl':7})

train_data['BsmtFinType1'] = train_data['BsmtFinType1'].map({'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})

train_data['BsmtFinType2'] = train_data['BsmtFinType2'].map({'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})

train_data['LandSlope'] = train_data['LandSlope'].map({'Gtl':0,'Mod':1,'Sev':2})

train_data['Street'] = train_data['Street'].map({'Grvl':0,'Pave':1})

train_data['MasVnrType'] = train_data['MasVnrType'].map({'None':0,'BrkCmn':1,'BrkFace':2,'CBlock':3,'Stone':4})

train_data['CentralAir'] = train_data['CentralAir'].map({'N':0,'Y':1})

train_data['GarageFinish'] = train_data['GarageFinish'].map({'None':0,'Unf':1,'RFn':2,'Fin':3})

train_data['PavedDrive'] = train_data['PavedDrive'].map({'N':0,'P':1,'Y':2})

train_data['BsmtExposure'] = train_data['BsmtExposure'].map({'None':0,'No':1,'Mn':2,'Av':3,'Gd':4})

train_data['ExterQual'] = train_data['ExterQual'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

train_data['ExterCond'] = train_data['ExterCond'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

train_data['BsmtCond'] = train_data['BsmtCond'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

train_data['BsmtQual'] = train_data['BsmtQual'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

train_data['HeatingQC'] = train_data['HeatingQC'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

train_data['KitchenQual'] = train_data['KitchenQual'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

train_data['FireplaceQu'] = train_data['FireplaceQu'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

train_data['GarageQual'] = train_data['GarageQual'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

train_data['GarageCond'] = train_data['GarageCond'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})



# Encode Categorical Variables

train_data['Foundation'] = train_data['Foundation'].map({'BrkTil':0,'CBlock':1,'PConc':2,'Slab':3,'Stone':4,'Wood':5})

train_data['Heating'] = train_data['Heating'].map({'Floor':0,'GasA':1,'GasW':2,'Grav':3,'OthW':4,'Wall':5})

train_data['Electrical'] = train_data['Electrical'].map({'SBrkr':0,'FuseA':1,'FuseF':2,'FuseP':3,'Mix':4})

train_data['Functional'] = train_data['Functional'].map({'Sal':0,'Sev':1,'Maj2':2,'Maj1':3,'Mod':4,'Min2':5,'Min1':6,'Typ':7})

train_data['GarageType'] = train_data['GarageType'].map({'None':0,'Detchd':1,'CarPort':2,'BuiltIn':3,'Basment':4,'Attchd':5,'2Types':6})

train_data['SaleType'] = train_data['SaleType'].map({'Oth':0,'ConLD':1,'ConLI':2,'ConLw':3,'Con':4,'COD':5,'New':6,'VWD':7,'CWD':8,'WD':9})

train_data['SaleCondition'] = train_data['SaleCondition'].map({'Partial':0,'Family':1,'Alloca':2,'AdjLand':3,'Abnorml':4,'Normal':5})

train_data['MSZoning'] = train_data['MSZoning'].map({'A':0,'FV':1,'RL':2,'RP':3,'RM':4,'RH':5,'C (all)':6,'I':7})

train_data['LotConfig'] = train_data['LotConfig'].map({'Inside':0,'Corner':1,'CulDSac':2,'FR2':3,'FR3':4})

train_data['Neighborhood'] = train_data['Neighborhood'].map({'Blmngtn':0,'Blueste':1,'BrDale':2,'BrkSide':3, 'ClearCr':4,'CollgCr':5,'Crawfor':6,'Edwards':7,'Gilbert':8,

                                                             'IDOTRR':9,'MeadowV':10,'Mitchel':11, 'NAmes':12,'NoRidge':13,'NPkVill':14,'NridgHt':15, 'NWAmes':16,

                                                             'OldTown':17,'SWISU':18,'Sawyer':19, 'SawyerW':20,'Somerst':21,'StoneBr':22,'Timber':23,'Veenker':24})

train_data['Condition1'] = train_data['Condition1'].map({'Artery':0,'Feedr':1,'Norm':2,'RRNn':3, 'RRAn':4,'PosN':5,'PosA':6,'RRNe':7,'RRAe':8})

train_data['Condition2'] = train_data['Condition2'].map({'Artery':0,'Feedr':1,'Norm':2,'RRNn':3, 'RRAn':4,'PosN':5,'PosA':6,'RRNe':7,'RRAe':8})

train_data['RoofStyle'] = train_data['RoofStyle'].map({'Flat':0,'Gable':1,'Gambrel':2,'Hip':3,'Mansard':4,'Shed':5})

train_data['RoofMatl'] = train_data['RoofMatl'].map({'ClyTile':0,'CompShg':1,'Membran':2,'Metal':3,'Roll':4,'Tar&Grv':5,'WdShake':6,'WdShngl':7})

train_data['Exterior1st'] = train_data['Exterior1st'].map({'AsbShng':0,'AsphShn':1,'BrkComm':2,'BrkFace':3,'CBlock':4,'CemntBd':5,'HdBoard':6,'ImStucc':7,'MetalSd':8,

                                                           'Other':9,'Plywood':10,'PreCast':11,'Stone':12,'Stucco':13,'VinylSd':14,'Wd Sdng':15,'WdShing':16})

train_data['Exterior2nd'] = train_data['Exterior2nd'].map({'AsbShng':0,'AsphShn':1,'Brk Cmn':2,'BrkFace':3,'CBlock':4,'CmentBd':5,'HdBoard':6,'ImStucc':7,'MetalSd':8,

                                                           'Other':9,'Plywood':10,'PreCast':11,'Stone':12,'Stucco':13,'VinylSd':14,'Wd Sdng':15,'Wd Shng':16})  



# Confirm encoding

pd.options.display.float_format = '{:.2f}'.format

np.set_printoptions(suppress = False)

train_data.describe().transpose()
# Statistical Summary

print("SalePrice Statistical Summary:\n")

print(train_data['SalePrice'].describe())

print("Median Sale Price:", train_data['SalePrice'].median(axis = 0))

print('Skewness:',train_data['SalePrice'].skew())

skew = train_data['SalePrice'].skew()



# mean distribution

mu = train_data['SalePrice'].mean()



# std distribution

sigma = train_data['SalePrice'].std()

num_bins = 40



# Histogram of SalesPrice

plt.figure(figsize=(11, 6))

n, bins, patches = plt.hist(train_data['SalePrice'], num_bins, normed=1,edgecolor = 'black', lw = 1, alpha = .40)



# Normal Distribution

y = mlab.normpdf(bins, mu, sigma)

plt.plot(bins, y, 'r--', linewidth=2)

plt.xlabel('Sale Price')

plt.ylabel('Probability density')



plt.title(r'$\mathrm{Histogram\ of\ SalePrice:}\ \mu=%.3f,\ \sigma=%.3f$'%(mu,sigma))

plt.grid(True)

#fig.tight_layout()

plt.show()
# Normalize SalePrice using log-transformation

sale_price_norm = np.log1p(train_data['SalePrice'])



# Mean distribution

mu = sale_price_norm.mean()



# Standard distribution

sigma = sale_price_norm.std()

num_bins = 40

plt.figure(figsize=(11, 6))

n, bins, patches = plt.hist(sale_price_norm, num_bins, normed=1, edgecolor = 'black', lw = 1,alpha = .40)



y = mlab.normpdf(bins, mu, sigma)

plt.plot(bins, y, 'r--', linewidth=2)

plt.xlabel('Sale Price')

plt.ylabel('Probability density')



plt.title(r'$\mathrm{Histogram\ of\ SalePrice:}\ \mu=%.3f,\ \sigma=%.3f$'%(mu,sigma))

plt.grid(True)

#fig.tight_layout()

plt.show()
# Lasso Regression



# Split

# Create matrix of all x features

X = train_data.drop(['SalePrice'], axis = 1)



# Create array of target variable

y = train_data['SalePrice']



# Split training data into train and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .20,random_state = 101)



# Fit

# Import model

from sklearn.linear_model import Lasso



# Instantiate object

lasso = Lasso()



# Fit model to training data

lasso = lasso.fit(X_train, y_train)



# Predict

y_pred_lasso = lasso.predict(X_test)



# Score It

from sklearn import metrics

print('Linear Regression Performance')

print('MAE',metrics.mean_absolute_error(y_test, y_pred_lasso))

print('MSE',metrics.mean_squared_error(y_test, y_pred_lasso))

print('RMSE',np.sqrt(metrics.mean_squared_error(y_test, y_pred_lasso)))

print('R^2 =',metrics.explained_variance_score(y_test,y_pred_lasso))



# Lasso Coefficients

pd.set_option('display.float_format', lambda x: '%.2f' % x)

cdf = pd.DataFrame(data = lasso.coef_,index = X_train.columns, columns = ['Lasso Coefficients'])

# **RANDOM FOREST**

cdf.sort_values(by = 'Lasso Coefficients', ascending = False)
# Select top 20% of features



# Create matrix of x features

X = train_data.drop(['SalePrice'],axis = 1)



# Create array of target variable y

y =train_data['SalePrice']



# Feature Selector

# Import

from sklearn.feature_selection import SelectPercentile, f_regression



# Instantiate object

selector_f = SelectPercentile(f_regression, percentile=20)



# Fit and transform

x_best = selector_f.fit_transform(X, y)
support = np.asarray(selector_f.get_support())



# Supress displaying long numbers in scientific notation

#pd.set_option('display.float_format', lambda x: '%.4f' % x)



# Enable scientific notation

pd.set_option('display.float_format', '{:.2e}'.format)



# Column names of top 20%

features = np.asarray(X.columns.values)

features_with_support = features[support]

# print('Top 20% of the best, associated features to SalePrice\n',columns_with_support)

# print('Number of Features:', len(columns_with_support))



#f-scores of top 20%

fscores = np.asarray(selector_f.scores_)

fscores_with_support = fscores[support]



# p-values of top 20%

pvalues = np.asarray(selector_f.pvalues_)

pvalues_with_support = pvalues[support]



# Dataframe of top 20%

top20 = pd.DataFrame({'F-score':fscores_with_support,

                      'p-value':pvalues_with_support},

                     index = features_with_support)

# top20.index.name = 'Feature'

print('Top 20% best associated features to SalePrice\nNumber of features:',len(features_with_support))

print(top20.sort_values(by = 'p-value', ascending = 'True'))



#Print All Selector_f.scores_

# for n,s in zip(train_data.columns,Selector_f.scores_):

#      print('F-score: %3.2ft for feature %s ' % (s,n))



# Dataframe of all f-scores

# fscores = pd.DataFrame(selector_f.scores_,X.columns,['F-score'])

# fscores.sort_values(by = 'F-score', ascending = False)



# Dataframe of all p-values

# pscores = pd.DataFrame(selector_f.pvalues_,X.columns, ['P_Value'])

# pscores.sort_values(by = 'P_Value', ascending = True)
# # Correlations to SalePrice

# corr = train_data.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()

# cor_dict = corr['SalePrice'].to_dict()

# del cor_dict['SalePrice']

# print("List the numerical features in decending order by their correlation with Sale Price:\n")

# for ele in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):

#     print("{0}: \t{1}".format(*ele))

    

# #Correlation matrix heatmap

# corrmat = train_data.corr()

# plt.figure(figsize=(30, 20))



# #number of variables for heatmap

# k = 76

# cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

# cm = np.corrcoef(train_data[cols].values.T)



# #generate mask for upper triangle

# mask = np.zeros_like(cm, dtype=np.bool)

# mask[np.triu_indices_from(mask)] = True



# sns.set(font_scale=.80)

# sns.heatmap(cm, mask=mask, cbar=True, annot=True, square=True,\

#                  fmt='.2f',annot_kws={'size': 7}, yticklabels=cols.values,\

#                  xticklabels=cols.values, cmap = 'coolwarm',lw = .1)

# plt.show() 



# Feature-to-Feature Correlation

# corr = train_data.drop('SalePrice', axis=1).corr() # We already examined SalePrice correlations

# plt.figure(figsize=(12, 10))



# sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 

#             cmap='RdYlGn', vmax=1.0, vmin=-1.0, linewidths=0.1,

#             annot=True, annot_kws={"size": 8}, square=True);
best_feat = train_data[features_with_support]

corr =best_feat.corr() # We already examined SalePrice correlations

plt.figure(figsize=(12, 10))



sns.heatmap(corr[(corr >= 0.7) | (corr <= -0.7)], 

            cmap='coolwarm', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);
# Correlations to SalePrice

from scipy import stats

print('Correlation to SalePrice')

print('GrLivArea:',stats.pearsonr(best_feat['GrLivArea'],train_data['SalePrice'])[0])

print('TotRmsAbvGrd:',stats.pearsonr(best_feat['TotRmsAbvGrd'],train_data['SalePrice'])[0])

print('--'*40)

print('GarageCars:',stats.pearsonr(best_feat['GarageCars'],train_data['SalePrice'])[0])

print('GarageArea:',stats.pearsonr(best_feat['GarageArea'],train_data['SalePrice'])[0])
# Remove redundant features

best_feat = best_feat.drop(['TotRmsAbvGrd','GarageArea'], axis = 1)

best_feat.columns
# Random Forest Regression with Best Features

# Split

# Create matrix of best x features

X_best = train_data[['OverallQual', 'YearBuilt', 'YearRemodAdd', 'ExterQual', 'BsmtQual', 'TotalBsmtSF', 

               '1stFlrSF', 'GrLivArea', 'FullBath', 'KitchenQual', 'TotRmsAbvGrd', 'FireplaceQu', 

               'GarageFinish', 'GarageCars', 'GarageArea']]



# Create array of target variable

y = train_data['SalePrice']



# Split training data into train and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_best,y, test_size = .20,random_state = 101)



# Fit

from sklearn.ensemble import RandomForestRegressor

rforest = RandomForestRegressor(n_estimators = 300, random_state = 0) 

rforest.fit(X_best,y)



# Predict

y_pred_rforest = rforest.predict(X_test)



# Score It

from sklearn import metrics

print('Random Forest Regression Performance')

print('MAE',metrics.mean_absolute_error(y_test, y_pred_rforest))

print('MSE',metrics.mean_squared_error(y_test, y_pred_rforest))

print('RMSE',np.sqrt(metrics.mean_squared_error(y_test, y_pred_rforest)))

print('R^2 =',metrics.explained_variance_score(y_test,y_pred_rforest))
# Load test data

test_data = pd.read_csv('../input/test.csv')



# Test data info

test_data.info()



# Test data shape

print('shape',test_data.shape)
# Missing Value Count Function

def show_missing():

    missing = test_data.columns[test_data.isnull().any()].tolist()

    return missing



# Missing data counts and percentage

pd.set_option('display.float_format', lambda x: '%.4f' % x)

print('Missing Data Count')

print(test_data[show_missing()].isnull().sum().sort_values(ascending = False))

print('--'*40)

print('Missing Data Percentage')

print(round(test_data[show_missing()].isnull().sum().sort_values(ascending = False)/len(test_data)*100,2))
# Function to impute missing values

def feat_impute(column, value):

    test_data.loc[test_data[column].isnull(),column] = value
# Features with over 50% of its observations missings will be removed

test_data = test_data.drop(['PoolQC','MiscFeature','Alley','Fence'],axis = 1)
# FireplaceQu missing data

print('FireplaceQu Missing Before:', test_data['FireplaceQu'].isnull().sum())

print('--'*40)



# The null values may be homes that do not have fireplaces at all. Need to check this assumption

print(test_data[test_data['FireplaceQu'].isnull()][['Fireplaces','FireplaceQu']])

print(test_data[test_data['FireplaceQu'].isnull()][['Fireplaces','FireplaceQu']].shape)

print('--'*40)



# Impute the nulls with None 

test_data['FireplaceQu'] = test_data['FireplaceQu'].fillna('None')

print('FireplaceQu Missing After:', test_data['FireplaceQu'].isnull().sum())



print('--'*40)

# Cross check columns

print('Confirm Imputation')

print(pd.crosstab(test_data.FireplaceQu,test_data.Fireplaces))
# LotFrontage nulls

print('LotFrontage Missing Before:', test_data['LotFrontage'].isnull().sum())



# Impute with mean

test_data['LotFrontage'] = test_data['LotFrontage'].fillna(test_data['LotFrontage'].median())

print('LotFrontage Missing After:', test_data['LotFrontage'].isnull().sum())
# Garage features null

print('Garage Features Missing Before')

print(test_data[['GarageYrBlt','GarageCond','GarageQual', 'GarageFinish',

                 'GarageType','GarageCars','GarageArea']].isnull().sum())



# The null values may be homes that do not have Garages at all.

# Need to check this assumption and inpute accordingly

print('Assumption Check')

print(test_data[(test_data['GarageYrBlt'].isnull())|

                 (test_data['GarageCond'].isnull())|

                 (test_data['GarageQual'].isnull())|

                (test_data['GarageFinish'].isnull())|

                (test_data['GarageType'].isnull())|

                (test_data['GarageCars'].isnull())|

                (test_data['GarageArea'].isnull())]

                 [['GarageYrBlt','GarageCond','GarageQual', 'GarageFinish',

                 'GarageType','GarageCars','GarageArea']])



# Most of the nulls are associated with homes without a garage.  

# However, there are exceptions that must be addressed before we can inpute the remaining nulls with 'None'

# Inpute nulls at index 666 that have a garage with most common value in each column for categorical variables 

test_data.iloc[666, test_data.columns.get_loc('GarageYrBlt')] = test_data['GarageYrBlt'].mode()[0]

test_data.iloc[666, test_data.columns.get_loc('GarageCond')] = test_data['GarageCond'].mode()[0]

test_data.iloc[666, test_data.columns.get_loc('GarageFinish')] = test_data['GarageFinish'].mode()[0]

test_data.iloc[666, test_data.columns.get_loc('GarageQual')] = test_data['GarageQual'].mode()[0]

test_data.iloc[666, test_data.columns.get_loc('GarageType')] = test_data['GarageType'].mode()[0]



# Inpute nulls at index 1116 that have a garage with most common value in each column for categorical variables 

test_data.iloc[1116, test_data.columns.get_loc('GarageYrBlt')] = test_data['GarageYrBlt'].mode()[0]

test_data.iloc[1116, test_data.columns.get_loc('GarageCond')] = test_data['GarageCond'].mode()[0]

test_data.iloc[1116, test_data.columns.get_loc('GarageFinish')] = test_data['GarageFinish'].mode()[0]

test_data.iloc[1116, test_data.columns.get_loc('GarageQual')] = test_data['GarageQual'].mode()[0]

test_data.iloc[1116, test_data.columns.get_loc('GarageType')] = test_data['GarageType'].mode()[0]



# Inpute nulls at index 1116 that have a garage with median value in each column for continuous variables 

test_data.iloc[1116, test_data.columns.get_loc('GarageCars')] = test_data['GarageCars'].median()

test_data.iloc[1116, test_data.columns.get_loc('GarageArea')] = test_data['GarageArea'].median()



# Impute the remaining nulls as None

null_garage = ['GarageYrBlt','GarageCond','GarageFinish','GarageQual', 

                 'GarageType','GarageCars','GarageArea']



for cols in null_garage:

   if test_data[cols].dtype ==np.object:

         feat_impute(cols, 'None')

   else:

         feat_impute(cols, 0)



# Basement Features After

print('--'*40)

print('Garage Features Missing After')

print(test_data[['GarageYrBlt','GarageCond','GarageQual', 'GarageFinish',

                 'GarageType','GarageCars','GarageArea']].isnull().sum())



print('--'*40)

# Cross check columns

# print('Confirm Imputation')

# for cols in null_basement:

#     print(pd.crosstab(test_data.TotalBsmtSF,test_data[cols]))
null_basement1 = ['BsmtCond','BsmtExposure','BsmtQual','BsmtFinType1','BsmtFinType2']



print('Basement Features Part 1 Missing Before')

for cols in null_basement:

    print (cols ,test_data[cols].isnull().sum())

 

# The null values may be homes that do not have Garages at all.

# Need to check this assumption against BsmtFinSF1 and inpute accordingly

print('Assumption Check')

print(test_data[(test_data['BsmtCond'].isnull())|(test_data['BsmtExposure'].isnull())|

                 (test_data['BsmtQual'].isnull())| (test_data['BsmtFinType1'].isnull())|

                (test_data['BsmtFinType2'].isnull())]

                 [['BsmtCond','BsmtExposure','BsmtQual','BsmtFinType1','BsmtFinType2']])



print(test_data[(test_data['BsmtCond'].isnull())|(test_data['BsmtExposure'].isnull())|

                 (test_data['BsmtQual'].isnull())| (test_data['BsmtFinType1'].isnull())|

                (test_data['BsmtFinType2'].isnull())]

                 [['BsmtCond','BsmtExposure','BsmtQual','BsmtFinType1','BsmtFinType2']].shape)



# Most of the nulls are associated with homes without a basement.  

# However, there are exceptions that must be addressed before we can inpute the remaining nulls with 'None'

# Inpute nulls of BasmtExposure that have a basement with most common value 

test_data.iloc[27, test_data.columns.get_loc('BsmtExposure')] = test_data['BsmtExposure'].mode()[0]

test_data.iloc[888, test_data.columns.get_loc('BsmtExposure')] = test_data['BsmtExposure'].mode()[0]



# Inpute nulls of BsmtCond that have a basement with most common value 

test_data.iloc[540, test_data.columns.get_loc('BsmtCond')] = test_data['BsmtCond'].mode()[0]

test_data.iloc[580, test_data.columns.get_loc('BsmtCond')] = test_data['BsmtCond'].mode()[0]

test_data.iloc[725, test_data.columns.get_loc('BsmtCond')] = test_data['BsmtCond'].mode()[0]

test_data.iloc[1064, test_data.columns.get_loc('BsmtCond')] = test_data['BsmtCond'].mode()[0]

test_data.iloc[1064, test_data.columns.get_loc('BsmtCond')] = test_data['BsmtCond'].mode()[0]



# Inpute nulls in BsmetQualthat have a basement with most common value

test_data.iloc[757, test_data.columns.get_loc('BsmtQual')] = test_data['BsmtQual'].mode()[0]

test_data.iloc[758, test_data.columns.get_loc('BsmtQual')] = test_data['BsmtQual'].mode()[0]



# Inpute nulls in basement features with 'None' for categorical variables or zero for numeric variables

for cols in null_basement1:

   if test_data[cols].dtype ==np.object:

         feat_impute(cols, 'None')

   else:

         feat_impute(cols, 0)

        

print('Basement Features Part 1 Missing After')

for cols in null_basement:

    print (cols ,test_data[cols].isnull().sum())
null_basement2= ['BsmtFullBath','BsmtHalfBath','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']  



# Need to check that nulls are homes without basements 

print('Assumption Check')

print(test_data[ (test_data['BsmtFullBath'].isnull())|(test_data['BsmtHalfBath'].isnull())|

                (test_data['BsmtFinSF1'].isnull())|(test_data['BsmtFinSF2'].isnull())|

                 (test_data['BsmtUnfSF'].isnull())|(test_data['TotalBsmtSF'].isnull())]

                 [['BsmtFullBath','BsmtHalfBath','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']])



# Since assumption has been confirmed, impute accordingly

for cols in null_basement2:

   if test_data[cols].dtype ==np.object:

         feat_impute(cols, 'None')

   else:

         feat_impute(cols, 0)



# Basement Features After

print('--'*40)

print('Basement Features Part 1 Missing After')

for cols in null_basement2:

    print (cols ,test_data[cols].isnull().sum())
null_masonry = ['MasVnrType','MasVnrArea']

print('Missing Data Before')

for cols in null_masonry:

    print(cols,test_data[cols].isnull().sum())

    

# View nulls in masonry features

print('--'*40,'\nAssumption Check')

print(test_data[(test_data['MasVnrType'].isnull())|(test_data['MasVnrType'].isnull())|

                (test_data['MasVnrArea'].isnull())|(test_data['MasVnrArea'].isnull())]

                 [['MasVnrType','MasVnrArea']])



# Impute exceptions to assumption that nulls correspond to homes with no exposure

test_data.iloc[1150, test_data.columns.get_loc('MasVnrType')] = test_data['MasVnrType'].mode()[0]



# Impute the remaining nulls with 'None' or zero

for cols in null_masonry:

   if test_data[cols].dtype ==np.object:

         feat_impute(cols, 'None')

   else:

         feat_impute(cols, 0)

        

print('--'*40,'\nMissing Data After')

for cols in null_masonry:

    print(cols,test_data[cols].isnull().sum())
# Impute other categorical features with most frequent value

null_others = ['MSZoning', 'Utilities','Functional','Exterior2nd','Exterior1st','SaleType','KitchenQual'] 



print('Missing Data Before')

for cols in null_others:

    print(cols,test_data[cols].isnull().sum())



# Impute with most common value

for cols in null_others:

    test_data[cols] = test_data[cols].mode()[0]



print('--'*40)

print('Missing Data After')

for cols in null_others:

    print(cols,test_data[cols].isnull().sum())
# LotFrontage nulls

print('LotFrontage Missing Before:', test_data['LotFrontage'].isnull().sum())



# Impute with mean

test_data['LotFrontage'] = test_data['LotFrontage'].fillna(test_data['LotFrontage'].median())

print('LotFrontage Missing After:', test_data['LotFrontage'].isnull().sum())
# Confirm Imputations in test data

test_data.info()
# Encode ordinal data

test_data['LandContour'] = test_data['LandContour'].map({'Low':0,'HLS':1,'Bnk':2,'Lvl':3})

test_data['Utilities'] = test_data['Utilities'].map({'NoSeWa':0,'NoSeWa':1,'AllPub':2})

test_data['BldgType'] = test_data['BldgType'].map({'Twnhs':0,'TwnhsE':1,'Duplex':2,'2fmCon':3,'1Fam':4})

test_data['HouseStyle'] = test_data['HouseStyle'].map({'1Story':0,'1.5Fin':1,'1.5Unf':2,'2Story':3,'2.5Fin':4,'2.5Unf':5,'SFoyer':6,'SLvl':7})

test_data['BsmtFinType1'] = test_data['BsmtFinType1'].map({'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})

test_data['BsmtFinType2'] = test_data['BsmtFinType2'].map({'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})

test_data['LandSlope'] = test_data['LandSlope'].map({'Gtl':0,'Mod':1,'Sev':2})

test_data['Street'] = test_data['Street'].map({'Grvl':0,'Pave':1})

test_data['MasVnrType'] = test_data['MasVnrType'].map({'None':0,'BrkCmn':1,'BrkFace':2,'CBlock':3,'Stone':4})

test_data['CentralAir'] = test_data['CentralAir'].map({'N':0,'Y':1})

test_data['GarageFinish'] = test_data['GarageFinish'].map({'None':0,'Unf':1,'RFn':2,'Fin':3})

test_data['PavedDrive'] = test_data['PavedDrive'].map({'N':0,'P':1,'Y':2})

test_data['BsmtExposure'] = test_data['BsmtExposure'].map({'None':0,'No':1,'Mn':2,'Av':3,'Gd':4})

test_data['ExterQual'] = test_data['ExterQual'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

test_data['ExterCond'] = test_data['ExterCond'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

test_data['BsmtCond'] = test_data['BsmtCond'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

test_data['BsmtQual'] = test_data['BsmtQual'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

test_data['HeatingQC'] = test_data['HeatingQC'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

test_data['KitchenQual'] = test_data['KitchenQual'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

test_data['FireplaceQu'] = test_data['FireplaceQu'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

test_data['GarageQual'] = test_data['GarageQual'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

test_data['GarageCond'] = test_data['GarageCond'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})



# Encode Categorical Variables

test_data['Foundation'] = test_data['Foundation'].map({'BrkTil':0,'CBlock':1,'PConc':2,'Slab':3,'Stone':4,'Wood':5})

test_data['Heating'] = test_data['Heating'].map({'Floor':0,'GasA':1,'GasW':2,'Grav':3,'OthW':4,'Wall':5})

test_data['Electrical'] = test_data['Electrical'].map({'SBrkr':0,'FuseA':1,'FuseF':2,'FuseP':3,'Mix':4})

test_data['Functional'] = test_data['Functional'].map({'Sal':0,'Sev':1,'Maj2':2,'Maj1':3,'Mod':4,'Min2':5,'Min1':6,'Typ':7})

test_data['GarageType'] = test_data['GarageType'].map({'None':0,'Detchd':1,'CarPort':2,'BuiltIn':3,'Basment':4,'Attchd':5,'2Types':6})

test_data['SaleType'] = test_data['SaleType'].map({'Oth':0,'ConLD':1,'ConLI':2,'ConLw':3,'Con':4,'COD':5,'New':6,'VWD':7,'CWD':8,'WD':9})

test_data['SaleCondition'] = test_data['SaleCondition'].map({'Partial':0,'Family':1,'Alloca':2,'AdjLand':3,'Abnorml':4,'Normal':5})

test_data['MSZoning'] = test_data['MSZoning'].map({'A':0,'FV':1,'RL':2,'RP':3,'RM':4,'RH':5,'C (all)':6,'I':7})

test_data['LotConfig'] = test_data['LotConfig'].map({'Inside':0,'Corner':1,'CulDSac':2,'FR2':3,'FR3':4})

test_data['Neighborhood'] = test_data['Neighborhood'].map({'Blmngtn':0,'Blueste':1,'BrDale':2,'BrkSide':3, 'ClearCr':4,'CollgCr':5,'Crawfor':6,'Edwards':7,'Gilbert':8,

                                                             'IDOTRR':9,'MeadowV':10,'Mitchel':11, 'NAmes':12,'NoRidge':13,'NPkVill':14,'NridgHt':15, 'NWAmes':16,

                                                             'OldTown':17,'SWISU':18,'Sawyer':19, 'SawyerW':20,'Somerst':21,'StoneBr':22,'Timber':23,'Veenker':24})

test_data['Condition1'] = test_data['Condition1'].map({'Artery':0,'Feedr':1,'Norm':2,'RRNn':3, 'RRAn':4,'PosN':5,'PosA':6,'RRNe':7,'RRAe':8})

test_data['Condition2'] = test_data['Condition2'].map({'Artery':0,'Feedr':1,'Norm':2,'RRNn':3, 'RRAn':4,'PosN':5,'PosA':6,'RRNe':7,'RRAe':8})

test_data['RoofStyle'] = test_data['RoofStyle'].map({'Flat':0,'Gable':1,'Gambrel':2,'Hip':3,'Mansard':4,'Shed':5})

test_data['RoofMatl'] = test_data['RoofMatl'].map({'ClyTile':0,'CompShg':1,'Membran':2,'Metal':3,'Roll':4,'Tar&Grv':5,'WdShake':6,'WdShngl':7})

test_data['Exterior1st'] = test_data['Exterior1st'].map({'AsbShng':0,'AsphShn':1,'BrkComm':2,'BrkFace':3,'CBlock':4,'CemntBd':5,'HdBoard':6,'ImStucc':7,'MetalSd':8,

                                                           'Other':9,'Plywood':10,'PreCast':11,'Stone':12,'Stucco':13,'VinylSd':14,'Wd Sdng':15,'WdShing':16})

test_data['Exterior2nd'] = test_data['Exterior2nd'].map({'AsbShng':0,'AsphShn':1,'Brk Cmn':2,'BrkFace':3,'CBlock':4,'CmentBd':5,'HdBoard':6,'ImStucc':7,'MetalSd':8,

                                                           'Other':9,'Plywood':10,'PreCast':11,'Stone':12,'Stucco':13,'VinylSd':14,'Wd Sdng':15,'Wd Shng':16})  
test_data.describe().transpose()
# Missing data counts and percentage

pd.set_option('display.float_format', lambda x: '%.4f' % x)

print('Missing Data Count')

print(test_data[show_missing()].isnull().sum().sort_values(ascending = False))

print('--'*40)

print('Missing Data Percentage')

print(round(test_data[show_missing()].isnull().sum().sort_values(ascending = False)/len(test_data)*100,2))



print(test_data.info())
# Split

# Create matrix of x features for training data

X_train2 = train_data[['OverallQual', 'YearBuilt', 'YearRemodAdd', 'ExterQual', 'BsmtQual', 'TotalBsmtSF', 

               '1stFlrSF', 'GrLivArea', 'FullBath', 'KitchenQual', 'TotRmsAbvGrd', 'FireplaceQu', 

               'GarageFinish', 'GarageCars', 'GarageArea']]



# Create target variable array for training data

y_train2 = train_data['SalePrice']



# Create matrix of x features for test data

X_test2 = test_data[['OverallQual', 'YearBuilt', 'YearRemodAdd', 'ExterQual', 'BsmtQual', 'TotalBsmtSF', 

               '1stFlrSF', 'GrLivArea', 'FullBath', 'KitchenQual', 'TotRmsAbvGrd', 'FireplaceQu', 

               'GarageFinish', 'GarageCars', 'GarageArea']]



# There is no target variable array for test data



# Confirm data shapes

print('Data Shapes')

print('x_train shape', X_train2.shape)

print('y_train shape',y_train2.shape)

print('x_test shape', X_test2.shape)



# Fit Random Forest to training data

from sklearn.ensemble import RandomForestRegressor

rforest = RandomForestRegressor(n_estimators = 300, random_state = 0) 

rforest.fit(X_train2,y_train2)



# Predict using test data

y_pred_rforest2 = rforest.predict(X_test2)
# Create contest submission

submission = pd.DataFrame({

        "Id": test_data["Id"],

        "SalePrice": y_pred_rforest2

    })



submission.to_csv('HousingSubmissionbb.csv', index=False)