# librabries to use
import pandas as pd
pd.options.display.max_columns = 400
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from scipy.stats import norm, skew
import warnings
warnings.filterwarnings('ignore') # remove usless warning
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test['SalePrice'] = 0
combined = pd.concat((train,test))
combined = combined.loc[:,test.columns] # during CSV creation, columns sorted alphabetacally
print('train file size is'+"   " , train.shape)
print('test file size is'+  "    ", test.shape)
print('combined file size is' , combined.shape)
# exploring Heads and to see if all is OK
train.head()
test.head()
combined.head()
combined.tail()
mu, sigma=norm.fit(train.SalePrice)
sns.set_style('darkgrid')
sns.distplot(train['SalePrice'] , fit=norm)
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)])
# it's right skewed, can't get much without normalization
print('Mean  = ${:.2f} \n Sigma = ${:.2f}'.format(mu,sigma))
train.SalePrice.describe()
# find storng correlations
cormat = train.corr().SalePrice.sort_values(ascending=False)
cormat.head(12) # observation : GarageArea and GarageCars are very similar in high correlation, will decide what to do with them
train.plot.scatter('GrLivArea', 'SalePrice') # use pd.plot is much better than direct matplotlib
train=train[train.GrLivArea<4600] # 2 assumed farm land data points cropped
combined=combined[combined.GrLivArea<4600] 
train.shape
train.plot.scatter('OverallQual','SalePrice') # seems like categrical data
train.plot.scatter('OverallCond','SalePrice') # Invidtigate further later
sns.boxplot('OverallQual','SalePrice', data = train) # nicer way to observe 
# Kaggel Scoring (log1p of target)
train['SalePrice'] = np.log1p(train['SalePrice']) # taking ln will return relative SalePrice values, and errors for cheap and expensive will affect results equally
# remove features that have no impact on result
combined.drop('Id',inplace=True,axis=1) # Id is a nominal feature (without inplace, you get and empty columns)
# remove leaking data features (we can'y know them except after the sale)
combined.drop(['SaleType','SaleCondition','MoSold'],inplace=True,axis=1) # Year Sold will be removed later after doing some opertaions on it
# Check missing values
mv_perc = combined.isnull().sum()/len(combined)*100
mv_count = combined.isnull().sum()
mv_table = pd.concat([mv_count,mv_perc],axis=1,keys=['Count','Percent'])
mv_table = mv_table[mv_table['Percent']>0].sort_values('Percent',ascending=False)
mv_table
# dealing with missing vlaues
combined.Alley = combined.Alley.fillna('NA') # if alley don't exists, fill with NA
combined.PoolQC = combined.PoolQC.fillna('NA') # if pool don't exists, fill with NA
combined.MiscFeature = combined.MiscFeature.fillna('NA') # assume no features by default
# notice that Garage Info is missing for the same 159 cells
combined[['GarageQual','GarageCond','GarageFinish','GarageType']] = combined[['GarageQual','GarageCond','GarageFinish','GarageType']].fillna('NA')
combined[['GarageCars','GarageArea','GarageYrBlt']] = combined[['GarageCars','GarageArea','GarageYrBlt']].fillna(0) #Number of garage cars and area, if no garage, no cars, only 1 observation
combined[['BedroomAbvGr','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF']] = combined[['BedroomAbvGr','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF']].fillna(0)
combined[['BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual','BsmtCond']] = combined[['BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual','BsmtCond']].fillna('NA')
combined.Fence=combined.Fence.fillna('NA') # NA mean no fence
combined.KitchenQual = combined.KitchenQual.fillna('TA')
combined.Functional = combined.Functional.fillna('Typ')
combined.Electrical = combined.Electrical.fillna(combined.Electrical.mode()[0]) # use the mode, since there is no NA option
combined.Utilities = combined.Utilities.fillna(combined.Utilities.mode()[0])
combined.MasVnrArea = combined.MasVnrArea.fillna(0)
combined.MasVnrType = combined.MasVnrType.fillna('None')
combined.MSZoning = combined.MSZoning.fillna(combined.MSZoning.mode()[0]) # type of sale zone
combined.FireplaceQu = combined.FireplaceQu.fillna('NA')
combined["LotFrontage"] = combined.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median())) # Area of one front in neighborhood is the same for all houses
combined.Exterior1st = combined.Exterior1st.fillna(combined.Exterior1st.mode()[0])
combined.Exterior2nd = combined.Exterior2nd.fillna(combined.Exterior2nd.mode()[0])
mv_perc = combined.isnull().sum()/len(combined)*100 
print('Missing Values count is', mv_perc.max()) # Nothing is missing
combined.info()
numfeat = train.select_dtypes(include = ['int64','float64']).columns
objfeat = train.select_dtypes(include ='object').columns
years_sold = combined['YrSold'] - combined['YearBuilt'] # we are not intersted in abosulute dates, aonly relative date ( ratio not interval scales)
years_mod = combined['YrSold'] - combined['YearRemodAdd']
combined['YearsSold'] = years_sold
combined['YearsRemod'] = years_sold
combined['YearABS'] = combined['YearBuilt'].apply(lambda x: abs(datetime.datetime.now().year - x))
cpmbined = combined.drop(['YrSold','YearBuilt','YearRemodAdd'], inplace=True, axis=1)
combined[years_sold<0] # a year where the house is modified aftr its sold is @ index 2295
combined.drop([2295],axis=0)
fig=plt.figure(figsize=(12,12))
sns.heatmap(train[numfeat].corr())
# Garage cars and Garage Area have the same correlation accross most fields, the one with lower correlation to sale will be removed
combined = combined.drop('GarageArea', axis=1) # Collinearity, removing this improved rmse
# disguised
combined['MSSubClass'] = combined['MSSubClass'].apply(str)
combined['OverallCond'] = combined['OverallCond'].astype(str)
combined.shape
# low variability selector (temporary)
# Since we know that total area of a house is really import 
combined['TotalIntrSF'] = combined['1stFlrSF'] + combined['2ndFlrSF']+combined['TotalBsmtSF']
combined["AllSF"] = combined["GrLivArea"] + combined["TotalIntrSF"]
combined["AllPorchSF"] = combined["OpenPorchSF"] + combined["EnclosedPorch"] + combined["3SsnPorch"] + combined["ScreenPorch"]
# Skewdness
numfeat = combined.dtypes[combined.dtypes != "object"].index
skewed = combined[numfeat].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = abs(skewed)>0.75
from scipy.special import boxcox1p
skewedfeat = skewness.index
lam = 0.0
for feat in skewedfeat:
    combined[feat] = boxcox1p(combined[feat], lam) # or choose lam=0
    # combined[feat] = combined[feat].apply(lambda x: np.log1p(x),1) # taking the log(x+1) resulted in mych better rmse than boxcox1p
skewed.head() # before
combined.shape
# After BoxCox
skewed = combined[numfeat].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewed.head()
combined.shape
# Transform categroical features with label_encoder (question: do we need to perofrom box cox before or after encoding?)
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond')

transformed_cat_cols = []
for col in cols:
    if col in combined.columns:
        transformed_cat_cols.append(col)

value_count_dict = {}
for col in transformed_cat_cols:
    value = combined[col].value_counts().shape[0]
    value_count_dict[col] = value

#  remove features with so many categories
nr_category = 12  # Experiment at differnt cutoffs
uniqueness_counts = combined[transformed_cat_cols].apply(lambda col: len(col.value_counts())).sort_values()
drop_nonuniq_cols = uniqueness_counts[uniqueness_counts > nr_category].index
combined = combined.drop(drop_nonuniq_cols, axis=1)
# get_dummies
text_cols = combined.select_dtypes(include=['object'])
for col in text_cols:
    combined[col] = combined[col].astype('category')
combined = pd.concat([combined,pd.get_dummies(combined.select_dtypes(include='category'))],axis=1)
combined = combined.drop(combined.columns[(combined.dtypes=='category')],axis=1) # drop categories after conversion to dummies
print(value_count_dict)
print(combined.shape)
# Turn top 10 most correlated into polynomial
# split into train and test data 
ntrain = 1458
train = combined[:ntrain]
test = combined[ntrain:].drop('SalePrice',axis=1)
features = test.columns
lr = LinearRegression()
# Validation function
n_folds = 10

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train[features], train['SalePrice'], scoring = "neg_mean_squared_error", cv = kf))
    return(rmse)
rmsle_cv(lr).mean()
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = rmsle_cv(lasso)
print('mean: {}'.format(score.mean()), '\n std: {}'.format(score.std()))
lasso.fit(train[features],train['SalePrice'])
prediction = np.expm1(lasso.predict(test[features])) #invert price back to actual dollars
prediction[:10]