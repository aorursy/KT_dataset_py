# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import scipy.stats as stats

from sklearn import ensemble, tree, linear_model

import missingno as msno
train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# Plot the Correlation map to see how features are correlated with target: SalePrice

corr_matrix = train_df.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corr_matrix, vmax=0.9, square=True)
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars','GarageArea' ,'TotalBsmtSF', 'FullBath', 'YearBuilt','TotRmsAbvGrd']

sns.pairplot(train_df[cols], size = 2.5)

plt.show();
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corr_matrix.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train_df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
plt.scatter(train_df['GrLivArea'],train_df['SalePrice'])

plt.xlabel('GrLivArea')

plt.ylabel('SalePrice')

plt.show()
train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)
plt.scatter(train_df['TotalBsmtSF'],train_df['SalePrice'])

plt.xlabel('TotalBsmtSF')

plt.ylabel('SalePrice')

plt.show()
train_df = train_df.drop(train_df[(train_df['TotalBsmtSF']>3000)].index)
plt.scatter(train_df['OverallQual'],train_df['SalePrice'])

plt.xlabel('OverallQual')

plt.ylabel('SalePrice')

plt.show()
train_df = train_df.drop(train_df[(train_df['OverallQual']>9) & (train_df['SalePrice']>700000)].index)
#distribution plot- histogram

sns.distplot(train_df['SalePrice']).set_title("Distribution of SalePrice")



# probability plot

fig = plt.figure()

res = stats.probplot(train_df['SalePrice'], plot=plt)
#Using the log1p function applies log(1+x) to all elements of the column

train_df["SalePrice"] = np.log1p(train_df["SalePrice"])



#Check the new distribution after log transformation 

sns.distplot(train_df['SalePrice'] , fit=stats.norm);



# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(train_df['SalePrice'])

print( '\n mean = {:.2f} and std dev = {:.2f}\n'.format(mu, sigma))



#NPlotting the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Distribution of Log SalePrices')



#Also the QQ plot

fig = plt.figure()

res = stats.probplot(train_df['SalePrice'], plot=plt)

plt.show()
total_df = train_df + test_df


# enlisting variables we want to drop

feature_drop1= ['GarageYrBlt','TotRmsAbvGrd'] # will remove 1stFlrSF and GarageArea later-- after creating additional features

#removing features-- with multicollinearity or low correlation with target variable

total_df.drop(feature_drop1,

              axis=1, inplace=True)

total_df.head()
# columns with attributes like Pool, Fence etc. marked as NaN indicate the absence of these features.

attributes_with_na = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',

               'GarageQual','GarageCond','GarageFinish','GarageType',

               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2']



# replace 'NaN' with 'None' in these columns

for col in attributes_with_na:

    total_df[col].fillna('None',inplace=True)

    

#NAs in basement related columns will indicate no masonry veneer. Thus replacing MasVnr Area with 0

total_df.MasVnrArea.fillna(0,inplace=True)



#NAs in basement related columns will indicate no basement. Thus replacing all areas and count column with 0 

total_df.BsmtFullBath.fillna(0,inplace=True)

total_df.BsmtHalfBath.fillna(0,inplace=True)

total_df.BsmtFinSF1.fillna(0,inplace=True)

total_df.BsmtFinSF2.fillna(0,inplace=True)

total_df.BsmtUnfSF.fillna(0,inplace=True)

total_df.TotalBsmtSF.fillna(0,inplace=True)



#Similarly for Garage Cars-- fill with 0; cause if no garage, no cars can parked in it

total_df.GarageCars.fillna(0,inplace=True)   

# doing the same for GarageArea

total_df.GarageArea.fillna(0,inplace=True)   
msno.matrix(total_df.sample(100))
# Let us first focus on the lesser null percentages (except LoTFrontage)

# Let us see the distribution of data across these fields



# first up: Utilities

total_df.groupby(['Utilities']).size() # only one NoSeWa value and 2 nulls 

train_df.groupby(['Utilities']).size() # train data contains the 'NoSeWa'i.e. Test has no NoSeWa value

# 2 null values come from Test data

## intuitively this will not play a significant role in our model prediction

# for now let us populate the nulls with the most frequent value 'AllPub'-- can drop it later

total_df['Utilities'] = total_df['Utilities'].fillna(total_df['Utilities'].mode()[0]) 



# next is: Functional

# Similarly for Functional

#Functional : by the definition of the column, 'NA' means typical

total_df.groupby(['Functional']).size() # typ has 2717 as of now

# Since 'typ' is also the most frequent value, let us replace 'NA' with 'typ'

total_df["Functional"] = total_df["Functional"].fillna("Typ")

total_df.groupby(['Functional']).size() # typ= 2719 now



# Let us now look at: Electrical

total_df.groupby(['Electrical']).size() # this has one missing value in Train i.e. SBrKr (currently 2671)

# Let us just populate the NA with the most frequent entry

total_df['Electrical'] = total_df['Electrical'].fillna(total_df['Electrical'].mode()[0])

total_df.groupby(['Electrical']).size() # now SBrKr= 2672



# Like Electrical, KitchenQual has 1 missing value

total_df.groupby(['KitchenQual']).size() # the missing value is in Test; most frequent value is 'TA'= 1492

# Let us just replace null with 'TA'

total_df['KitchenQual'] = total_df['KitchenQual'].fillna(total_df['KitchenQual'].mode()[0])

total_df.groupby(['KitchenQual']).size() # 'TA'= 1493



# The next column is SaleType

total_df.groupby(['SaleType']).size() # one NA in Test, most frequent value is 'WD'=2525

#populating nulls with the most frequent values

total_df['SaleType'] = total_df['SaleType'].fillna(total_df['SaleType'].mode()[0])

total_df.groupby(['SaleType']).size() # 'WD'= 2526'



# Doing the same thing for Exterior1st and 2nd

total_df['Exterior1st'] = total_df['Exterior1st'].fillna(total_df['Exterior1st'].mode()[0])

total_df['Exterior2nd'] = total_df['Exterior2nd'].fillna(total_df['Exterior2nd'].mode()[0])



# Moving on to the higher null percentages: MSZoninng

total_df.groupby(['MSZoning']).size() #most frequent value is 'RL'=2265

# Let us just substitute the 4 nulls with the most frequent values

total_df['MSZoning'] = total_df['MSZoning'].fillna(total_df['MSZoning'].mode()[0])

total_df.groupby(['MSZoning']).size() 
# function to scale a column

def norm_minmax(col):

    return (col-col.min())/(col.max()-col.min())

    

# By business definition, LotFrontage is the area of each street connected to the house property

# Intuitively it should be highly correlated to variables like LotArea

# It should also depend on LotShape, LotConfig

# Let us make a simple Linear regressor to get the most accurate values



# convert categoricals to dummies

#also dropping the target 'SalePrice' for now as the target currently is 'LotFrontage'

total_df_dummy = pd.get_dummies(total_df.drop('SalePrice',axis=1))

# scaling all numerical columns

for col in total_df_dummy.drop('LotFrontage',axis=1).columns:

    total_df_dummy[col] = norm_minmax(total_df_dummy[col])



frontage_train = total_df_dummy.dropna()

frontage_train_y = frontage_train.LotFrontage

frontage_train_X = frontage_train.drop('LotFrontage',axis=1)  



# fit model

lin_reg= linear_model.LinearRegression()

lin_reg.fit(frontage_train_X, frontage_train_y)



# check model results

lr_coefs = pd.Series(lin_reg.coef_,index=frontage_train_X.columns)

print(lr_coefs.sort_values(ascending=False))





# use model predictions to populate nulls

nulls_in_lotfrontage = total_df.LotFrontage.isnull()

features = total_df_dummy[nulls_in_lotfrontage].drop('LotFrontage',axis=1)

target = lin_reg.predict(features)



# fill nan values

total_df.loc[nulls_in_lotfrontage,'LotFrontage'] = target