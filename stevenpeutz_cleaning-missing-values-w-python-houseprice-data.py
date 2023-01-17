import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import os

os.getcwd()
houseprice = pd.read_csv('/kaggle/input/train.csv')

# or ...pd.read_csv('../input/train.csv')



#pandas.set_option('display.max_columns', None)
houseprice.head(15)
# To check how many columns have missing values:

def show_missing():

    missing = houseprice.columns[houseprice.isnull().any()].tolist()

    return missing



# This can also be done without creating a function, but this is somewhat less 'pretty

# by using the following:

# houseprice[houseprice.columns[houseprice.isnull().any()].tolist()].isnull().sum()
# Let's see how much work there is to be done regarding cleaning up NaN's and missing values

# this bit will come back several times to check out progress.

houseprice[show_missing()].isnull().sum()
# check correlation with LotArea (because maybe we can replace missing values in LotFrontage 

# with values from LotArea, if they are indeed as similar as we expect)

houseprice['LotFrontage'].corr(houseprice['LotArea'])
# let's see if the square root of lotarea is not a better correlation. If so, we can

# use these values to replace missing values in LotFrontage feature.. 

# improvement - and good enough for now

houseprice['SqrtLotArea']=np.sqrt(houseprice['LotArea'])

houseprice['LotFrontage'].corr(houseprice['SqrtLotArea'])
# Looking at categorical values

def cat_exploration(column):

    return houseprice[column].value_counts()
# Imputing the missing values

def cat_imputation(column, value):

    houseprice.loc[houseprice[column].isnull(),column] = value
houseprice.head(10)
# Saeborn for visualisations, '%pylab inline' will make them within this notebook window.

import seaborn as sns

%pylab inline
# pairplot is good for visualising small amount of variables

# Keep in mind when chosing pairplot; amount of plots is exponential, 2 vars is 2^2, 

# for 10 vars is is 10^2, etc..

sns.pairplot(houseprice[['LotFrontage','SqrtLotArea']].dropna())
# take the cells with empty values in LotFrontage

cond = houseprice['LotFrontage'].isnull()
#replace those cells with values from the correlated SqrtLotArea

houseprice.LotFrontage[cond] = houseprice.SqrtLotArea[cond]
houseprice.head(8)
#check whether LotFrontage is no longer in list of missing values

houseprice[show_missing()].isnull().sum()
cat_exploration('Alley')

# This cat_exploration is possible because we have created this function (def..)

# If you haven't, same result can be gotten with te following:



# houseprice['Alley'].value_counts()
# I assume empty fields here means no alley access

cat_imputation('Alley','None')

# again, this is possible because we have created this function (input 12).

# If we hadn't done this, we would get the same result with the following:



# houseprice.loc[houseprice['Alley'].isnull(),'Alley'] = 'None'
# Let's see how much work there is to be done regarding cleaning up NaN's and missing values

# this bit will come back several times to check out progress.



houseprice[show_missing()].isnull().sum()

# As said before, this can be done without using the created function (def..) 

# by using the following:



# houseprice[houseprice.columns[houseprice.isnull().any()].tolist()].isnull().sum()
houseprice['MasVnrType'].isnull().sum()
# Is MasVnrArea empty when MasVnrType is empty?

houseprice[['MasVnrType','MasVnrArea']][houseprice['MasVnrType'].isnull()==True]
# What do the values look like for MasVnrType?

cat_exploration('MasVnrType')

# or:

# houseprice['MasVnrType'].value_counts()
# Mostely 'None', so we for now will replace NaN's with None's, 

# and for MasVnrArea replace with zero.

cat_imputation('MasVnrType', 'None')

cat_imputation('MasVnrArea', 0.0)
# A lot of variables which are all basement related. Create group, see group to check whether all are

# Nan and zero together:

basement_cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2']

houseprice[basement_cols][houseprice['BsmtQual'].isnull()==True]
# Little loop here. Goes through each of column (x) within

# the created basement 'group'. For each column it goes through the cat_imputation function which

# does this: 



# houseprice.loc[houseprice['x'].isnull(),'x'] = 'None'



# The 'FinSF' is to differentiate between the numerical (both contain 'FinSF' in header) 

# and the categorical which already contain zero's. 



for x in basement_cols:

    if 'FinSF'not in x:

        cat_imputation(x,'None')
# this bit will come back several times to check out progress.



houseprice[show_missing()].isnull().sum()
cat_exploration('Electrical')

# or:

# houseprice['Electrical'].value_counts()
houseprice['Electrical'].isnull().sum()
# Just one missing, impute most frequent value (SBrkr with 1334 instances)

cat_imputation('Electrical','SBrkr')
# Let's see how much work there is to be done regarding cleaning up NaN's and missing values

# this bit will come back several times to check out progress.



houseprice[show_missing()].isnull().sum()
cat_exploration('FireplaceQu')

# or:

# houseprice['FireplaceQu'].value_counts()
houseprice['FireplaceQu'].isnull().sum()
#houseprice['Fireplaces'][houseprice['FireplaceQu'].isnull()==True].describe()



#checking whether FireplaceQu might be empty especially when fireplace itself is missing

houseprice[['Fireplaces','FireplaceQu']][houseprice['FireplaceQu'].isnull()==True]
#So yes, it seems that indeed FireplaceQu is empty when Fireplaces is missing.

#Assumption therefore will be that FireplacesQu(ality) is empty because there is no fireplace.

cat_imputation('FireplaceQu','None')
pd.crosstab(houseprice.Fireplaces, houseprice.FireplaceQu)
# Let's see how much work there is to be done regarding cleaning up NaN's and missing values

# this bit will come back several times to check out progress.



houseprice[show_missing()].isnull().sum()
#Same idea as with basement columns but now for garage related features:

garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']

houseprice[garage_cols][houseprice['GarageType'].isnull()==True]
#Garage Imputation

# This is similar to the loop done for the basement features. The difference here is that

# this one has to differentate feature datatypes to replace categoricals missings with 'none'

# and numerical missings with '0'.



for i in garage_cols:

    if houseprice[i].dtype==np.object:

        cat_imputation(i,'None')

    else:

        cat_imputation(i, 0)

    

# Let's see how much work there is to be done regarding cleaning up NaN's and missing values

# this bit will come back several times to check out progress.



houseprice[show_missing()].isnull().sum()
cat_exploration('PoolQC')
# is Poolarea missing when PoolQC is missing?



houseprice[['PoolArea','PoolQC']][houseprice['PoolQC'].isnull()==True]
# So, here I am going to delete this feature because Pool itself is so often not present, poolQC

# will not be a good feature for modelling. (this should technically be done only after visualising etc)

del houseprice['PoolQC']
# If you don't want to delete: cat_imputation('PoolQC', 'None')
cat_imputation('Fence', 'None')
cat_imputation('MiscFeature', 'None')
# Let's see how much work there is to be done regarding cleaning up NaN's and missing values

# this bit will come back several times to check out progress.



houseprice[show_missing()].isnull().sum()
os.getcwd()
#houseprices.to_csv('../pathhere../submission.csv', index=False)

houseprice.to_csv('/kaggle/working/testsave.csv')
houseprice.head(10)

# At least NaN's should be 'None' in categorical (if conceptually alright) and 0 in numerical