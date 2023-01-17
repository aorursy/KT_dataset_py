# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime

sns.set(style="white")

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Configuring dataset view option

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 100)
#Reading data 

trdf = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

tsdf = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

#Combining test and train to gain time in data engineering

fulldf = pd.concat([trdf,tsdf], ignore_index=True, sort = False )
#Checking overall result and DF Shape

print('Train Shape ', trdf.shape)

print('Test Shape ', tsdf.shape)

print('All data Shape ', fulldf.shape)
fulldf
#Dropping ID Column

#Dropping in Fulldata

fulldf.drop(['Id'],axis=1, inplace=True)

#Dropping in Train data

trdf.drop(['Id'],axis=1, inplace=True)

#Dropping in Train data

tsdf.drop(['Id'],axis=1, inplace=True)

#Checking data tipe of FULL variables

fulldf.info()
#Checking statistical information of FULL data

describe1df=trdf.describe().T

trdf.describe().T
#Understandinng Correlation for ALL data(mind the NaN in sales prices)

plt.figure(figsize=(20,20))

sns.heatmap(fulldf.corr(),

            vmin=-1,

            cmap='coolwarm',

            annot=True);
#Checking Group formation and distribution (This worths better for categorical data) of FULL data

column = fulldf.columns

for i in column:

    print('***** ',i, '\n')

    print(fulldf[i].value_counts(),'\n')

#Checking NaN values on FULL data

miss_sum = pd.DataFrame(fulldf.isnull().sum().sort_values(ascending=False), columns = ['Total'])

miss_percent = pd.DataFrame((fulldf.isnull().sum()/fulldf.isnull().count()*100), columns = ['Percentage'])

missfulldf = pd.concat([miss_sum,miss_percent], axis=1)

print(missfulldf[missfulldf['Total']>0])

print('\n********TOTALS**********\n',missfulldf[missfulldf['Total']>0].count())
#Checking NaN values on TRAIN data

miss_sum = pd.DataFrame(trdf.isnull().sum().sort_values(ascending=False), columns = ['Total'])

miss_percent = pd.DataFrame((trdf.isnull().sum()/trdf.isnull().count()*100), columns = ['Percentage'])

missfulldf = pd.concat([miss_sum,miss_percent], axis=1)

print(missfulldf[missfulldf['Total']>0])

print('\n********TOTALS**********\n',missfulldf[missfulldf['Total']>0].count())
#Dropping high missing rate columns

#Dropping in FULL df

fulldf.drop(['PoolQC','Alley','MiscFeature','Fence','FireplaceQu'], axis=1,inplace=True)

#Dropping in TRAIN df

trdf.drop(['PoolQC','Alley','MiscFeature','Fence','FireplaceQu'], axis=1,inplace=True)

#Dropping in TEST df

tsdf.drop(['PoolQC','Alley','MiscFeature','Fence','FireplaceQu'], axis=1,inplace=True)



#Checking NaN values on FULL data

miss_sum = pd.DataFrame(fulldf.isnull().sum().sort_values(ascending=False), columns = ['Total'])

miss_percent = pd.DataFrame((fulldf.isnull().sum()/fulldf.isnull().count()*100), columns = ['Percentage'])

missfulldf = pd.concat([miss_sum,miss_percent], axis=1)

print(missfulldf[missfulldf['Total']>0])

print('\n********TOTALS**********\n',missfulldf[missfulldf['Total']>0].count())
#Checking 0 values on FULL data

zeros_sum = pd.DataFrame(fulldf[(fulldf==0)].count().sort_values(ascending=False), columns = ['Total'])

zeros_percent = pd.DataFrame((fulldf[(fulldf==0)].count()/fulldf.isnull().count()*100), columns = ['Percentage'])

zerosfulldf = pd.concat([zeros_sum,zeros_percent], axis=1)

print(zerosfulldf[zerosfulldf['Total']>0])

print('\n********TOTALS**********\n',zerosfulldf[zerosfulldf['Total']>0].count())
#Dropping high Zeros rate columns

#Dropping in FULL df

fulldf.drop(['PoolArea','3SsnPorch','LowQualFinSF','MiscVal','BsmtHalfBath','ScreenPorch','BsmtFinSF2','EnclosedPorch'\

             ,'HalfBath','MasVnrArea','BsmtFullBath','2ndFlrSF','WoodDeckSF','Fireplaces'], axis=1,inplace=True)

#Dropping in TRAIN df

trdf.drop(['PoolArea','3SsnPorch','LowQualFinSF','MiscVal','BsmtHalfBath','ScreenPorch','BsmtFinSF2','EnclosedPorch'\

             ,'HalfBath','MasVnrArea','BsmtFullBath','2ndFlrSF','WoodDeckSF','Fireplaces'], axis=1,inplace=True)

#Dropping in TEST df

tsdf.drop(['PoolArea','3SsnPorch','LowQualFinSF','MiscVal','BsmtHalfBath','ScreenPorch','BsmtFinSF2','EnclosedPorch'\

             ,'HalfBath','MasVnrArea','BsmtFullBath','2ndFlrSF','WoodDeckSF','Fireplaces'], axis=1,inplace=True)



#Checking 0 values on FULL data

zeros_sum = pd.DataFrame(fulldf[(fulldf==0)].count().sort_values(ascending=False), columns = ['Total'])

zeros_percent = pd.DataFrame((fulldf[(fulldf==0)].count()/fulldf.isnull().count()*100), columns = ['Percentage'])

zerosfulldf = pd.concat([zeros_sum,zeros_percent], axis=1)

print(zerosfulldf[zerosfulldf['Total']>0])

print('\n********TOTALS**********\n',zerosfulldf[zerosfulldf['Total']>0].count())
fulldf.info()
#Converting Category(Qualitative) data



#Converting on FULL df

fulldf = fulldf.astype({'MSSubClass': 'category', 'MSZoning': 'category', 'Street': 'category', 'LotShape': 'category',\

                   'LandContour': 'category', 'Neighborhood': 'category', 'LandSlope': 'category', 'LotConfig': 'category', \

                   'Utilities': 'category', 'Condition1': 'category', 'Condition2': 'category', 'BldgType': 'category', \

                   'OverallQual': 'category', 'RoofStyle': 'category', 'RoofMatl': 'category', 'Exterior2nd': 'category', \

                   'MasVnrType': 'category', 'ExterQual': 'category', 'ExterCond': 'category', 'Foundation': 'category', \

                   'BsmtQual': 'category', 'BsmtCond': 'category', 'BsmtExposure': 'category', 'BsmtFinType1': 'category',\

                   'BsmtFinType2': 'category', 'Heating': 'category', 'HeatingQC': 'category', 'CentralAir': 'category', \

                   'Electrical': 'category', 'KitchenQual': 'category', 'Functional': 'category', 'GarageType': 'category',\

                   'GarageFinish': 'category', 'GarageCond': 'category', 'PavedDrive': 'category', 'SaleType': 'category', \

                   'Exterior1st': 'category', 'SaleCondition': 'category', 'HouseStyle': 'category', 'OverallCond': 'category',\

                        'KitchenAbvGr': 'category','GarageQual': 'category'})



#Converting on TRAIN df

trdf = trdf.astype({'MSSubClass': 'category', 'MSZoning': 'category', 'Street': 'category', 'LotShape': 'category',\

                   'LandContour': 'category', 'Neighborhood': 'category', 'LandSlope': 'category', 'LotConfig': 'category', \

                   'Utilities': 'category', 'Condition1': 'category', 'Condition2': 'category', 'BldgType': 'category', \

                   'OverallQual': 'category', 'RoofStyle': 'category', 'RoofMatl': 'category', 'Exterior2nd': 'category', \

                   'MasVnrType': 'category', 'ExterQual': 'category', 'ExterCond': 'category', 'Foundation': 'category', \

                   'BsmtQual': 'category', 'BsmtCond': 'category', 'BsmtExposure': 'category', 'BsmtFinType1': 'category',\

                   'BsmtFinType2': 'category', 'Heating': 'category', 'HeatingQC': 'category', 'CentralAir': 'category', \

                   'Electrical': 'category', 'KitchenQual': 'category', 'Functional': 'category', 'GarageType': 'category',\

                   'GarageFinish': 'category', 'GarageCond': 'category', 'PavedDrive': 'category', 'SaleType': 'category', \

                   'Exterior1st': 'category', 'SaleCondition': 'category', 'HouseStyle': 'category', 'OverallCond': 'category',\

                    'KitchenAbvGr': 'category','GarageQual': 'category'})



#Converting on TEST df

tsdf = tsdf.astype({'MSSubClass': 'category', 'MSZoning': 'category', 'Street': 'category', 'LotShape': 'category',\

                   'LandContour': 'category', 'Neighborhood': 'category', 'LandSlope': 'category', 'LotConfig': 'category', \

                   'Utilities': 'category', 'Condition1': 'category', 'Condition2': 'category', 'BldgType': 'category', \

                   'OverallQual': 'category', 'RoofStyle': 'category', 'RoofMatl': 'category', 'Exterior2nd': 'category', \

                   'MasVnrType': 'category', 'ExterQual': 'category', 'ExterCond': 'category', 'Foundation': 'category', \

                   'BsmtQual': 'category', 'BsmtCond': 'category', 'BsmtExposure': 'category', 'BsmtFinType1': 'category',\

                   'BsmtFinType2': 'category', 'Heating': 'category', 'HeatingQC': 'category', 'CentralAir': 'category', \

                   'Electrical': 'category', 'KitchenQual': 'category', 'Functional': 'category', 'GarageType': 'category',\

                   'GarageFinish': 'category', 'GarageCond': 'category', 'PavedDrive': 'category', 'SaleType': 'category', \

                   'Exterior1st': 'category', 'SaleCondition': 'category', 'HouseStyle': 'category', 'OverallCond': 'category',\

                    'KitchenAbvGr': 'category','GarageQual': 'category'})
fulldf.info()
#Checking categorical data statistics of FULL data

fulldf.describe(include='category').T
#Checking categorical with more than 80% of the same category wich is dispendable for analisys

dfcatdescrb = fulldf.describe(include='category').T

freq_sum = pd.DataFrame(dfcatdescrb['freq'].sort_values(ascending=False)).rename(columns = {'freq':'Total'})

duplic_percent = pd.DataFrame(dfcatdescrb['freq']/fulldf.select_dtypes("category").count()*100).rename(columns = {0:'Percentage'})

duplic_full = pd.concat([freq_sum, duplic_percent], axis=1)

duplic_full

print(duplic_full[duplic_full['Percentage']>=80])

print('\n********TOTALS**********\n',duplic_full[duplic_full['Percentage']>=80].count())
#Dropping high duplicated rate columns

droplist = duplic_full[duplic_full['Percentage']>=80].index



#Dropping in FULL df

fulldf.drop(droplist, axis=1, inplace=True)

#Dropping in TRAIN df

trdf.drop(droplist, axis=1,inplace=True)

#Dropping in TEST df

tsdf.drop(droplist, axis=1,inplace=True)



#Checking categorical with more than 80% of the same category wich is dispendable for analisys

dfcatdescrb = fulldf.describe(include='category').T

freq_sum = pd.DataFrame(dfcatdescrb['freq'].sort_values(ascending=False)).rename(columns = {'freq':'Total'})

duplic_percent = pd.DataFrame(dfcatdescrb['freq']/fulldf.select_dtypes("category").count()*100).rename(columns = {0:'Percentage'})

duplic_full = pd.concat([freq_sum, duplic_percent], axis=1)

duplic_full

print(duplic_full[duplic_full['Percentage']>=80])

print('\n********TOTALS**********\n',duplic_full[duplic_full['Percentage']>=80].count())

#Checking the remaining cleasing to be done

print('****************************************************************************************************')

print('Remaining NaN Variables')

print('****************************************************************************************************\n')

#Checking NaN values on FULL data

dftype = pd.DataFrame(fulldf.dtypes, columns = ['DType'])

miss_sum = pd.DataFrame(fulldf.isnull().sum().sort_values(ascending=False), columns = ['Total'])

miss_percent = pd.DataFrame((fulldf.isnull().sum()/fulldf.isnull().count()*100), columns = ['Percentage'])

missfulldf = pd.concat([miss_sum,miss_percent,dftype], axis=1)

print('--> NaN Category \n')

print(missfulldf[(missfulldf['Total']>0) & (missfulldf['DType']=='category')])

print('\n********TOTALS**********\n',missfulldf[(missfulldf['Total']>0) & (missfulldf['DType']=='category')].count())

print('\n --> NaN Numeric \n')

print(missfulldf[(missfulldf['Total']>0) & (missfulldf['DType']!='category')])

print('\n********TOTALS**********\n',missfulldf[(missfulldf['Total']>0) & (missfulldf['DType']!='category')].count())

print('\n \n****************************************************************************************************')

print('Remaining Zeros Variables')

print('****************************************************************************************************\n')

#Checking 0 values on FULL data

zeros_sum = pd.DataFrame(fulldf[(fulldf==0)].count().sort_values(ascending=False), columns = ['Total'])

zeros_percent = pd.DataFrame((fulldf[(fulldf==0)].count()/fulldf.isnull().count()*100), columns = ['Percentage'])

zerosfulldf = pd.concat([zeros_sum,zeros_percent,dftype], axis=1)

print('--> Zeros Category \n')

print(zerosfulldf[(zerosfulldf['Total']>0) & (zerosfulldf['DType']=='category')])

print('\n********TOTALS**********\n',zerosfulldf[(zerosfulldf['Total']>0) & (zerosfulldf['DType']=='category')].count())

print('\n --> Zeros Numeric \n')

print(zerosfulldf[(zerosfulldf['Total']>0) & (zerosfulldf['DType']!='category')])

print('\n********TOTALS**********\n',zerosfulldf[(zerosfulldf['Total']>0) & (zerosfulldf['DType']!='category')].count())

nonelist = ['GarageFinish','GarageType',\

            'BsmtExposure','BsmtQual','BsmtFinType1','MasVnrType']

nonelistwithout_none = ['GarageFinish','GarageType',\

            'BsmtExposure','BsmtQual','BsmtFinType1']



#adding None Category to the variables

for colunaa in nonelistwithout_none:

    fulldf[colunaa] = fulldf[colunaa].cat.add_categories('None')

    trdf[colunaa] = trdf[colunaa].cat.add_categories('None')

    tsdf[colunaa] = tsdf[colunaa].cat.add_categories('None')



for coluna in nonelist:

    #Replacing in FULL df

    fulldf[coluna].fillna('None', inplace=True)

    #Replacing in TRAIN df

    trdf[coluna].fillna('None', inplace=True)

    #Replacing in TEST df

    tsdf[coluna].fillna('None', inplace=True)

    #Checking the remaining cleasing to be done

print('****************************************************************************************************')

print('Remaining NaN Variables')

print('****************************************************************************************************\n')

#Checking NaN values on FULL data

dftype = pd.DataFrame(fulldf.dtypes, columns = ['DType'])

miss_sum = pd.DataFrame(fulldf.isnull().sum().sort_values(ascending=False), columns = ['Total'])

miss_percent = pd.DataFrame((fulldf.isnull().sum()/fulldf.isnull().count()*100), columns = ['Percentage'])

missfulldf = pd.concat([miss_sum,miss_percent,dftype], axis=1)

print('--> NaN Category \n')

print(missfulldf[(missfulldf['Total']>0) & (missfulldf['DType']=='category')])

print('\n********TOTALS**********\n',missfulldf[(missfulldf['Total']>0) & (missfulldf['DType']=='category')].count())

print('\n \n****************************************************************************************************')

print('Remaining Zeros Variables')

print('****************************************************************************************************\n')

modelist = ['MSZoning','Exterior1st','Exterior2nd','KitchenQual']

for colzer in modelist:

    #Replacing in FULL df

    fulldf[colzer].fillna(fulldf[colzer].mode()[0], inplace=True)

    #Replacing in TRAIN df

    trdf[colzer].fillna(trdf[colzer].mode()[0], inplace=True)

    #Replacing in TEST df

    tsdf[colzer].fillna(tsdf[colzer].mode()[0], inplace=True)

    #Checking the remaining cleasing to be done

print('****************************************************************************************************')

print('Remaining NaN Variables')

print('****************************************************************************************************\n')

#Checking NaN values on FULL data

dftype = pd.DataFrame(fulldf.dtypes, columns = ['DType'])

miss_sum = pd.DataFrame(fulldf.isnull().sum().sort_values(ascending=False), columns = ['Total'])

miss_percent = pd.DataFrame((fulldf.isnull().sum()/fulldf.isnull().count()*100), columns = ['Percentage'])

missfulldf = pd.concat([miss_sum,miss_percent,dftype], axis=1)

print('--> NaN Category \n')

print(missfulldf[(missfulldf['Total']>0) & (missfulldf['DType']=='category')])

print('\n********TOTALS**********\n',missfulldf[(missfulldf['Total']>0) & (missfulldf['DType']=='category')].count())

print('\n \n****************************************************************************************************')

print('Remaining Zeros Variables')

print('****************************************************************************************************\n')
nunnanlist = [ 'LotFrontage','GarageCars','BsmtUnfSF','GarageArea','BsmtFinSF1','TotalBsmtSF']

for colnum in nunnanlist:

    #Replacing in FULL df

    fulldf[colnum].fillna(0, inplace=True)

    #Replacing in TRAIN df

    trdf[colnum].fillna(0, inplace=True)

    #Replacing in TEST df

    tsdf[colnum].fillna(0, inplace=True)

#Checking the remaining cleasing to be done

print('****************************************************************************************************')

print('Remaining NaN Variables')

print('****************************************************************************************************\n')

#Checking NaN values on FULL data

dftype = pd.DataFrame(fulldf.dtypes, columns = ['DType'])

miss_sum = pd.DataFrame(fulldf.isnull().sum().sort_values(ascending=False), columns = ['Total'])

miss_percent = pd.DataFrame((fulldf.isnull().sum()/fulldf.isnull().count()*100), columns = ['Percentage'])

missfulldf = pd.concat([miss_sum,miss_percent,dftype], axis=1)

print('\n --> NaN Numeric \n')

print(missfulldf[(missfulldf['Total']>0) & (missfulldf['DType']!='category')])

print('\n********TOTALS**********\n',missfulldf[(missfulldf['Total']>0) & (missfulldf['DType']!='category')].count())
#fixing GarageYrBlt

fulldf['GarageYrBlt'].fillna(fulldf['YearBuilt'], inplace = True)

trdf['GarageYrBlt'].fillna(trdf['YearBuilt'], inplace = True)

tsdf['GarageYrBlt'].fillna(tsdf['YearBuilt'], inplace = True)

#Checking the remaining cleasing to be done

print('****************************************************************************************************')

print('Remaining NaN Variables')

print('****************************************************************************************************\n')

#Checking NaN values on FULL data

dftype = pd.DataFrame(fulldf.dtypes, columns = ['DType'])

miss_sum = pd.DataFrame(fulldf.isnull().sum().sort_values(ascending=False), columns = ['Total'])

miss_percent = pd.DataFrame((fulldf.isnull().sum()/fulldf.isnull().count()*100), columns = ['Percentage'])

missfulldf = pd.concat([miss_sum,miss_percent,dftype], axis=1)

print('\n --> NaN Numeric \n')

print(missfulldf[(missfulldf['Total']>0) & (missfulldf['DType']!='category')])

print('\n********TOTALS**********\n',missfulldf[(missfulldf['Total']>0) & (missfulldf['DType']!='category')].count())

#Checking the remaining cleasing to be done

print('****************************************************************************************************')

print('Remaining NaN Variables')

print('****************************************************************************************************\n')

#Checking NaN values on FULL data

dftype = pd.DataFrame(fulldf.dtypes, columns = ['DType'])

miss_sum = pd.DataFrame(fulldf.isnull().sum().sort_values(ascending=False), columns = ['Total'])

miss_percent = pd.DataFrame((fulldf.isnull().sum()/fulldf.isnull().count()*100), columns = ['Percentage'])

missfulldf = pd.concat([miss_sum,miss_percent,dftype], axis=1)

print('--> NaN Category \n')

print(missfulldf[(missfulldf['Total']>0) & (missfulldf['DType']=='category')])

print('\n********TOTALS**********\n',missfulldf[(missfulldf['Total']>0) & (missfulldf['DType']=='category')].count())

print('\n --> NaN Numeric \n')

print(missfulldf[(missfulldf['Total']>0) & (missfulldf['DType']!='category')])

print('\n********TOTALS**********\n',missfulldf[(missfulldf['Total']>0) & (missfulldf['DType']!='category')].count())

print('\n \n****************************************************************************************************')

print('Remaining Zeros Variables')

print('****************************************************************************************************\n')

#Checking 0 values on FULL data

zeros_sum = pd.DataFrame(fulldf[(fulldf==0)].count().sort_values(ascending=False), columns = ['Total'])

zeros_percent = pd.DataFrame((fulldf[(fulldf==0)].count()/fulldf.isnull().count()*100), columns = ['Percentage'])

zerosfulldf = pd.concat([zeros_sum,zeros_percent,dftype], axis=1)

print('--> Zeros Category \n')

print(zerosfulldf[(zerosfulldf['Total']>0) & (zerosfulldf['DType']=='category')])

print('\n********TOTALS**********\n',zerosfulldf[(zerosfulldf['Total']>0) & (zerosfulldf['DType']=='category')].count())

print('\n --> Zeros Numeric \n')

print(zerosfulldf[(zerosfulldf['Total']>0) & (zerosfulldf['DType']!='category')])

print('\n********TOTALS**********\n',zerosfulldf[(zerosfulldf['Total']>0) & (zerosfulldf['DType']!='category')].count())



#Checking overall variable

featdf = pd.DataFrame(fulldf.dtypes,columns=['Data_Type'])

print('****************************************************************************************************')

print('Category Variables')

print('****************************************************************************************************\n')

print(featdf[featdf['Data_Type']=='category'])

print('\n**Total: ',featdf[featdf['Data_Type']=='category'].count(),'**')

print('\n ****************************************************************************************************')

print('Numerical Variables')

print('****************************************************************************************************\n')

print(featdf[featdf['Data_Type']!='category'])

print('\n**Total: ',featdf[featdf['Data_Type']!='category'].count(),'**')
#Getting list of headers

nunfeat = list(featdf[featdf['Data_Type']!='category'].index)

catfeat = list(featdf[featdf['Data_Type']=='category'].index)
#Checking statistical data Numerical Data

(fulldf.describe().T).sort_values(by=['std'], ascending=False)
#Plotting variables histogram

fulldf.hist(figsize=(15,15))

plt.show()
#Checking variables distribution for numerical variables

for graf in nunfeat:

    sns.distplot(trdf[graf])

    plt.show()
#Checking Scatterplot for numerical data

for graf in nunfeat:

    sns.scatterplot(x=graf, y='SalePrice', data=trdf)

    plt.show()
#Checking BoxPlot looking for outliers

for graf in nunfeat:

    sns.boxplot(trdf[graf])

    plt.show()
#Checking statistical data Categorical Data

fulldf.describe(include='category').T
#checking categorial data distribution

for cgraf in catfeat:

    sns.boxplot(trdf[cgraf],trdf['SalePrice'] )

    plt.show()
from scipy.stats import chi2_contingency

#Creating a fuction to apply test in the DF

def chisq_func(df, c1, c2):

    groupsizes = df.groupby([c1, c2]).size()

    ctsum = groupsizes.unstack(c1)

    # fillna(0) is necessary to remove any NAs which will cause exceptions

    return(chi2_contingency(ctsum.fillna(0)))

#

print('************CHISQUARE TEST*****************\n')

pvalues = []

for feat in catfeat:

    print(feat)

    chires = chisq_func(trdf,feat,'SalePrice')

    print(chires[1])

    pvalues.append(chires[1])

chidf = pd.DataFrame(pvalues,index=catfeat,columns=['P-Value'])

chidf.sort_values(by='P-Value')
#Checking the only varibles with correlation with SalesPrice

chidf[chidf['P-Value']<0.05]
#Checking the only varibles with NO correlation with SalesPrice

chidf[chidf['P-Value']>0.05]
#Dropping variables with no correlation with SalesPrice

droplist = list(chidf[chidf['P-Value']>0.05].index)

trdf.drop(droplist,axis=1,inplace=True)

tsdf.drop(droplist,axis=1,inplace=True)

fulldf.drop(droplist,axis=1,inplace=True)
#Removing outliers from Train data

from scipy.stats import zscore

z_scores = zscore(trdf[nunfeat])

abs_z_scores = np.abs(z_scores)

outdroplist = (abs_z_scores > 3)

trdf.drop(trdf[outdroplist].index, inplace = True)
#We need to concat both Test and Train data otherwise we will have different normalization coeficients

df_final = pd.concat([trdf,tsdf],axis=0)

#Checking overall result and DF Shape

print('Train Shape ', trdf.shape)

print('Test Shape ', tsdf.shape)

print('Full data Shape ', fulldf.shape)

print('CONCAT data Shape ', df_final.shape)
#finding skewed variables in Train data

skewed_feats = df_final[nunfeat].apply(lambda x: x.skew()).sort_values(ascending=False)

skewed_feats
#We want to sperate the target feature because we need its coeficient latter to convert it back

skewed_feats = skewed_feats.drop('SalePrice')
#Normalizing data using BOXCOX Full data

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



high_skewt = skewed_feats[skewed_feats > 0.5]

skew_index = high_skewt.index



# Normalise skewed features

for i in skew_index:

    boxcox_lambda = boxcox_normmax(df_final[i] + 1)

    print(i, ' LAMBDA: ',boxcox_lambda, '\n' )

    df_final[i] = boxcox1p(df_final[i], boxcox_lambda)
#Normalizing tagert feature

sales_boxcox_lambda = boxcox_normmax(trdf['SalePrice'] + 1)

print( ' SalesPrice LAMBDA: ',sales_boxcox_lambda, '\n' )

trdf['SalePrice'] = boxcox1p(trdf['SalePrice'], sales_boxcox_lambda)
#Plotting variables histogram

fulldf.hist(figsize=(15,15))

plt.show()
## Plot fig sizing. 

import matplotlib.style as style

style.use('ggplot')

sns.set_style('whitegrid')

plt.subplots(figsize = (30,20))

## Plotting heatmap. 



# Generate a mask for the upper triangle (taken from seaborn example gallery)

mask = np.zeros_like(trdf.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



sns.heatmap(trdf.corr(), cmap=sns.diverging_palette(20, 220, n=200), mask = mask, annot=True, center = 0, );

plt.title("Heatmap of all the Features", fontsize = 30);
#With combined DF we create the dummie columns

df_final_dummie = pd.get_dummies(df_final).reset_index(drop=True)

#Them we can split data again using the number of lines from the df

line_count = trdf.shape[0]

trdf_dummie = df_final_dummie[:line_count]

tsdf_dummie = df_final_dummie[line_count:]



#Checking overall result and DF Shape

print('Train Shape ', trdf.shape)

print('Test Shape ', tsdf.shape)

print('Dummie Train Shape ', trdf_dummie.shape)

print('Dummie Test Shape ', tsdf_dummie.shape)
#Spliting X data from Y data

# Seleção de variáveis preditoras (Feature Selection)

atributos2 = list(trdf_dummie.columns)

atributos2.remove('SalePrice')

# Selection the target feature

atrib_prev2 = ['SalePrice']

# Creating objetcs for ML

X2 = trdf_dummie[atributos2].values

Y2 = trdf[atrib_prev2].values

# Defininng the splitting rate

split_test_size = 0.30

# Creating test and train objects

from sklearn.model_selection import train_test_split

X2_treino, X2_teste, Y2_treino, Y2_teste = train_test_split(X2, Y2, test_size = split_test_size, random_state = 42)



#Printing out the Test and Train frames information

print("\n******************************************************************************")

print("Imprimindo os dados de treino e teste")

print("{0:0.2f}% nos dados de treino".format((len(X2_treino)/len(trdf_dummie.index)) * 100))

print("{0:0.2f}% nos dados de teste".format((len(X2_teste)/len(trdf_dummie.index)) * 100))

print("******************************************************************************\n")





# LINEAR REGRESSION ALGORITHM

print("\n******************************************************************************")

print("Utilizando um classificador Regressao Linear")

from sklearn import linear_model

from sklearn.metrics import r2_score, mean_squared_error

# Criando o modelo preditivo

modelo_v3 = linear_model.LinearRegression()

modelo_v3.fit(X2_treino, Y2_treino.ravel())

# Verificando os dados de treino

modelo_v3_train = modelo_v3.predict(X2_treino)

print('R2 TREINO Regressao Linear score: %.2f' % r2_score(Y2_treino, modelo_v3_train))

print('Mean Regressao Linear score: %.2f' % mean_squared_error(Y2_treino, modelo_v3_train))

# Verificando nos dados de teste

modelo_v3_test = modelo_v3.predict(X2_teste)

print('\nR2 TESTE Regressao Linear score: %.2f' % r2_score(Y2_teste, modelo_v3_test))

print('Mean TESTE Regressao Linear score: %.2f' % mean_squared_error(Y2_teste, modelo_v3_test))

print("******************************************************************************\n")



#LASSO, RIDGE & ELASTIC ALGORITHMS

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score



e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]



kfolds = KFold(n_splits=10, shuffle=True, random_state=42)



# Kernel Ridge Regression

print("\n******************************************************************************")

print("Utilizando Ridge Regression")

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))

ridge_model = ridge.fit(X2_treino, Y2_treino)

ridge_pred = ridge_model.predict(X2_teste)

print('R2 Ridge Regression %.2f' % r2_score(Y2_teste, ridge_pred))

print('Mean Ridge Regressions: %.2f' % mean_squared_error(Y2_teste, ridge_pred))

print("******************************************************************************\n")



# LASSO Regression

print("\n******************************************************************************")

print("Utilizando LASSO Regression")

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7,alphas=alphas2,random_state=42, cv=kfolds))

lasso_model = lasso.fit(X2_treino, Y2_treino)

lasso_pred = lasso_model.predict(X2_teste)

print('R2 LASSO Regression %.2f' % r2_score(Y2_teste, lasso_pred))

print('Mean LASSO Regression: %.2f' % mean_squared_error(Y2_teste, lasso_pred))

print("******************************************************************************\n")



# Elastic Net Regression

print("\n******************************************************************************")

print("Utilizando Elastic Net Regression")

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7,alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))

elastic_model = elasticnet.fit(X2_treino, Y2_treino)

el_pred = elasticnet.predict(X2_teste)

print('R2 Elastic Net Regression %.2f' % r2_score(Y2_teste, el_pred))

print('Mean Elastic Net Regression: %.2f' % mean_squared_error(Y2_teste, el_pred))

print("******************************************************************************\n")
#Creating the imput data with real data

X_tsdf_dummie=tsdf_dummie[atributos2].values



# LINEAR REGRESSION ALGORITHM

Y_linear = modelo_v3.predict(X_tsdf_dummie)

# RIDGE REGRESSION ALGORITHM

Y_ridge = ridge_model.predict(X_tsdf_dummie)

# LASSO REGRESSION ALGORITHM

Y_lasso = lasso_model.predict(X_tsdf_dummie)

# ELASTICNET REGRESSION ALGORITHM

Y_elasticnet = elasticnet.predict(X_tsdf_dummie)



#Creating a dataframe with all the 4 predicitons

tsdf2=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

#Dropping unecessary columns

drdp = list(tsdf2.columns)

tsdf2.drop(drdp[1:],axis=1,inplace = True)

tsdf2['Linear_Predict']=Y_linear

tsdf2['Ridge_Predict']=Y_ridge

tsdf2['Lasso_Predict']=Y_lasso

tsdf2['Elastic_Predict']=Y_elasticnet



#Reverting the normalization

from scipy.special import boxcox, inv_boxcox

tsdf2['Linear_Predict']=inv_boxcox(tsdf2['Linear_Predict'], sales_boxcox_lambda)

tsdf2['Ridge_Predict']=inv_boxcox(tsdf2['Ridge_Predict'], sales_boxcox_lambda)

tsdf2['Lasso_Predict']=inv_boxcox(tsdf2['Lasso_Predict'], sales_boxcox_lambda)

tsdf2['Elastic_Predict']=inv_boxcox(tsdf2['Elastic_Predict'], sales_boxcox_lambda)



#Creating a column with the mean value of the algorithms

tsdf2['SalePrice']=(tsdf2['Linear_Predict'].values+tsdf2['Ridge_Predict'].values\

                       +tsdf2['Lasso_Predict'].values+tsdf2['Elastic_Predict'].values)/4



tsdf2.sort_values(by='Id')
subdf = tsdf2.drop(['Linear_Predict','Ridge_Predict','Lasso_Predict','Elastic_Predict'],axis=1)

subdf
#Exporting the submissionn file

subdf.to_csv("/kaggle/working/submission_file.csv", header=True, index = False)