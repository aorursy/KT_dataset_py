# Code you have previously used to load data

import numpy as np

import pandas as pd

import collections

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *

pd.set_option('display.max_columns',500)

pd.set_option('display.max_rows',500)
# Set up code checking

import os

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex7 import *



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/home-data-for-ml-course/train.csv'

iowa_file_test = '../input/home-data-for-ml-course/test.csv'

hd = pd.read_csv(iowa_file_path)

td = pd.read_csv(iowa_file_test)

# Create target object and call it y

y = hd.SalePrice

td.head()
# Defining some necessary Function to work with missing values

# Defining the functions to fetch missing only fields 

def missonly (k):

    percent_missing = k.isnull().sum()*100/len(k)

    missing_value_df = pd.DataFrame({'Column_dtype': k.dtypes,'percent_missing': percent_missing.round(2)})

    missy = missing_value_df[missing_value_df['percent_missing']>0.00]

    print(missy) 

    return;



# Defining the function to fetch missing all fields

def missall (p):

    percent_missing = p.isnull().sum()*100/len(p)

    missing_value_df = pd.DataFrame({'Column_dtype': p.dtypes,'percent_missing': percent_missing.round(2)})

    print(missing_value_df) 

    return;



# Defining function to replace the fields of int8 or float types with mean

def repnum(d):

    for i in d:

        if (d[i].dtypes == 'float64') or (d[i].dtypes == 'int64'):

            print("Before change ", d[i].name , " has " , collections.Counter(d[i]), " \n ")

            d[i]=d[i].replace(np.nan,round(d[i].mean(),1))

            print("After change ", d[i].name,  " has " , collections.Counter(d[i]), " \n ")    

    return;



# defining function to fetch description, skewness, kurtosis, standard deviation and variance

def allnumbasics(d):

    for i in d:

        skew=d[i].skew()

        kurt=d[i].kurt()

        std=d[i].std()

        var=d[i].var()

        print("Here are the important stats of: "+d[i].name)

        print("Skewness of "+d[i].name+" is: "+"{0:.2f}".format(skew))

        print("Kurtosis of "+d[i].name+" is: "+"{0:.2f}".format(kurt))

        print("Standard Deviation of "+d[i].name+" is: "+"{0:.2f}".format(std))

        print("Variance of "+d[i].name+" is: "+"{0:.2f}".format(var))

        sns.distplot(d[i])

        plt.show()

        sns.scatterplot(x=d[i], y="SalePrice", data=d)

        plt.show()

    return;
# Lets get all the names of columns at first

hd.columns

td.columns
# Understanding SalesPrice -- the target variable

hd['SalePrice'].describe()
# Picturing SalesPrice histogram

sns.distplot(hd['SalePrice'])
# List out all the numeric variables in data

hdnum=hd.loc[:,hd.dtypes!=np.object]

tdnum=td.loc[:,td.dtypes!=np.object]

# Deleting Id column from numeric dataframe

hdnum=hdnum.drop(["Id"],axis=1)

tdnum=tdnum.drop(["Id"],axis=1)

#print(hdnum)



# Listing out numeric columns with missing values

# missall(hdnum)

missonly(hdnum)

missonly(tdnum)
hdnum["LotFrontage"].describe() # there are only 1201 rows of 1460 total rows

# imputing the missing values in lotfrontage

hdnum['LotFrontage'].fillna(round(hdnum['LotFrontage'].mean(),2),inplace=True)

hdnum['LotFrontage'].describe() # Checking data after imputation

# imputing the missing values in MasVnrArea 

hdnum["MasVnrArea"].describe()# there are only 1452 rows of 1460 total rows

hdnum['MasVnrArea'].fillna(round(hdnum['MasVnrArea'].mean(),2),inplace=True)

hdnum["MasVnrArea"].describe() # Checking data after imputation

# Imputing the missing values in GarageYrBlt

hdnum["GarageYrBlt"].describe() # there are only 1379 rows of 1460 total rows

# Though this is in numeric format its a year so its a categorical variable

# Lets impute this is with median than mean

hdnum['GarageYrBlt'].fillna(round(hdnum['GarageYrBlt'].median(),2),inplace=True)

hdnum["GarageYrBlt"].describe() # after replacing there are only 1460 rows of 1460 total rows



# Cleaning Test Data

''' "LotFrontage","OverallQual","YearBuilt","YearRemodAdd","MasVnrArea","BsmtFinSF1",

                  "TotalBsmtSF","1stFlrSF","2ndFlrSF","GrLivArea","FullBath",

                  "TotRmsAbvGrd","Fireplaces","GarageYrBlt","GarageCars","GarageArea",

                  "WoodDeckSF","OpenPorchSF","SalePrice" '''

tdnum["LotFrontage"].describe() # there are only 1201 rows of 1460 total rows

# imputing the missing values in lotfrontage

tdnum['LotFrontage'].fillna(round(tdnum['LotFrontage'].mean(),2),inplace=True)

tdnum['LotFrontage'].describe() # Checking data after imputation



tdnum['MasVnrArea'].describe() # there are only 1201 rows of 1460 total rows

# imputing the missing values in lotfrontage

tdnum['MasVnrArea'].fillna(round(tdnum['MasVnrArea'].mean(),2),inplace=True)

tdnum['MasVnrArea'].describe() # Checking data after imputation



tdnum['BsmtFinSF1'].describe() # there are only 1201 rows of 1460 total rows

# imputing the missing values in lotfrontage

tdnum['BsmtFinSF1'].fillna(round(tdnum['BsmtFinSF1'].mean(),2),inplace=True)

tdnum['BsmtFinSF1'].describe() # Checking data after imputation



tdnum['TotalBsmtSF'].describe() # there are only 1201 rows of 1460 total rows

# imputing the missing values in lotfrontage

tdnum['TotalBsmtSF'].fillna(round(tdnum['TotalBsmtSF'].mean(),2),inplace=True)

tdnum['TotalBsmtSF'].describe() # Checking data after imputation



# Imputing the missing values in GarageYrBlt

tdnum["GarageYrBlt"].describe() # there are only 1379 rows of 1460 total rows

# Though this is in numeric format its a year so its a categorical variable

# Lets impute this is with median than mean

tdnum['GarageYrBlt'].fillna(round(tdnum['GarageYrBlt'].median(),2),inplace=True)

tdnum["GarageYrBlt"].describe() # after replacing there are only 1460 rows of 1460 total rows



# Imputing the missing values in GarageYrBlt

tdnum["GarageCars"].describe() # there are only 1379 rows of 1460 total rows

# Though this is in numeric format its a year so its a categorical variable

# Lets impute this is with median than mean

tdnum['GarageCars'].fillna(round(tdnum['GarageCars'].median(),2),inplace=True)

tdnum["GarageCars"].describe() # after replacing there are only 1460 rows of 1460 total rows



# Imputing the missing values in GarageYrBlt

tdnum["GarageArea"].describe() # there are only 1379 rows of 1460 total rows

# Though this is in numeric format its a year so its a categorical variable

# Lets impute this is with median than mean

tdnum['GarageArea'].fillna(round(tdnum['GarageArea'].median(),2),inplace=True)

tdnum["GarageArea"].describe() # after replacing there are only 1460 rows of 1460 total rows

allnumbasics(hdnum)
# For all Numerical variables lets just plot a heatmap with correlation coefficient

sns.set(rc={'figure.figsize':(25,15)})

sns.heatmap(hdnum.corr(),annot=True,cmap= 'coolwarm')

round((hdnum.corr())["SalePrice"].sort_values(ascending = False)[1:],4)
# From above analysis lets just list out all the numerical variables with significant correlation coefficient which means correlation above 0.5

realhdNums=hdnum[["OverallQual","YearBuilt","YearRemodAdd",

                  "TotalBsmtSF","1stFlrSF","GrLivArea","FullBath",

                  "TotRmsAbvGrd","GarageCars","GarageArea","SalePrice"]].copy()

realhdNums.describe()
# From above analysis lets just list out all the numerical variables with significant correlation coefficient which means correlation above 

realtdNums=tdnum[["OverallQual","YearBuilt","YearRemodAdd",

                  "TotalBsmtSF","1stFlrSF","GrLivArea","FullBath",

                  "TotRmsAbvGrd","GarageCars","GarageArea"]].copy()

realtdNums.describe()
# For all real Numerical variables lets just plot a heatmap with correlation coefficient > = 0.3

sns.heatmap(realhdNums.corr(),annot = True,cmap='RdYlGn')
# Get Categorical Variables into a DF

hdCat = hd.loc[:,hd.dtypes==np.object]

hdCat.describe()

tdCat =td.loc[:,td.dtypes==np.object]

tdCat.describe()
# From above description we can see that there are missing values in categorical data too lets just find them and get rid of them for this iteration

missonly(hdCat)

print(hdCat.columns)

missonly(tdCat)
# Using categorical variables with less than 15% of missing data

hdCat1=hdCat.drop(["Alley","FireplaceQu","PoolQC","Fence","MiscFeature"],axis=1)

hdCat1.columns

# missall(hdCat1)

missonly(hdCat1)

# Using categorical variables with less than 15% of missing data for testing data

tdCat1=tdCat.drop(["Alley","FireplaceQu","PoolQC","Fence","MiscFeature"],axis=1)

missonly(tdCat1)
# Cleaning Categorical data with proper imputation

# Checking each variable for proper imputation

# This changes from case to case for MasVnrType use None instead of Nan assuming None/ Non available MasVnrType

hdCat1['MasVnrType'].unique()

print(collections.Counter(hdCat1['MasVnrType']))

hdCat1['MasVnrType']=hdCat1['MasVnrType'].replace(np.nan,'None')

print(collections.Counter(hdCat1['MasVnrType']))



# As Basement Qual Missing values are accounting to only 2.53 percent we are replacing them with mode value 'TA'

print("Basement Qual: ", collections.Counter(hdCat1['BsmtQual']))

hdCat1['BsmtQual']=hdCat1['BsmtQual'].replace(np.nan,'TA')

print(collections.Counter(hdCat1['BsmtQual']))



# As Basement Cond Missing values are accounting to only 2.53 percent we are replacing them with mode value 'TA'

print("Basement Cond: ", collections.Counter(hdCat1['BsmtCond']))

hdCat1['BsmtCond']=hdCat1['BsmtCond'].replace(np.nan,'TA')

print(collections.Counter(hdCat1['BsmtCond']))



# Checking BsmtFinType1 which is of data type object with percentage of missing values at 2.53%

print(collections.Counter(hdCat1['BsmtFinType1']))

hdCat1['BsmtFinType1']=hdCat1['BsmtFinType1'].replace(np.nan,'Unf')

print(collections.Counter(hdCat1['BsmtFinType1']))



# Checking BsmtFinType2 which is of data type object with percentage of missing values at 2.60%

print(collections.Counter(hdCat1['BsmtFinType2']))

hdCat1['BsmtFinType2']=hdCat1['BsmtFinType2'].replace(np.nan,'Unf')

print(collections.Counter(hdCat1['BsmtFinType2']))



# Checking Electrical which is of data type object with percentage of missing values at 0.07%

print(collections.Counter(hdCat1['Electrical']))

hdCat1['Electrical']=hdCat1['Electrical'].replace(np.nan,'SBrkr')

print(collections.Counter(hdCat1['Electrical']))



# Checking GarageType which is of data type object with percentage of missing values at 0.07%

print(collections.Counter(hdCat1['GarageType']))

hdCat1['GarageType']=hdCat1['GarageType'].replace(np.nan,'Attchd')

print(collections.Counter(hdCat1['GarageType']))



# Checking GarageType which is of data type object with percentage of missing values at 5.55%

print(collections.Counter(hdCat1['GarageFinish']))

hdCat1['GarageFinish']=hdCat1['GarageFinish'].replace(np.nan,'Unf')

print(collections.Counter(hdCat1['GarageFinish']))



# Checking GarageType which is of data type object with percentage of missing values at 5.55%

print(collections.Counter(hdCat1['GarageQual']))

hdCat1['GarageQual']=hdCat1['GarageQual'].replace(np.nan,'TA')

print(collections.Counter(hdCat1['GarageQual']))



# Checking GarageType which is of data type object with percentage of missing values at 5.55%

print(collections.Counter(hdCat1['GarageCond']))

hdCat1['GarageCond']=hdCat1['GarageCond'].replace(np.nan,'TA')

print(collections.Counter(hdCat1['GarageCond']))



# Features selected -- Treating test data only for selected features

missonly(tdCat1)



# "BsmtQual_Ex","KitchenQual_Ex","Foundation_PConc","ExterQual_Gd","ExterQual_Ex",

#                        "BsmtFinType1_GLQ","HeatingQC_Ex","GarageFinish_Fin","Neighborhood_NridgHt","SaleType_New",

#                        "SaleCondition_Partial","MasVnrType_Stone","Neighborhood_NoRidge","KitchenQual_Gd","BsmtExposure_Gd",

#                        "Exterior2nd_VinylSd","Exterior1st_VinylSd","SalePrice"



# As Basement Qual Missing values are accounting to only 2.53 percent we are replacing them with mode value 'TA'

print("Basement Qual: ", collections.Counter(tdCat1['BsmtQual']))

tdCat1['BsmtQual']=tdCat1['BsmtQual'].replace(np.nan,'TA')

print(collections.Counter(tdCat1['BsmtQual']))
# hdcat1 now has categorical data, now lets just convert the categorical variables into encoded variables with le transform 

onehotcats = pd.get_dummies(hdCat1)

print(onehotcats)

onehotcats.shape



thotcats = pd.get_dummies(tdCat1)

thotcats.columns
# adding Saleprice at the end of onehot encoded cats

onehotcat = onehotcats.assign(SalePrice=hdnum['SalePrice'])

onehotcat.dtypes

print(onehotcat.columns)



# Plotting cats vs saleprice

# sns.heatmap(onehotcat.corr(),annot = True,cmap='RdYlGn')

round((onehotcat.corr())["SalePrice"].sort_values(ascending = False)[1:],4)



# Now lets take only those columns both categorical and numeric with correlation >= .3 form single data set

selectedCat = onehotcat[["BsmtQual_Ex","KitchenQual_Ex","Foundation_PConc","SalePrice"]]

type(selectedCat)



selectedtestCat = thotcats[["BsmtQual_Ex","KitchenQual_Ex","Foundation_PConc"]]

sns.heatmap(selectedCat.corr(),annot = True,cmap='RdYlGn')
selectedCat = selectedCat.drop(["SalePrice"],axis=1)

fifeat = pd.concat([selectedCat, realhdNums], axis=1)

print(fifeat.describe())



testfeat = pd.concat([selectedtestCat,realtdNums],axis=1)
# So here is the list of features with minimum correlation with saleprice being 0.30

sns.heatmap(fifeat.corr(),annot = True,cmap='RdYlGn')
X = fifeat.drop(["SalePrice"],axis=1)

y = fifeat[["SalePrice"]]

X.columns

y.columns
# fitting data model to the train data 

# To improve accuracy, create a new Random Forest model which you will train on all training data

# rf_model_on_full_data = DecisionTreeRegressor(random_state=42)



# fit rf_model_on_full_data on all data from the training data

# rf_model_on_full_data.fit(X,y)
# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = testfeat



# make predictions which we will submit

# test_preds = rf_model_on_full_data.predict(test_X)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(X,y)

test_preds = forest_model.predict(test_X)

# print(mean_absolute_error(val_y, melb_preds))
# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them

output = pd.DataFrame({'Id': td.Id,'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
# Check your answer

step_1.check()

# step_1.solution()