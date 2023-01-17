import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
hp = pd.read_csv('../input/train.csv')
#pd.options.display.max_columns = None
hp.info()
NULL_Data = pd.DataFrame({'Columns':hp.columns, 
                          'Null_Values' : hp.isnull().sum(),
                          'Null_Perc' : hp.isnull().sum()*100/hp.shape[0]})
NULL_Data['DType'] = [hp.iloc[:,i].dtype for i in range(hp.shape[1])]
NULL_Data[NULL_Data.Null_Values!=0]
# Summarizing the missing values >>>
NULL_Data[NULL_Data.Null_Values!=0].DType.value_counts()
# Looking at the Missing values for Numerical attributes >>>
NULL_Data[NULL_Data.DType=='float64']
pd.options.display.max_columns=None
hp_temp1 = hp._get_numeric_data()
hp_temp1 = hp_temp1[hp_temp1.LotFrontage.notnull() & hp_temp1.MasVnrArea.notnull() & hp_temp1.GarageYrBlt.notnull()].loc[:,:]
hp_temp1.drop('Id',axis=1,inplace=True)
print(hp_temp1.shape)
hp_temp1.head(2)
# This data has no null values and only numerical variables. 
# We have created this dataset to take a look at the correlation coefficients and VIF values before >>>
# we move ahead to fill the missing data. 
#Lets create and see the correlation Coeff for the three vars which have missing data >>>

corr_coeff = hp_temp1.corr()
corr_coeff.loc['SalePrice',['LotFrontage','MasVnrArea','GarageYrBlt']]
# Lets also calculate the VIF : Variation Inflation factor
# Step 1 > Divide the data into Dep and Ind vars. 
# Step 2 > Insert "Intercept" into Ind var dataframe
# Step 3 > Apply VIF algorithm on the dataset

from statsmodels.stats.outliers_influence import variance_inflation_factor 

Ind = hp_temp1.drop('SalePrice',axis=1)  # Step1
Dep = hp_temp1[['SalePrice']]            # Step1
Ind.insert(0,'Intercept',1)              # Step2
vif = pd.DataFrame({'Attributes':hp_temp1.columns})
vif['VIF'] = [variance_inflation_factor(Ind.values,i) for i in range(Ind.shape[1])]
vif.iloc[[1,7,24],:]

# P.S. >> the division of dataset into Dep and Ind vars using dmatrices algo as well. 
# We need to see how LotFrontage data is spread >>> 
plt.figure(figsize=(10,7))
plt.hist(hp[hp.LotFrontage.notnull()].LotFrontage,bins=50,width=3)
plt.xticks(np.arange(20,320,10))
plt.yticks(np.arange(0,190,5))
plt.show()
print('Mean',hp.LotFrontage.mean())
print('Median',hp.LotFrontage.median())
print('Mode',hp.LotFrontage.mode())
print('Std Deviation',hp.LotFrontage.std())
print('Min',hp.LotFrontage.min())
print('Max',hp.LotFrontage.max())
# As the dataset is completely skewed towards the mean value hence we can safely fill the missing >>>
# ...values with the mean value. 

hp.LotFrontage.replace(hp.LotFrontage[7],hp.LotFrontage.mean(),inplace=True)
# Lets check if any missing values are left in LotFrontage 
hp.LotFrontage.isnull().sum() 
# zero value means no null values left. 
# Lets see the spread of LotFrontage values again >>> 
sns.distplot(hp.LotFrontage)
plt.xticks(np.arange(0,313,20))
plt.show()
print('Mean',hp.MasVnrArea.mean())
print('Median',hp.MasVnrArea.median())
print('Mode',hp.MasVnrArea.mode())
print('Std Deviation',hp.MasVnrArea.std())
print('Min',hp.MasVnrArea.min())
print('Max',hp.MasVnrArea.max())
# Lets see the spread of "MasVnrArea" values again >>> 
plt.figure(figsize=(15,4))
sns.distplot(hp[hp.MasVnrArea.notnull()].MasVnrArea)
plt.xticks(np.arange(0,1600,50))
plt.show()
# As the data is complete skewed towards value '0' hence we can not impute missing values with the mean value. 
# Filling the missing data with '0' value. 
# Run this code before running the below code to find out the index for any NaN inside this column ==> hp[hp.MasVnrArea.isnull()].MasVnrArea

hp.MasVnrArea.replace(hp.MasVnrArea[234],0.00,inplace=True)
# Lets see the missing values again if left >>>
hp.MasVnrArea.isnull().sum() 
# zero value means no null values left. 
# Comparing the GarageYrBlt with YearBuilt to find if there is any corelation between them >>>
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.distplot(hp[hp.GarageYrBlt.notnull()].GarageYrBlt)
plt.xlabel('GarageYrBlt',size=15)
plt.subplot(1,2,2)
sns.distplot(hp[hp.YearBuilt.notnull()].YearBuilt)
plt.xlabel('YearBuilt',size=15)
plt.show()
# Showing below the correlation between YearBuilt and GarageYrBlt
corr_coeff.loc['YearBuilt','GarageYrBlt']
# Hence we are dropping this column from the dataset to avoid multicolineariy >>> 
hp.drop('GarageYrBlt',axis=1,inplace=True)
hp.drop('Id',axis=1,inplace=True)  # We do not need Id for regression model. 
hp._get_numeric_data().isnull().sum()
#Checking if any numerical vars are left with any missing values >>>
hp_temp2 = hp._get_numeric_data()
#Creating correlation coefficient matrix again >>>
corr_coeff2 = hp_temp2.corr()
# Creating VIF and Corr-Coeff table >>>
Dep = hp_temp2.drop('SalePrice',axis=1)
Dep.insert(0,'Intercept',1)
Ind = hp_temp2[['SalePrice']]
vif = pd.DataFrame({'Attribute':Dep.columns})
vif['VIF_Value'] = [variance_inflation_factor(Dep.values,i) for i in range(Dep.shape[1])]
vif.set_index('Attribute',inplace=True)  # setting index as "Attribute" to insert Corr_coeff values in the dataset
vif['Corr_coeff'] = corr_coeff2.SalePrice  # Adding Corr_coeff values
vif
# Filtering above data for corr-coeff less than (+ve)0.25 >>>
vif[vif.Corr_coeff<=0.25].sort_values(by='Corr_coeff',ascending = True).T
# With the help of above, dropping below columns which have corr_coeff between (+ve)0.25 to (-ve)0.25 >>>
hp_temp3 = hp_temp2.drop(['KitchenAbvGr','EnclosedPorch','MSSubClass','OverallCond','YrSold',
                          'LowQualFinSF','MiscVal','BsmtHalfBath','BsmtFinSF2','3SsnPorch',
                          'MoSold','PoolArea','ScreenPorch','BedroomAbvGr','BsmtUnfSF','BsmtFullBath'],
                           axis=1)
# Again running Correlation and VIF >>>
corr_coeff3 = hp_temp3.corr()
Dep = hp_temp3.drop('SalePrice',axis=1)
Dep.insert(0,'Intercept',1)
Ind = hp_temp3[['SalePrice']]
vif = pd.DataFrame({'Attribute':Dep.columns})
vif['VIF_Value'] = [variance_inflation_factor(Dep.values,i) for i in range(Dep.shape[1])]
vif.set_index('Attribute',inplace=True)  # setting index as "Attribute" to insert Corr_coeff values in the dataset
vif['Corr_coeff'] = corr_coeff3.SalePrice  # Adding Corr_coeff values
vif
# Lets visualize the Correlation Matrix >>>
plt.figure(figsize=(18,18))
sns.heatmap(corr_coeff3,annot=True,cmap='Blues')
plt.show()
# Based on the VIF table and Correlation Matirx below attributes are selected
# I have selected vars which have high corr-coeff and low VIF
# Also to avoid multi-colinearity I have selected columns by looking at their corr-coeff with other vars as well.

hp_temp4 = hp_temp3[['LotArea','YearBuilt','MasVnrArea', 'TotalBsmtSF', 
                     'FullBath', 'TotRmsAbvGrd', 'GarageArea','Fireplaces' ,'SalePrice']]
Dep.shape[1]
# Again running VIF and Correlation >>>
corr_coeff4 = hp_temp4.corr()
Dep = hp_temp4.drop('SalePrice',axis=1)
Dep.insert(0,'Intercept',1.0)
Ind = hp_temp4[['SalePrice']]
vif = pd.DataFrame({'Attribute':Dep.columns})
vif['VIF_Value'] = [variance_inflation_factor(Dep.values,i) for i in range(Dep.shape[1])]
vif.set_index('Attribute',inplace=True)  # setting index as "Attribute" to insert Corr_coeff values in the dataset
vif['Corr_coeff'] = corr_coeff4.SalePrice  # Adding Corr_coeff values
vif
plt.figure(figsize=(8,5))
sns.heatmap(corr_coeff4,annot=True,cmap='coolwarm_r')
plt.show()
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
train,test = train_test_split(hp_temp4,test_size=0.30,random_state=123)
regressor = ols(formula='SalePrice~LotArea+YearBuilt+MasVnrArea+TotalBsmtSF+FullBath+TotRmsAbvGrd+GarageArea+Fireplaces',
                data=train).fit()
regressor.summary()
test['Pred'] = round(regressor.predict(test[['LotArea','YearBuilt', 'MasVnrArea', 'TotalBsmtSF',
       'FullBath', 'TotRmsAbvGrd', 'GarageArea','Fireplaces']]),2)
test.head()
def MAPE(PP,AP):
    '''
    PP = Predicted Price
    AP = Actual Price
    '''
    Dif = (PP-AP)/AP
    ABS = np.abs(Dif)
    MEAN = np.mean(ABS)*100
    return MEAN
# MAPE : Mean Absolute Percentage Error
MAPE(test.SalePrice,test.Pred)    
#Plotting Predicted vs Actual prices ===> 
plt.scatter(test.index,test.SalePrice)
plt.scatter(test.index,test.Pred)
plt.legend(loc='upper right',
          bbox_to_anchor=(1.35,1))
plt.title('Actual vs Predicted Price',size=15)
plt.xlabel('House Id',size=15)
plt.ylabel('Price',size=15)
plt.show()
# Script to filter dataframe for object variables only ==> 
temp = hp.columns
hp_cat = pd.DataFrame()
x = 0
for i in temp:
    if hp[i].dtype == 'O':
        hp_cat.insert(x,i,hp[i])
        x=x+1
hp_cat.head().T
def encoder(in_df,out_df,Columns=[]):
    '''
    in_df = Input dataframe whose columns are coded
    out_df = output dataframe where the coded columns are inserted
    Columns = list of column headers of input dataframe which are to be coded
    ''' 
    for c in Columns:
        A = []
        x = 0
        temp = in_df[c].sort_values().unique()
        for i in temp:
            for B in np.arange(in_df[c].count()):
                if in_df[c][B] == i:
                    A.append(1)
                else:
                    A.append(0)
            out_df.insert(x,i,A)
            if i == temp[len(temp)-2]:
                break
            x = x+1
            A = []

# This encoder is soley defined by me and works on numpy and pandas only
# This works for both LabelEncoder and OneHotEncoder
# Using this, we can code the values of categorical vars as 0 & 1 and it will insert the values as >>>
# new columns in the dataset of your choice. 
# Besides this also takes care of creating only columns equal to values n-1. Hence for example >>>
# if there are 10 unique values in a column, this encoder will only create 10-1 = 9 columns. 
hp_reg = hp_temp4.copy()  # deep copy so that original dataset wont change
# As 'HouseStyle' column has some values which are starting with numbers hence we will need to code those values otherwise >>>
# our regression model will not run as it understands column title example 123Sample ==> 123 as one column and Sample as >>>
# another column.

hp_cat['HouseStyle'] = ('A'+hp_cat['HouseStyle'])
hp_cat.HouseStyle.value_counts()
# Using below script, we can run Linear Regression for all the categorical variables one-by-one by cumulating the already >>>
# added columns. Hence first round will run LR for 'CentralAir' and then in second round, the dataset will contain the coded >>>
# values for CentralAir and will run LR with adding coded values from next column i.e. 'SaleType'. 
# This script can be proved very useful to analyze the impact of each categorical variable on the LR as we are getting error >>>
# after adding every column. 
# Hence if error is increased hence we should be careful while predicting values finally using that variable. 
# With this script, we can run LR for any number of categorical variables, and it will simply give us error for the impact >>> 
# relative to each variable. 
# THIS IS AMAZING !!!

hp_reg = hp_temp4.copy()
Columns = ['MSZoning','CentralAir','SaleType','Foundation','HouseStyle']
for D in Columns:
    encoder(hp_cat,hp_reg,Columns=[D])   # Encoding the values of the columns and inserting the coded values in target dataset. 
    train,test = train_test_split(hp_reg,test_size=0.30,random_state=123)   # Splitting dataset into train and test. 
    train.columns = train.columns.str.replace('[.,(,), ]', '')     # any special characters which are present in the values.
    test.columns = test.columns.str.replace('[.,(,), ]', '')       # performing same step on test dataset as well. 
    features = "+".join(train.drop('SalePrice',axis=1).columns)    # Collecting all the columns as features to use in LR. 
    regressor = ols(formula='SalePrice~'+features,data=train).fit()# Creating model and fitting the model
    test['Pred'] = round(regressor.predict(test[train.drop('SalePrice',axis=1).columns]),2) # predicting the values
    print(D,'-->',MAPE(test.SalePrice,test.Pred))                  # calculating MeanAbsolutePercentageError (MAPE).
    

# In the above, we see that MSZoning, Foundation and HouseStyle has decreased the error. Hence we will use them again >>>
# and create the model >>> 
hp_reg = hp_temp4.copy()

Columns = ['MSZoning','Foundation','HouseStyle']
for D in Columns:
    encoder(hp_cat,hp_reg,Columns=[D])   
    train,test = train_test_split(hp_reg,test_size=0.30,random_state=123)   
    train.columns = train.columns.str.replace('[.,(,), ]', '')     
    test.columns = test.columns.str.replace('[.,(,), ]', '')     
    features = "+".join(train.drop('SalePrice',axis=1).columns)     
    regressor = ols(formula='SalePrice~'+features,data=train).fit()
    test['Pred'] = round(regressor.predict(test[train.drop('SalePrice',axis=1).columns]),2) 
    print(D,'-->',MAPE(test.SalePrice,test.Pred)) 
hp_reg = hp_temp4.copy()

Columns = ['MSZoning','Foundation']
for D in Columns:
    encoder(hp_cat,hp_reg,Columns=[D])   
    train,test = train_test_split(hp_reg,test_size=0.30,random_state=123)   
    train.columns = train.columns.str.replace('[.,(,), ]', '')     
    test.columns = test.columns.str.replace('[.,(,), ]', '')     
    features = "+".join(train.drop('SalePrice',axis=1).columns)     
    regressor = ols(formula='SalePrice~'+features,data=train).fit()
    test['Pred'] = round(regressor.predict(test[train.drop('SalePrice',axis=1).columns]),2) 
    print(D,'-->',MAPE(test.SalePrice,test.Pred)) 
regressor.summary()
#Plotting Predicted vs Actual prices ===> 
plt.scatter(test.index,test.SalePrice)
plt.scatter(test.index,test.Pred)
plt.legend(loc='upper right',
          bbox_to_anchor=(1.35,1))
plt.title('Actual vs Predicted Price',size=15)
plt.xlabel('House Id',size=15)
plt.ylabel('Price',size=15)
plt.show()
test_kaggle = pd.read_csv('../input/test.csv')
# encoder(test_kaggle,test_kaggle,Columns = ['MSZoning','HouseStyle'])
# While running the above encoder, I got the error hence I further checked the data and I found >>>
# that the MSZoning has the missing values
test_kaggle.MSZoning.isnull().sum()
test_kaggle[test_kaggle.Foundation.isnull()].Foundation
# To find if there is any missing values in "Foundation"
# In order to fill in the missing values, lets summarize the MSZoning data >>>
test_kaggle.MSZoning.value_counts() 
# Replacing missing values with "RL" values as it has the highest frequency in the dataset. 
test_kaggle.MSZoning.replace(test_kaggle.MSZoning[455],'RL',inplace=True)
test_kaggle.MSZoning.isnull().sum()  # To check if there is any missing value left. 

# I have used "455" as the row number to input the NaN value in the code because otherwise NaN value >>>
# is empty value and it cant be referenced as NaN itself in this code. 
# Instead of 455, we can provide any other row number as well wherever MSZoning has missing value >>>
# Using this code, we can find such a rownumber >>> test_kaggle[test_kaggle.MSZoning.isnull()].MSZoning
# Further I am checking below for any numerical vars in test data if there is any missing values >>>
test_kaggle[hp_temp4.drop("SalePrice",axis=1).columns].isnull().sum()
# Filling out the missing values using the same as used earlier in this project to handle the missing values >>>
test_kaggle.MasVnrArea.replace(test_kaggle.MasVnrArea[231],0.00,inplace=True)
test_kaggle.TotalBsmtSF.replace(test_kaggle.TotalBsmtSF[660],test_kaggle.TotalBsmtSF.mean(),inplace=True)
test_kaggle.GarageArea.replace(test_kaggle.GarageArea[1116],test_kaggle.GarageArea.mean(),inplace=True)
# Checking again if there is any missing values left in our numerical vars >>>
test_kaggle[hp_temp4.drop("SalePrice",axis=1).columns].isnull().sum()
# In order to predict the SalePrice in this "test" dataset, we need to encode the columns of this >>>
# dataset as well >>>
encoder(test_kaggle,test_kaggle,Columns = ['MSZoning','Foundation'])
test_kaggle.columns = test_kaggle.columns.str.replace('[.,(,), ]', '')
# Lets predict SalePrice using the regressor we have created earlier >>>
test_kaggle['SalePrice'] = round(regressor.predict(test_kaggle[train.drop('SalePrice',axis=1).columns]),2) 
test_kaggle[['Id','SalePrice']].head()
