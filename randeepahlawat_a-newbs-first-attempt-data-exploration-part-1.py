# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt # for plotting graphs

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Training data will be used to help us make predictions of Sale Prices
#Test data will be used to evaluate how good we are able to make predictions
training_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
#lets have a look at our training data
#DataFrame.head() shows us first 5 rows of a Dataframe
training_data.head()
#Lets have a look at the columns of training data and test data
print('Training Data columns :: %s' % training_data.columns)
print('Test Data columns :: %s' % test_data.columns)
#Test data is missing SalePrice columns, this is what we need to predict
#scatter is used to plot a scatter plot, which is useful for analyzing the relationship between 2 features.
#here x is for x axis, y is for y axis and c is for color of the dots
#In a dataframe you can access a row like how you would access a value in a dictionary
plt.scatter(x = training_data['GrLivArea'], y = training_data['SalePrice'], c="g")
plt.xlabel("GrLiveArea")
plt.ylabel("SalePrice")
plt.show()

#From the graph we can see 4 values which seem concerning.
#Have a look at the houses which have GrLivArea > 4000. 2 of the houses have very high SalePrices. While the 2 at the botton have very low SalePrices with high GrLivArea
#Lets remove these 4
#drop method can be used to remove a row or column from a datagram. Here we select rows with GrLiveArea > 4000 by using index and remove them. Inplace is set to true
#to apply drop operation on training_data otherewise a copy is created
#You can plot the above plot again to see that the values are removed
training_data.drop(training_data[training_data.GrLivArea > 4000].index, inplace = True)
plt.scatter(x = training_data['TotRmsAbvGrd'], y = training_data['SalePrice'], c="g")
plt.xlabel("Total Rooms")
plt.ylabel("SalePrice")
plt.show()
#The plot seems to be normal as the price is tending to increase with more rooms
plt.scatter(x = training_data['YearBuilt'], y = training_data['SalePrice'], c="g")
plt.xlabel("Year Built")
plt.ylabel("SalePrice")
plt.show()
#The plot again indicates a general increase in price for newer houses
#One value seems concerning where Saleprice exceeds 400000 abd is built before 1900, lets analyze this house further
#Lets print this old house.
print(training_data[(training_data['YearBuilt'] < 1900) & (training_data['SalePrice'] > 400000)])
#So this old house has Id 186, lets see if houses priced similarly have similar GrLivArea
#loc is a dataframe method which is quite useful for querying rows of data depending on some conditional statement
print(training_data.loc[(training_data['SalePrice'] > 450000), ['GrLivArea', 'YearBuilt', 'SalePrice']])
#our suspicious house seems to have normal values wrt to GrLivArea
#Although our suspicion was wrong its better to be safe than sorry
plt.scatter(x = training_data['TotalBsmtSF'], y = training_data['SalePrice'], c="g")
plt.xlabel("Total Basement Area")
plt.ylabel("SalePrice")
plt.show()
#Everything seems normal or atleast acceptable. House prices are generally increasing with increasing TotalBsmtSF
#On second thought, it feels quite hard to estimate a house as an outlier using TotalBsmtSF
#Bottom part of the graph seems fascinating, since it indicated housed having 0 TotalBsmtSF exist which should indicate an absence of a basement
#lets store the ids, since we will remove them later
test_ids = test_data['Id']
training_ids = training_data['Id']

#lets store our training SalePrices
y = training_data['SalePrice']

#we again use drop but this time to remove a column. To remove a column we need to set axis to 1.
training_data.drop(['SalePrice'], axis = 1, inplace = True)

#concat method joins two DataFrames.
#reset_index(drop = True) is so that index isnt changed and remains the same as it was before the join.
#lets check the number of rows and columns of our training_data
print(training_data.shape)
complete_data = pd.concat([training_data, test_data]).reset_index(drop = True)

#lets check the number of rows and columns of our complete input data
print(complete_data.shape)
#We can see our original training data had 1456 houses and after combining our data now has 2915 houses.
#lets count the number of missing values in our complete_data
#isnull returns true for rows with Nan values and we can use sum method to assums true to be 1 and add them together
#the below line creates a single column dataframe called series with the count of missing values as the column and column name as index
#try to print missing_count to see how it looks
missing_count = complete_data.isnull().sum().sort_values(ascending = False)
#print(missing_count)
#count is the total count of the rows. The isnull() part might seems redundant but if we dnt include it, count method only counts non Nan values
count = complete_data.isnull().count().sort_values(ascending = False)
missing_count = pd.DataFrame({'Column' : missing_count.index, 'Missing Count' : missing_count.values, 'Missing Percent' : (missing_count.values/count) * 100})
#we remove non zero Missing count rows.
#Try to see if you can use the drop method to remove the rows
missing_count = missing_count[missing_count['Missing Count'] > 0]
#print(missing_count)
#Lets plot the percentages 
#N is the number of rows of Missing_count aka the number of columns with missing values

N = missing_count.shape[0]
#Get the percentage column
percentages = missing_count['Missing Percent']
#generate indices ie from 0 to N-1
ind = np.arange(N)
#change size
plt.figure(figsize=(50,20))
p = plt.bar(ind, percentages, width = .7)
plt.ylabel('Missing Percent')
plt.title('Missing Values of Columns')
plt.xticks(ind, missing_count['Column'])
plt.show()

print(complete_data[(complete_data['PoolQC'].isnull()) & (complete_data['PoolArea'] != 0)])

#There seems to be 3 values which have a pool area but no PoolQC
#Interestingly enough they are present in the test data (maybe a test?)
#lets see what values occur in PoolQC column
print(complete_data['PoolQC'].value_counts())

#We have 2 options either set PoolQC to None for these 3 along with the other Nan values or set it the mode 
#which is the value which occurs most frequently
#setting them to None makes no sense since PoolArea means there is a pool, so lets set them to Ex
#First lets set PoolQC to 'None' where PoolArea is 0
complete_data.loc[(complete_data['PoolQC'].isnull()) & (complete_data['PoolArea'] == 0), 'PoolQC'] = 'None'

#the above operation can be accomplished with fillna method also. Why dont you give it a try?
#lets now fill in the remaining 3 exceptions
complete_data['PoolQC'].fillna('Ex', inplace = True)
for col in ('MiscFeature', 'Alley', 'Fence'):
    complete_data[col].fillna('None', inplace = True)
print(complete_data[(complete_data['FireplaceQu'].isnull()) & (complete_data['Fireplaces'] != 0)])

#Yay!! We can simply put all FireplaceQu Nans to None
complete_data['FireplaceQu'].fillna('None', inplace = True)
#A little excercise, calculate the number of rows where LotFrontage is null, then calculate the number of rows where LotArea is not 0 and LotFrontage is null
#check if these values are same
#If you know how group by works in SQL this statement should be easy to decifer.
#Group by like tis name groups rows which have same Neighorhood and then apply fillna on each of those rows.
#The transform method sends 1 column at a time to the lambda function as x but since we have selected LotFrontage column x is just LotFrontage
#and fillna is called on LotFrontage column calculating median with respect to the rows which have same neighborhood
#This might seem hard to understand, what might it easier is to print each part 1 by 1 and see the output
complete_data['LotFrontage'] = complete_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ('GarageArea', 'GarageCars'):
    complete_data[col].fillna(0, inplace = True)
    
for col in('GarageFinish', 'GarageQual', 'GarageCond', 'GarageType'):
    complete_data.loc[(complete_data[col].isnull()) & (complete_data['GarageArea'] == 0), col] = 'None'
    
#lets print rows which have non zero GarageArea and Nan values in the above columns
print(complete_data.loc[(complete_data['GarageQual'].isnull()) & (complete_data['GarageArea'] != 0), ['GarageCond', 'GarageFinish']])

for col in('GarageQual', 'GarageCond', 'GarageFinish'):
    complete_data[col].fillna(complete_data[col].mode()[0], inplace = True)
    
#If there is no Garage, what should we set GarageYrBlt? Lets set it to 0 when GarageArea is 0
complete_data.loc[(complete_data['GarageYrBlt'].isnull()) & (complete_data['GarageArea'] == 0), 'GarageYrBlt'] = 0

#Lets check if there are any rows with non zero GarageArea and null GarageYrBlt
print("Number of GarageYrBlt Nan remaining :: %d" % complete_data[(complete_data['GarageYrBlt'].isnull()) & (complete_data['GarageArea'] != 0)].shape[0])

#Lets set this missing GarageYrBlt to the corresponding YearBuilt
complete_data.loc[complete_data['GarageYrBlt'].isnull(), 'GarageYrBlt'] = complete_data['YearBuilt']
complete_data['TotalBsmtSF'].fillna(0, inplace = True)

for col in ('BsmtFullBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtHalfBath', 'BsmtUnfSF'):
    complete_data.loc[(complete_data[col].isnull()) & (complete_data['TotalBsmtSF'] == 0), col] = 0

for col in('BsmtCond', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2'):
    complete_data.loc[(complete_data[col].isnull()) & (complete_data['TotalBsmtSF'] == 0), col] = 'None'
    
complete_data.loc[(complete_data['BsmtExposure'].isnull()) & (complete_data['TotalBsmtSF'] == 0), 'BsmtExposure'] = 'No'  

#Remaining Bsmt columns which have Nan rows, have non zero TotalBsmtSF
for col in('BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2'):
    complete_data[col].fillna(complete_data[col].mode()[0], inplace = True)
complete_data['MasVnrArea'].fillna(0, inplace = True)

complete_data.loc[(complete_data['MasVnrType'].isnull()) & (complete_data['MasVnrArea'] == 0), 'MasVnrType'] = 'None'

complete_data['MasVnrType'].fillna(complete_data['MasVnrType'].mode()[0], inplace = True)
complete_data['MSZoning'] = complete_data.groupby("Neighborhood")['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
print(complete_data['Utilities'].value_counts())

#all houses except 1 have same utilities, so its safe to drop this column
complete_data.drop(['Utilities'], axis = 1, inplace = True)
for col in('Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional'):
    complete_data[col].fillna(complete_data[col].mode()[0], inplace = True)
#Lets make a final check if any values are still missing
missing_count = complete_data.isnull().sum().sort_values(ascending = False)
print(missing_count[missing_count > 0])
#Yay we removed all missing values