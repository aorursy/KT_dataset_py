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
train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
train_data.head()
train_data.shape
train_data.describe()
NanLocator = train_data.isnull()

for column in NanLocator.columns.values.tolist():
    print(column)
    print (NanLocator[column].value_counts())
    print("")   
# Drop the above three columns
train_data.drop (['Alley', 'PoolQC', 'Fence'], axis=1, inplace=True)
train_data['MSZoning'].value_counts().to_frame()
train_data['Street'].value_counts().to_frame()
train_data['LotShape'].value_counts().to_frame()
train_data['LandContour'].value_counts().to_frame()
train_data['Utilities'].value_counts().to_frame()
train_data['LotConfig'].value_counts(dropna=False).to_frame()
train_data['LandSlope'].value_counts().to_frame()
train_data['Neighborhood'].value_counts().to_frame()
train_data['Condition1'].value_counts().to_frame()
train_data['Condition2'].value_counts().to_frame()
train_data['BldgType'].value_counts().to_frame()
train_data['HouseStyle'].value_counts().to_frame()
train_data['RoofMatl'].value_counts().to_frame()
train_data['Exterior1st'].value_counts().to_frame()
train_data['Exterior2nd'].value_counts().to_frame()
train_data['MasVnrType'].value_counts().to_frame()
train_data['MasVnrArea'].value_counts().to_frame()
train_data['ExterQual'].value_counts().to_frame()
train_data['ExterCond'].value_counts().to_frame()
train_data['Foundation'].value_counts().to_frame()
train_data['BsmtQual'].value_counts(normalize=True, dropna=False).to_frame()
train_data['BsmtExposure'].value_counts().to_frame()
train_data['Heating'].value_counts().to_frame()
train_data['HeatingQC'].value_counts().to_frame()
train_data['CentralAir'].value_counts().to_frame()
train_data['Electrical'].value_counts().to_frame()
train_data['KitchenAbvGr'].value_counts().to_frame()
train_data['KitchenQual'].value_counts().to_frame()
train_data['Functional'].value_counts().to_frame()
train_data['Fireplaces'].value_counts().to_frame()
train_data['GarageType'].value_counts(dropna=False).to_frame()
train_data['GarageQual'].value_counts().to_frame()
train_data['PavedDrive'].value_counts().to_frame()
train_data['MiscFeature'].value_counts().to_frame()
train_data['MoSold'].value_counts().to_frame()
train_data['SaleCondition'].value_counts().to_frame()
train_data['SaleType'].value_counts().to_frame()
train_data.drop (['MSZoning', 'LotShape', 'Utilities', 'Electrical', 'Functional' , 'MiscFeature', 'SaleCondition'], axis=1, inplace=True)
train_data
import seaborn as sea
# Get  columns whose data type is object i.e. string
filteredColumns = train_data.dtypes[train_data.dtypes == np.object]
 
# list of columns whose data type is object i.e. string
listOfColumnNames = list(filteredColumns.index)
 
print(listOfColumnNames)
sea.boxplot(x='Street', y='SalePrice', data = train_data)
sea.boxplot(x='LandContour', y='SalePrice', data = train_data)
sea.boxplot(x='LotConfig', y='SalePrice', data = train_data)
sea.boxplot(x='LandSlope', y='SalePrice', data = train_data)
sea.boxplot(x='Neighborhood', y='SalePrice', data = train_data)
sea.boxplot(x='Condition1', y='SalePrice', data = train_data)
sea.boxplot(x='Condition2', y='SalePrice', data = train_data)
sea.boxplot(x='BldgType', y='SalePrice', data = train_data)
sea.boxplot(x='HouseStyle', y='SalePrice', data = train_data)
sea.boxplot(x='RoofStyle', y='SalePrice', data = train_data)
sea.boxplot(x='RoofMatl', y='SalePrice', data = train_data)
sea.boxplot(x='Exterior1st', y='SalePrice', data = train_data)
sea.boxplot(x='Exterior2nd', y='SalePrice', data = train_data)
sea.boxplot(x='MasVnrType', y='SalePrice', data = train_data)
sea.boxplot(x='Foundation', y='SalePrice', data = train_data)
sea.boxplot(x='BsmtExposure', y='SalePrice', data = train_data)
sea.boxplot(x='HeatingQC', y='SalePrice', data = train_data)
sea.boxplot(x='KitchenQual', y='SalePrice', data = train_data)
sea.boxplot(x='FireplaceQu', y='SalePrice', data = train_data)
sea.boxplot(x='GarageType', y='SalePrice', data = train_data)
sea.boxplot(x='SaleType', y='SalePrice', data = train_data)
train_data.drop(['BsmtExposure', 'BsmtFinType2', 'BsmtFinSF2', 'Heating', 'HeatingQC', 'FireplaceQu', 'SaleType'], axis=1, inplace=True)


train_data.drop (['Street', 'Condition2', 'MasVnrType', 'MasVnrArea', 'BsmtFinType1', 'BsmtFinSF1'], axis=1, inplace=True)
train_data
train_data.describe (include=['object'])
NanLocator = train_data.isnull()
for column in NanLocator.columns.values.tolist():
    print(column)
    print (NanLocator[column].value_counts())
    print("")   
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (30,10)
sea.regplot(x='LotFrontage', y='SalePrice', data=train_data)

sea.regplot(x='GarageYrBlt', y='SalePrice', data=train_data)
train_data.drop (['LotFrontage', 'GarageYrBlt'], inplace=True, axis=1)
# Get the Numeric columns

NumericCols = train_data.select_dtypes([np.number]).columns
NumericCols = NumericCols[1:]
#print (NumericCols)

# Get the Pearson Correlation
from scipy import stats
Pearson_PValue = pd.DataFrame(columns = {'ComparisonParameters', 'PearsonCoeff', 'PValue'})
for col in NumericCols :
    pearson_coeff, p_value = stats.pearsonr (train_data[col], train_data['SalePrice'])
    Pearson_PValue = Pearson_PValue.append( {'ComparisonParameters': col+'--SalePrice', 'PearsonCoeff': pearson_coeff, 'PValue':p_value}, ignore_index=True)

Pearson_PValue = Pearson_PValue[['ComparisonParameters', 'PearsonCoeff', 'PValue']]
Pearson_PValue
PearsonMatrix = train_data.corr(method='pearson')
LowImpactAttributes = {'BsmtUnfSF', '2ndFlrSF', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath',
                       'HalfBath', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'
                       }
PearsonMatrix[LowImpactAttributes]
# Now remove the columns identified in the last two steps that have a low correlation with the SalesPrice

train_data.drop ({'BsmtUnfSF', '2ndFlrSF', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath',
                       'HalfBath', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold',
                       'GrLivArea'}, axis=1, inplace=True)
train_data
# Let us run the check again on the columns that have NaN values
NanLocator = train_data.isnull()
for column in NanLocator.columns.values.tolist():
    print(column)
    print (NanLocator[column].value_counts())
    print("")       
    
# Let us get hold of just the GarageTyp

GarageType = train_data.copy()
GarageType = GarageType[['Id', 'GarageType']]
GarageType
GarageTypeNull = GarageType[GarageType['GarageType'].isnull()]
GarageTypeNull
GarageTypeNotNull = GarageType[GarageType['GarageType'].notnull()]

GarageTypeNotNull['GarageType'].value_counts(normalize=True).to_frame()
np.random.seed(100)

#DfSortKey = pd.DataFrame (np.random.random(81), columns={'Key'})

#GarageTypeNull.drop(['SortKey'],inplace=True)
#DfSortKey
#GarageTypeNull.loc[0:81,'SortKey'] = DfSortKey
#GarageTypeNull.insert(2, 'Key', 0.1)
GarageTypeNull
GarageTypeNull.loc[:,'Key'] = np.random.random(81)
GarageTypeNull
# The table is being sorted by the Random Key. This will eliminate human-bias
GarageTypeNullSorted = GarageTypeNull.sort_values (by='Key')
GarageTypeNullSorted
GarageTypeNullSorted.set_index(['Key'], inplace=True)
#GarageTypeNullSorted.drop(['index'], inplace=True)
GarageTypeNullSorted.reset_index(level=['Key'], inplace=True)
GarageTypeNullSorted
#GarageTypeNullCount = len(GarageTypeNullSorted)
#Attchd, Detchd, BuiltIn, Basment, CarPort, 2Types = np.split(GarageTypeNullSorted, [int(0.63*GarageTypeNullCount), int(0.28*GarageTypeNullCount), int(0.06*GarageTypeNullCount), int(0.01*GarageTypeNullCount), int(0.01*GarageTypeNullCount), int(0.01*GarageTypeNullCount) ])
 
#Attchd 0.63
#Detchd 0.28
#BuiltIn 0.06
#Basment 0.01
#CarPort 0.01
#2Types 0.01
NUllGarageCount = len(GarageTypeNullSorted)
# Assign "Attchd" to the first 63%
StrtPtr = 0
EndPtr = StrtPtr + int(0.63*NUllGarageCount)
GarageTypeNullSorted.loc[StrtPtr:EndPtr, 'GarageType'] ='Attchd'

#Assign Detchd to the next 28%
StrtPtr = EndPtr
EndPtr = EndPtr + int(0.28*NUllGarageCount)
GarageTypeNullSorted.loc[StrtPtr : EndPtr, 'GarageType'] ='Detchd'

#Assign BuiltIn  to the next 6%
StrtPtr = EndPtr
EndPtr = EndPtr + int(0.06*NUllGarageCount)
GarageTypeNullSorted.loc[StrtPtr : EndPtr, 'GarageType'] ='BuiltIn'

#Assign Basment  to the next 1%
StrtPtr = EndPtr
EndPtr = EndPtr + int(0.01*NUllGarageCount) +1
GarageTypeNullSorted.loc[StrtPtr : EndPtr, 'GarageType'] ='Basment'

#Assign CarPort  to the next 1%
StrtPtr = EndPtr
EndPtr = EndPtr + int(0.01*NUllGarageCount) +1
GarageTypeNullSorted.loc[StrtPtr : EndPtr, 'GarageType'] ='CarPort'

#Assign 2Types  to the next 1%
StrtPtr = EndPtr
EndPtr = EndPtr + int(0.01*NUllGarageCount)
GarageTypeNullSorted.loc[StrtPtr : , 'GarageType'] ='2Types'

GarageTypeNullSorted['GarageType'].value_counts(dropna=False).to_frame()
GarageTypeNullSorted
#train_data.loc[train_data['Id'].isin(GarageTypeNullSorted['Id']), ['GarageType']] = GarageTypeNullSorted[['GarageType']]

train_data[train_data['Id'].isin(GarageTypeNullSorted['Id'])]
train_data['GarageType'].value_counts(dropna=False).to_frame()
train_data.dropna(axis=0, how='any', inplace=True)
train_data.shape
# Get  columns whose data type is Int or Float 
filteredColumns = train_data.dtypes[train_data.dtypes != np.object]

filteredColumns
sea.regplot(x='MSSubClass', y='SalePrice', data=train_data)

sea.regplot(x='OverallQual', y='SalePrice', data=train_data)

sea.regplot(x='YearBuilt', y='SalePrice', data=train_data)
sea.regplot(x='1stFlrSF', y='SalePrice', data=train_data)

RegressionFeatures = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF',\
                      '1stFlrSF', 'FullBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars' ,\
                      'GarageArea', 'WoodDeckSF', 'OpenPorchSF']
      
         
        
#RegressionFeatures     
X_TrainFeaturesSample = train_data[RegressionFeatures]
y_TrainPriceSample    = train_data['SalePrice']

from sklearn.model_selection import train_test_split
X_Train, X_Test, y_train, y_test = train_test_split (X_TrainFeaturesSample,y_TrainPriceSample, test_size=0.25, random_state=1)
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree=2)
X_train_pr = pr.fit_transform (X_Train[RegressionFeatures])
X_Test_pr  = pr.fit_transform (X_Test [RegressionFeatures])
from sklearn.linear_model import LinearRegression
poly = LinearRegression()
poly.fit (X_train_pr, y_train)
yhat = poly.predict (X_Test_pr)
# Let us evaluate the model mathematically
poly.score(X_train_pr, y_train)
poly.score (X_Test_pr, y_test)
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sea.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sea.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Houses')

    plt.show()
    plt.close()
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data - Polynomial Degree=2'
DistributionPlot(y_test, yhat,"Actual Values (Test)","Predicted Values (Test)",Title)

test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
test_data.head()
test_data[RegressionFeatures].describe()
NanLocator = test_data[RegressionFeatures].isnull()
for column in NanLocator.columns.values.tolist():
    print(column)
    print (NanLocator[column].value_counts())
    print("")  
NaNColumnList = test_data.columns[test_data.isna().any()].tolist()
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 
print (intersection (RegressionFeatures, NaNColumnList))
MeanBsmtSF = test_data['TotalBsmtSF'].mean(skipna=True)
# test_data[test_data['TotalBsmtSF'].isna()]['TotalBsmtSF'] =MeanBsmtSF
test_data.loc[660, 'TotalBsmtSF'] =MeanBsmtSF

MedianCars = test_data['GarageCars'].median(skipna=True)

#test_data[test_data['GarageArea'].isna()]

#test_data[test_data['GarageCars'].isna()]
test_data.loc[1116, 'GarageCars'] = MedianCars
MeanGarageArea = test_data['GarageArea'].mean(skipna=True)
#test_data[test_data['GarageArea'].isna()]
test_data.loc[1116, 'GarageArea'] = MeanGarageArea

X_TestData_pr = pr.fit_transform(test_data[RegressionFeatures])
yhatTestData = poly.predict(X_TestData_pr)
yhatTestData
output = pd.DataFrame({'Property Id': test_data.Id, 'Predicted Sale Price': yhatTestData})
output
output.to_csv('my_submission.csv', index=False)
print("Submission was successfully saved!")