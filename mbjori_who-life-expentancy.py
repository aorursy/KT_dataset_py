# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
expectancyData = pd.read_csv('/kaggle/input/life-expectancy-who/Life Expectancy Data.csv')
expectancyData.info(verbose = True)
expectancyData.head()
expectancyData.describe()
#Checking if the columns have null values - replacing it with appropriate values /dropping rows decision
expectancyData.isnull().sum()
expectancyDataNullDropped = expectancyData.dropna()
print("Data reduced percentage : ", np.round(expectancyDataNullDropped.shape[0] / expectancyData.shape[0] * 100))
# DataFrame for the columns with null values sorted in ascending order
data = expectancyData.isnull().sum()
columnsToReplace = pd.DataFrame({'Variable' : data.index, 'Values' : data.values}).sort_values('Values')
# Following columns contains the null values -
# The column 'Values' shows the count of null values in that particular column
columnsToReplace = columnsToReplace[columnsToReplace['Values'] != 0]
columnsToReplace
# The integer assigning of the column is useful when the rms value is calculated - it helps to assign that t value using  these column encoding
cols = expectancyData.columns.str.strip()
colsIndex = {}
for i in range(0,len(cols)):
    colsIndex[cols[i]] = i
colsIndex
expectancyData.columns = expectancyData.columns.str.strip()
fillExpectancyData = expectancyData

# For each column wihich has null values
for i in range(columnsToReplace.shape[0]):
    # Removing spaces from the column header
    currentCol = (columnsToReplace.iloc[i][0]).strip()
    rowsWithNullColValue = expectancyData[expectancyData[currentCol].isnull()]

    # For each record with considered column with null value
    for i in range (0,rowsWithNullColValue.shape[0]):
        
        expectancyDataNullDropped = fillExpectancyData.dropna()
        
        year = rowsWithNullColValue.iloc[i][1]
        status = rowsWithNullColValue.iloc[i][2]
        country = rowsWithNullColValue.iloc[i][0]

        # Removing records with same country name and considering countries with same year and status for calculation
        masterData = expectancyDataNullDropped[expectancyDataNullDropped['Year'] == year]
        masterData = masterData[masterData['Country'] != country]
        masterData = masterData[masterData['Status'] == status]
        
        # If no rows match the filter, meaning no rms can be done with none data
        if masterData.shape[0] == 0:
            continue

        # considered row conversion to numpy
        testData = rowsWithNullColValue.iloc[i,:].dropna()
        testData = pd.DataFrame({'Symbols': testData.index, 'Values' : testData.values})
        testData = testData.set_index('Symbols').T

        
        colsToConsider = testData.columns
        colsArray = colsToConsider.to_numpy()

        newMasterData = masterData[[str(i) for i in colsArray]]
        newMasterData = newMasterData.drop('Status',axis = 1)

        testData = testData.drop(['Country', 'Year','Status'], axis = 1)    

        # Extracting country column from train data for later use
        trainCountryData = newMasterData[['Country']]

        # Dropping it from train data - for calculation
        newMasterData = newMasterData.drop(['Country','Year'], axis = 1)

        # RMS value calculation between the train data and test data
        testDataNumpy = testData.to_numpy()
        newMasterDataNumpy = newMasterData.to_numpy()
        colsCount = newMasterDataNumpy.shape[1]

        rmsDiff = np.sum((newMasterDataNumpy - testDataNumpy)**2, axis =1) / colsCount
        
        # Finding the closest value index to the testData
        minIndex = rmsDiff.argmin()

        # Considering the corresponding minimum value to insert into main dataframe 
        vals = fillExpectancyData[fillExpectancyData['Country'] == country]
        ind = vals[vals['Year'] == year]
        
        # Replacing null value with the calculated value
        fillExpectancyData.iloc[ind.index,colsIndex[currentCol]] = masterData.iloc[minIndex,colsIndex[currentCol]]

# See how many records were converted from null to some finite value
fillExpectancyData.isnull().sum()
# EDA
fillExpectancyData.info()
fillExpectancyData.head()
fillExpectancyData.describe()
fillExpectancyData.info()
fillExpectancyData.head()
# 1. What is average life expectancy across the years
lifeExpMean = fillExpectancyData.groupby('Year').mean()[:]['Life expectancy']

lifeExpMean = pd.DataFrame({'Year' : lifeExpMean.index, 'Life Expectancy' : lifeExpMean.values})

import seaborn as sns
sns.set(style="darkgrid")

g = sns.lineplot(x = 'Year', y = 'Life Expectancy', data = lifeExpMean)
# 2. Average life expectancy for developing and developed nations?

statusLifeExp = fillExpectancyData.groupby(['Year', 'Status'])
statusLifeExp = statusLifeExp.mean()[:]['Life expectancy']

# statusLifeExp = pd.DataFrame({'Year': statusLifeExp.index, 'Life Expectancy': statusLifeExp.values})
statusLifeExp = pd.DataFrame(statusLifeExp)
statusLifeExp.reset_index(level = [0,1], inplace = True)
# statusLifeExp
g = sns.catplot(x = 'Year', y = 'Life expectancy', hue = 'Status', data = statusLifeExp, kind = 'bar')
g.set_xticklabels(rotation=60)
# 3. Top 5 countries with least life expectancy

leastLifeExp = fillExpectancyData.groupby('Country').mean().sort_values(['Life expectancy'])

leastLifeExp = leastLifeExp[0:5][:]
leastLifeExp = leastLifeExp.reset_index(level = [0])
g = sns.barplot(x = leastLifeExp[:]['Country'], y = leastLifeExp[:]['Life expectancy'])
g.set_xticklabels(g.get_xticklabels(), rotation=60)
# 4. Top 5 countries with most life expectancy

leastLifeExp = fillExpectancyData.groupby('Country').mean().sort_values(['Life expectancy'], ascending = False)

leastLifeExp = leastLifeExp[0:5][:]
leastLifeExp = leastLifeExp.reset_index(level = [0])
g = sns.barplot(x = leastLifeExp[:]['Country'], y = leastLifeExp[:]['Life expectancy'])
g = g.set_xticklabels(g.get_xticklabels(), rotation=60)
# 5. Is alcohol a factor in life expectancy difference between developing and developed countires?

statusLifeExp = fillExpectancyData.groupby(['Year', 'Status'])
lifeExpAlcoholMean = statusLifeExp.mean()[:][['Alcohol', 'Life expectancy']]

# # statusLifeExp = pd.DataFrame({'Year': statusLifeExp.index, 'Life Expectancy': statusLifeExp.values})
lifeExpAlcoholMean = pd.DataFrame(lifeExpAlcoholMean)
lifeExpAlcoholMean.reset_index(level = [0,1], inplace = True)


lifeExpAlcoholMean
g= sns.scatterplot(x = 'Alcohol', y = 'Life expectancy', hue = 'Status', data = lifeExpAlcoholMean)
# 6. How is the percentage expenditure, total expenditure in countries with most and least life expectancy?
totalExpMean = fillExpectancyData.groupby(['Year', 'Status']).mean()
totalExpMean = totalExpMean[:]['Total expenditure']
totalExpMean = pd.DataFrame(totalExpMean)

totalExpMean.reset_index(level = [0,1], inplace = True)

g = sns.barplot(x = 'Year', y = 'Total expenditure', hue = 'Status', data= totalExpMean)
g = g.set_xticklabels(g.get_xticklabels(), rotation = 60)
# 7. Adult mortality vs life expectancy between countries (ideally should be linearly related)
g= sns.scatterplot(x = 'Life expectancy', y = 'Adult Mortality', hue = 'Status', data = fillExpectancyData)
# 8. GDP of countries - developed vs developing
g = sns.scatterplot(x = 'GDP', y = 'Life expectancy', data = fillExpectancyData)
# Dropping the rows which still has null values
fillExpectancyDataNullDropped = fillExpectancyData.dropna()
# Dropping 'Year' column as it is not necessary in linear regression
# Also, encoding 'Status' column to binary values

# fillExpectancyDataNullDropped
fillExpectancyDataNullDropped = fillExpectancyDataNullDropped.drop(labels = ['Year'], axis =1)
fillExpectancyDataNullDropped
# Getting codes for categorical variable 'Status' and then deleting the column from the dataframe
vals = fillExpectancyDataNullDropped['Status'].astype('category').cat.codes
fillExpectancyDataNullDropped.loc[:,'Status Code'] = vals.values
fillExpectancyDataNullDropped = fillExpectancyDataNullDropped.drop(['Status'], axis = 1)
fillExpectancyDataNullDropped
# Linear Regression model

countryData = fillExpectancyDataNullDropped['Country']

fillExpectancyDataNullDropped = fillExpectancyDataNullDropped.drop(['Country'], axis = 1)
y = fillExpectancyDataNullDropped['Life expectancy']

fillExpectancyDataNullDropped = fillExpectancyDataNullDropped.drop(['Life expectancy'], axis = 1)
X = fillExpectancyDataNullDropped
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)
ypred = reg.predict(X_test)
RSS = np.sum((ypred - y_test)**2 )
yBar = y.mean()
TSS = np.sum((y - yBar)**2)
R2 = 1 - RSS/TSS

print (f'The R squared value  = {R2}')
# F-statistic calculation

n = X.shape[0]
p = X.shape[1]

F = (((TSS - RSS)/ p)/ (RSS / (n-p-1)))
print (f'The value of f-statistic = {F}')
# Following code starts with selecting single column and goes to selecting all columns for prediction and sees which set of columns has least error and largest f-statistic value 
for i in range(1,X.shape[1]):
    X_train, X_test, y_train, y_test = train_test_split(X.iloc[:,0:i], y, test_size = 0.3)
    reg = LinearRegression().fit(X_train, y_train)
    ypred = reg.predict(X_test)
    RSS0 = np.sum((ypred - y_test)**2)
    
    n = X.shape[0]
    p = X.shape[1]
    q = p - X_train.shape[1]

    F = (((RSS0 - RSS)/ q)/ (RSS / (n-p-1)))
    print (f'RSS value of {i} columns = {RSS0}')
    print (f'The value of f-statistic = {F}')
    print (f'The RMSE value = {RSS0 / X.shape[0]}\n')
