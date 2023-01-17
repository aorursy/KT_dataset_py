import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

house_data = pd.read_csv('../input/train.csv')
def column_is_numeric(series):
    return series.dtype in ['int64', 'float64']
    
def count_nas(series):
    return sum(pd.isnull(series))

def set_thousands_separator_formatting(plot, xaxis=True):
    intFormatting = lambda x, p: format(int(x), ',')
    plot.yaxis.set_major_formatter(ticker.FuncFormatter(intFormatting))
    if (xaxis):
        plot.xaxis.set_major_formatter(ticker.FuncFormatter(intFormatting))

def scatter_plot_comparison(dataframe, column1, column2, plotSize=(20, 15)):
    scatterPlot = dataframe.plot.scatter(column1, column2, figsize=plotSize)
    scatterPlot.set_title('Comparison of {} and {}'.format(column1, column2))
    set_thousands_separator_formatting(scatterPlot)
    scatterPlot.grid()
    plt.grid(True)
    
def histogram_plot(dataframe, column, plotSize=(20, 15)):
    histogramPlot = dataframe[column].plot.hist(figsize=plotSize, bins=len(dataframe[column].unique()))
    histogramPlot.set_title('Distribution of values for {}'.format(column))
    set_thousands_separator_formatting(histogramPlot)
    plt.grid(True)

def distribution_plot(dataframe, column, plotSize=(20, 15)):
    distPlot = dataframe[column].value_counts().plot.bar(figsize=plotSize)
    distPlot.set_title('Distribution of values for {}'.format(column))
    set_thousands_separator_formatting(distPlot, xaxis=False)
    plt.grid(True)

def box_plot_comparison(dataframe, by_column1, column2, plotSize=(20, 15)):
    boxPlot = dataframe[[by_column1, column2]].boxplot(by=by_column1, figsize=plotSize)
    set_thousands_separator_formatting(boxPlot, xaxis=False)
    plt.grid(True)
    
def describe_house_field(fieldName, dataframe=house_data, predictedVariable='SalePrice'):
    series = dataframe[fieldName]
    
    print('First 5 rows')
    print(series.head(5))
    
    if (column_is_numeric(series)):
        print('\nMax: {:.2f}, Min: {:.2f}, Avg: {:.2f}, StdDev: {:.2f}'.format(series.max(), series.min(), series.mean(), series.std()))
        
        scatter_plot_comparison(dataframe, fieldName, predictedVariable)
        plt.figure() # create a new figure is created for this next plot, so they don't share axes
        histogram_plot(dataframe, fieldName)
    else:
        box_plot_comparison(dataframe, fieldName, predictedVariable)
        plt.figure() # create a new figure is created for this next plot, so they don't share axes
        distribution_plot(dataframe, fieldName)
        
    count_unique = len(series.unique())
    print('\nCount of unique values: {}'.format(count_unique))
    
    if (count_unique < 50):
        print('Unique values:')
        print(dataframe[fieldName].value_counts(sort=True, ascending=True))
    
    print('\nCount of NAs: {}'.format(count_nas(series)))
# drop unbalanced variables
dropped_data = house_data.drop(['Street', 'Alley', 'LandContour', 'Utilities', 'Condition2', 'RoofMatl', 'Heating', 'LowQualFinSF', 'GarageCond', '3SsnPorch', 'PoolArea', 'PoolQC', 'MiscFeature', 'MiscVal'], axis=1)
# drop variables without much prediction value
dropped_data = dropped_data.drop(['BsmtFinType2', 'Functional', 'MoSold', 'YrSold'], axis=1)
# drop Id Column: it's already the index
dropped_data = dropped_data.drop(['Id'], axis=1)

dropped_data.shape
# Removing outliers

rows_to_drop = (dropped_data['LotFrontage'] > 150)
rows_to_drop = rows_to_drop | (dropped_data['LotArea'] > 50000)
rows_to_drop = rows_to_drop | ((dropped_data['MSZoning'] == 'RL') & (dropped_data['SalePrice'] > 400000))
rows_to_drop = rows_to_drop | (((dropped_data['Neighborhood'] == 'NoRidge') | (dropped_data['Neighborhood'] == 'NridgHt')) & (dropped_data['SalePrice'] > 575000))
rows_to_drop = rows_to_drop | (((dropped_data['OverallCond'] == 5) | (dropped_data['OverallCond'] == 6)) & (dropped_data['SalePrice']) > 500000)
rows_to_drop = rows_to_drop | ((dropped_data['OverallQual'] == 10) & (dropped_data['SalePrice'] < 200000))
rows_to_drop = rows_to_drop | ((dropped_data['YearBuilt'] < 2000) & (dropped_data['SalePrice'] > 600000))
rows_to_drop = rows_to_drop | ((dropped_data['YearRemodAdd'] < 2000) & (dropped_data['SalePrice'] > 600000))
rows_to_drop = rows_to_drop | (dropped_data['TotalBsmtSF'] > 3000)
rows_to_drop = rows_to_drop | (dropped_data['1stFlrSF'] > 3000)
rows_to_drop = rows_to_drop | (dropped_data['2ndFlrSF'] > 1575)
rows_to_drop = rows_to_drop | (dropped_data['GrLivArea'] > 4000)
rows_to_drop = rows_to_drop | (dropped_data['BedroomAbvGr'] > 5)
rows_to_drop = rows_to_drop | (dropped_data['KitchenAbvGr'] == 3)
rows_to_drop = rows_to_drop | (dropped_data['TotRmsAbvGrd'] > 10)
rows_to_drop = rows_to_drop | (dropped_data['Fireplaces'] > 2)
rows_to_drop = rows_to_drop | ((dropped_data['GarageYrBlt'] > 1980) & (dropped_data['SalePrice'] > 600000))
rows_to_drop = rows_to_drop | (dropped_data['GarageCars'] > 3)
rows_to_drop = rows_to_drop | (dropped_data['GarageArea'] > 1200)
rows_to_drop = rows_to_drop | (dropped_data['OpenPorchSF'] > 400)

filtered_data = dropped_data.copy()
filtered_data.drop(filtered_data[rows_to_drop].index, inplace=True)

filtered_data.shape
# remove the values that are meaningless, let's set them to np.nan so we can later on deal with them in the right way
data_with_nas = filtered_data.copy()

def set_value_to_na(column, dataframe=data_with_nas, current_value=0):
    num_affected_rows = dataframe[column].value_counts()[current_value]
    print('Setting {:,} values to NaN (were all {} for column {}).'.format(num_affected_rows, current_value, column))
    dataframe[column].replace(to_replace=current_value, value=np.nan, inplace=True)

set_value_to_na('YearRemodAdd', current_value=1950)
set_value_to_na('MasVnrArea')
set_value_to_na('BsmtFinSF1')
set_value_to_na('BsmtFinSF2')
set_value_to_na('BsmtUnfSF')
set_value_to_na('TotalBsmtSF')
set_value_to_na('2ndFlrSF')
set_value_to_na('GarageArea')
set_value_to_na('WoodDeckSF')
set_value_to_na('OpenPorchSF')
set_value_to_na('EnclosedPorch')
set_value_to_na('ScreenPorch')
columns_to_avoid_imputting = ['MasVnrArea', 'BsmtFinSF2', '2ndFlrSF', 'WoodDeckSF', 'EnclosedPorch', 'ScreenPorch']
# Generate columns for categorical variables
data_with_dummies = pd.get_dummies(data_with_nas)

print("{} → {}".format(data_with_nas.shape, data_with_dummies.shape))
# Missing data: replace with mean but create a "was_missing" column alongside
imputed_data = data_with_dummies.copy()
columns_with_nas = list(col for col in imputed_data.columns if imputed_data[col].isnull().any() & (col not in columns_to_avoid_imputting))

for col in columns_with_nas:
    imputed_data[col + '_was_missing'] = imputed_data[col].isnull()
    
print('Columns to imput: {}'.format(columns_with_nas))

data_to_imput = imputed_data[columns_with_nas]
from sklearn.preprocessing import Imputer
imputer = Imputer()
imputed_np = imputer.fit_transform(data_to_imput)
data_to_imput = pd.DataFrame(imputed_np, columns=data_to_imput.columns)

for col in columns_with_nas:
    imputed_data[col] = data_to_imput[col]
    
print("{} → {}".format(data_with_dummies.shape, imputed_data.shape))
fixed_data = data_with_dummies
describe_house_field(dataframe=fixed_data, fieldName='LotFrontage')
describe_house_field(dataframe=fixed_data, fieldName='YearRemodAdd')
describe_house_field(dataframe=fixed_data, fieldName='MasVnrArea')
describe_house_field(dataframe=fixed_data, fieldName='BsmtFinSF1')
describe_house_field(dataframe=fixed_data, fieldName='BsmtFinSF2')
describe_house_field(dataframe=fixed_data, fieldName='BsmtUnfSF')
describe_house_field(dataframe=fixed_data, fieldName='TotalBsmtSF')
describe_house_field(dataframe=fixed_data, fieldName='2ndFlrSF')
describe_house_field(dataframe=fixed_data, fieldName='GarageArea')
describe_house_field(dataframe=fixed_data, fieldName='WoodDeckSF')
describe_house_field(dataframe=fixed_data, fieldName='OpenPorchSF')
describe_house_field(dataframe=fixed_data, fieldName='EnclosedPorch')
describe_house_field(dataframe=fixed_data, fieldName='ScreenPorch')