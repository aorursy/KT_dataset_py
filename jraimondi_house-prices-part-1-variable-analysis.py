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
house_data.columns
# MSSubClass: Building class
describe_house_field('MSSubClass')
# LotFrontage: Linear feet of street connected to property
describe_house_field('LotFrontage')
# LotArea: Lot size in square feet
describe_house_field('LotArea')
# OverallQual: Overall material and finish quality
describe_house_field('OverallQual')
# MSZoning: General zoning classification
describe_house_field('MSZoning')
# Street: type of road access
describe_house_field('Street')
# Alley: Type of alley access
describe_house_field('Alley')
describe_house_field('LotShape')
describe_house_field('LandContour')
describe_house_field('Utilities')
describe_house_field('LotConfig')
describe_house_field('LandSlope')
describe_house_field('Neighborhood')
describe_house_field('Condition1')
describe_house_field('Condition2')
describe_house_field('BldgType')
describe_house_field('HouseStyle')
describe_house_field('OverallCond')
describe_house_field('OverallQual')
describe_house_field('YearBuilt')
describe_house_field('YearRemodAdd')
describe_house_field('RoofStyle')
describe_house_field('RoofMatl')
describe_house_field('Exterior1st')
describe_house_field('Exterior2nd')
describe_house_field('MasVnrType')
describe_house_field('MasVnrArea')
describe_house_field('ExterQual')
describe_house_field('ExterCond')
describe_house_field('Foundation')
describe_house_field('BsmtQual')
describe_house_field('BsmtCond')
describe_house_field('BsmtExposure')
describe_house_field('BsmtFinType1')
describe_house_field('BsmtFinSF1')
describe_house_field('BsmtFinType2')
describe_house_field('BsmtFinSF2')
describe_house_field('BsmtUnfSF')
describe_house_field('TotalBsmtSF')
describe_house_field('Heating')
describe_house_field('HeatingQC')
describe_house_field('CentralAir')
describe_house_field('Electrical')
describe_house_field('1stFlrSF')
describe_house_field('2ndFlrSF')
describe_house_field('LowQualFinSF')
describe_house_field('GrLivArea')
describe_house_field('BsmtFullBath')
describe_house_field('BsmtHalfBath')
describe_house_field('FullBath')
describe_house_field('HalfBath')
describe_house_field('BedroomAbvGr')
describe_house_field('KitchenAbvGr')
describe_house_field('KitchenQual')
describe_house_field('TotRmsAbvGrd')
describe_house_field('Functional')
describe_house_field('Fireplaces')
describe_house_field('FireplaceQu')
describe_house_field('GarageType')
describe_house_field('GarageYrBlt')
describe_house_field('GarageFinish')
describe_house_field('GarageCars')
describe_house_field('GarageArea')
describe_house_field('GarageQual')
describe_house_field('GarageCond')
describe_house_field('PavedDrive')
describe_house_field('WoodDeckSF')
describe_house_field('OpenPorchSF')
describe_house_field('EnclosedPorch')
describe_house_field('3SsnPorch')
describe_house_field('ScreenPorch')
describe_house_field('PoolArea')
describe_house_field('PoolQC')
describe_house_field('Fence')
describe_house_field('MiscFeature')
describe_house_field('MiscVal')
describe_house_field('MoSold')
describe_house_field('YrSold')
describe_house_field('SaleType')
describe_house_field('SaleCondition')