# import libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Loading World Development indicators - Country data

mydata = pd.read_csv('../input/world-development-indicators/Country.csv')



# get the number of columns and rows

mydata.shape
# quick review of numeric columns of the loaded dat

mydata.columns
# get the description of the different data columns

mydata.describe()
mydata.info()
# Loading World Development indicators - Indicator data

indicators = pd.read_csv('../input/world-development-indicators/Indicators.csv')



# get the number of columns and rows

indicators.shape
# get columns name

indicators.columns
# List of Indicators included on this data

indicators['IndicatorName'].unique().shape

# There are 1344 different Indicator Names