from pandas import read_csv

from numpy import nan
#load the dataset, please make sure on your working directory

data = read_csv('../input/diamonds/diamonds.csv')



#view the dataset

data
# summarize the dataset

data.describe()
# Finding and counting 0

(data.loc[:, 'carat':'z'] == 0).sum()
# Filtering rows with column condition

data.query("x == 0 or y == 0 or z == 0")
# Replace 0 with NaN

data[['x','y','z']] = data[['x','y','z']].replace(0, nan)
# Counting NaN

data.isnull().sum()
# summarize the dataset

data.describe()