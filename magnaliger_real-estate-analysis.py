# Import functions from Python Script

#from Group_1_Capstone_Project_Code.py import fillfunc, get_daterecorded, barplotfunc, dropcol
# Import necessary packages

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import seaborn as sns

import numpy as np

from scipy import stats, special

from sklearn import tree

from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
# Data import

file = 'https://data.ct.gov/api/views/5mzw-sjtu/rows.csv'

data = pd.read_csv(file, header=0, dtype={'ID': int, 'SerialNumber': int, 'ListYear': int, 'DateRecorded': str,

                                          'Town': str, 'Address': str, 'AssessedValue': float, 'SaleAmount': float,

                                          'SalesRatio': float, 'PropertyType': str, 'NonUseCode': str, 'Remarks': str})
print(data.info())
print(data.head())
data.describe(include='all')
print(data.isnull().sum())
%matplotlib inline
def fillfunc(dat, g, h):

    """

    :param g: column name which you want missing values to be filled

    :param h: variable which you want to fill the rows with

    :return: the column with missing values filled with desired value

    """

    return dat[g].fillna(h, inplace=True)





def dropcol(dat, j):

    """

    :param dat: data frame to drop column from

    :param j: column to drop

    :return: data frame with dropped column

    """

    return dat.drop(j, axis=1, inplace=True)





def get_daterecorded(getdate):

    """

    :param getdate: column to split by blank and strip white space

    :return: split column

    """

    if ' ' in getdate:

        return getdate.split(' ')[0].strip()

    else:

        return 'Unknown'





def barplotfunc(dat, colum, rows):

    """

    :param dat: data frame to plot from

    :param colum: column to plot with

    :param rows: top number of frequency counts to show

    :return: bar plot chart

    """

    plt.figure(figsize=(20, 15))

    dat[colum].value_counts().head(rows).plot.bar()

    return plt.show()
# Set index to ID

data.set_index('ID', inplace=True)



# Missing value exploration

# Count the sum of missing data for each variable

missingTotal = data.isnull().sum()

# Select only missing variables

missingExist = missingTotal[missingTotal > 0]

# Sort missing variables in descending order

missingExist.sort_values(inplace=True)

# Present the ranked data in histogram

sns.set()

f, ax = plt.subplots(figsize=(12, 9))

missingExist.plot.bar()

plt.show()
# Check missing ratio

Fac_missing = pd.DataFrame(columns=("Fac", "missing_rate"))

for i in list(data.columns):

    # Cyclic reading of characteristic indicators in data

    x = (len(data)-data[i].count())/len(data)

    # Missing rate of each feature = (total length of data table - number of non-missing indicators)/indicator

    Fac_missing = Fac_missing.append([{"Fac": i, "missing_rate": x}], ignore_index=True)

Fac_missing = Fac_missing.sort_values('missing_rate', ascending=False)

print(Fac_missing)
# Drop missing values

data.drop(['Remarks'], axis=1, inplace=True)

data = data[pd.notnull(data['SaleAmount'])]

data = data[pd.notnull(data['DateRecorded'])]
# Town Histogram

barplotfunc(data, 'Town', 10)
# Data Cleaning for Town Variable

ll = data['Town'].value_counts().index[:10].tolist()

data['Town'] = np.where(data['Town'].isin(ll), data['Town'], 'Other')
# New Town Histogram

barplotfunc(data, 'Town', 11)
# Sale Amount Boxplot by Town

data.boxplot(column='SaleAmount', by='Town', figsize=(20, 15))

plt.xticks(rotation=90)

plt.ylim(0, 3000000)

plt.show()
# Date conversion

data['Date1'] = data['DateRecorded'].str[:10]

data['Date1'] = pd.to_datetime(data['Date1'], errors='coerce')

data['Date1'].describe()
# Locate outlier

data[data['Date1'] == '2102-08-23']
# Drop outlier

data = data[data.index != 260656]

data['Date1'].describe()
# Aggregate daily average sale amount

daily = data.groupby([data['Date1']], as_index=True)['SaleAmount'].mean()
# Aggregate monthly average sale amount

monthly = daily.groupby(pd.Grouper(freq='M')).mean()
# Daily average plot

plt.figure(figsize=(20, 15))

ax = daily.plot(kind='line')

plt.show()
# Monthly average plot

plt.figure(figsize=(20, 15))

ax1 = monthly.plot(kind='line')

plt.show()
# Aggregate annual average sale amount

annual = daily.groupby(pd.Grouper(freq='Y')).mean()
# Annual average plot

plt.figure(figsize=(20, 15))

ax1 = annual.plot(kind='line')

plt.show()
# Address histogram

barplotfunc(data, 'Address', 10)
# Property type histogram

barplotfunc(data, 'PropertyType', 100)
# Residential type histogram

barplotfunc(data, 'ResidentialType', 100)
# Non use code cleanup and histogram

data['NonUseCode'] = data['NonUseCode'].str[:2]

data['NonUseCode'] = data['NonUseCode'].fillna(100).astype(int)

plt.figure(figsize=(20, 15))

data['NonUseCode'].value_counts().head(100).plot.bar()

plt.show()
# Date recorded histogram

barplotfunc(data, 'DateRecorded', 20)
# Fill missing values

fillfunc(data, 'ResidentialType', 'Single Family')

fillfunc(data, 'NonUseCode', '100')

fillfunc(data, 'Address', 'MULTI ADDRESSES')

fillfunc(data, 'PropertyType', 'Residential')
print(data.describe(include=[np.object]))
# Locating and dropping duplicate rows

print('Duplicate rows before deletion : ', sum(data.duplicated(data.columns)))
data = data.drop_duplicates(data.columns, keep='last')
# Sale amount against List year jointplot

k = sns.jointplot("ListYear", "SaleAmount", data=data, kind="reg",

                  color="m", height=7)
# Sale amount against Sales ratio jointplot

l = sns.jointplot("SalesRatio", "SaleAmount", data=data, kind="reg",

                  color="m", height=7)

l.ax_marg_x.set_xlim(0, 1000)

l.ax_marg_y.set_ylim(0, 3000000)

plt.show()
# Locate maximum Sale amount

m = data['SaleAmount'].max()

data[data['SaleAmount'] == m]
# Locate maximum Sale ratio

o = data['SalesRatio'].max()

data[data['SalesRatio'] == o]
# One hot encoding

data = pd.get_dummies(data, columns=['Town', 'PropertyType', 'ResidentialType'])
print(data.info())
# Drop unusable variables

data['Date_record'] = data['DateRecorded'].apply(get_daterecorded)

# Delete the DateRecorded variable

dropcol(data,'DateRecorded')

# Delete the Address, Date_record, Date1

dropcol(data,'Address')

dropcol(data,'Date_record')

dropcol(data,'Date1')

data = data.dropna()
# Compute correlation

corrDf = data.corr()

corrDf_Sale = pd.DataFrame(corrDf['SaleAmount'].sort_values(ascending=False))

corrDf_Sale
x_train = data.loc[:, data.columns != 'SaleAmount']

y_train = data['SaleAmount']

# Divide training sets and test sets

train_X, test_X, train_y, test_y = train_test_split(x_train, y_train, train_size=.8, random_state=3)
# Linear Regression

lm = LinearRegression()

lm.fit(train_X, train_y)

lm.score(test_X, test_y)
y_predict = lm.predict(test_X)

mse = mean_squared_error(y_predict, test_y)

rmse = np.sqrt(mse)

print(mse)

print(rmse)
scores1 = cross_val_score(lm, train_X, train_y, scoring='neg_mean_squared_error', cv=10)

rmse_scores1 = np.sqrt(-scores1)

print(rmse_scores1)

print(rmse_scores1.mean())
# Using decision trees in model

tree_reg = DecisionTreeRegressor(random_state=3)

tree1 = tree_reg.fit(train_X, train_y)

sale_mount_pre = tree_reg.predict(test_X)

tree_mse = mean_squared_error(test_y, sale_mount_pre)

tree_rmse = np.sqrt(tree_mse)

print(tree_mse)

print(tree_rmse)
scores = cross_val_score(tree_reg, train_X, train_y, scoring='neg_mean_squared_error', cv=10)

rmse_scores = np.sqrt(-scores)

print(rmse_scores)

print(rmse_scores.mean())

print(tree_reg.score(test_X, test_y))
print(dict(zip(train_X.columns, tree1.feature_importances_)))