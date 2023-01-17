import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline

np.set_printoptions(suppress=True)
path = '../input/walmart-recruiting-trip-type-classification/'

data = pd.read_csv(path + 'train.csv.zip')
print('The number of samples {}'.format(data.shape))

data.head(5)
print('The unique value of the data {}'.format(data[['VisitNumber']].nunique()))

print('The number of the value each VisitNumber: \n{}'.format(data['VisitNumber'].value_counts().sort_values(ascending = False).head(5)))
data['TripType'].value_counts()
plt.figure(figsize = (12, 10))



sns.set_style('whitegrid')

ax = sns.countplot(x = 'TripType', data = data, palette = 'mako')

ax.set(title = 'The Frequent of Trip Type', ylabel = 'Counts', xlabel = 'Trip Type')
plt.figure(figsize = (12, 10))



sns.set_style('whitegrid')

ax1 = sns.countplot(x = 'Weekday', data = data, palette = 'mako')

ax.set(title = 'The Frequent of Weekday', ylabel = 'Counts', xlabel = 'Weekday')
plt.figure(figsize = (32, 10))



sns.set_style('whitegrid')

ax1 = sns.countplot(x = 'TripType', hue = 'Weekday', data = data, palette = 'mako')

ax.set(title = 'The Frequent of Weekday', ylabel = 'Counts', xlabel = 'Weekday')
data.groupby(['Weekday'])['ScanCount'].sum().plot.bar()
print('The types of goods are {}'.format(data['DepartmentDescription'].nunique()))

data['DepartmentDescription'].unique()
plt.figure(figsize = (30, 10))



sns.set_style('whitegrid')

ax1 = sns.countplot(x = 'DepartmentDescription', data = data, palette = 'mako')

plt.xticks(rotation = 90)

plt.xlabel('Department Description', fontsize = 15)

plt.ylabel('Counts', fontsize = 15)

plt.title('The Frequent of Department Description', fontsize = 15)
data['DepartmentDescription'].value_counts()
total = data.isnull().sum().sort_values(ascending = False)

print(total)

percentage = total / data.shape[0]

print('Percentage'.center(50, '-'))

print(percentage)

missingData = pd.concat([total, percentage], axis = 1, keys = ['Total', 'Percentage'])

missingData
data['Upc'].unique().tolist()[:10]
data.info()
data.select_dtypes(include = ["object"]).columns
def flot_to_str(obj):

    """

    Convert Upc code from float to string.

    Use this function by applying lambda

    Parameters: "Upc" column of DataFrame

    Return:string converted Upc removing dot

    """

    while obj != 'np.nan':

        obj = str(obj).split('.')[0]

        if len(obj) == 10:

            obj = obj + '0'

        elif len(obj) == 4:

            obj = obj + '0000000' 

        return obj
def company(upcData):

    """

    Return company code from given Upc code.

    Parameters:'Upc' column of DataFrame

    Return: company code

    """

    try:

        code = upcData[: 6]

        if code == '000000':

            return x[-5]

        return code

    except:

        return -9999
def prodct(upcData):

    """

    Return company code from given Upc code.

    Parameters:'Upc' column of DataFrame

    Return: company code

    """

    try:

        code = upcData[6 :]

        return code

    except:

        return -9999
data['handled_Upc'] = data['Upc'].apply(flot_to_str)
data['company_code'] = data['handled_Upc'].apply(company)
data['product_code'] = data['handled_Upc'].apply(prodct)
data['DepartmentDescription'].nunique()
data.drop(['Upc'], axis = 1, inplace = True)
data.drop(['handled_Upc'], axis = 1, inplace = True)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder



dummy_data = pd.get_dummies(data[['Weekday']])
data = pd.concat([data, dummy_data], axis = 1)
print('The number of ScanCount {}'.format(data['ScanCount'].nunique()))

data['ScanCount'].unique()
data['ScanCount'].value_counts().to_frame()
data['ScanCount_bool'] = 1

data.loc[data['ScanCount'] < 1, 'ScanCount_bool'] = 0

data['ScanCount_bool'].value_counts()
data['temp_ScanCount'] = data['ScanCount']

data.loc[data['ScanCount'] < 0, 'temp_ScanCount'] = 0

data['number_ScanCount'] = pd.cut(data['temp_ScanCount'], 3, labels = ['low', 'median', 'high'])

concatData = pd.get_dummies(data['number_ScanCount'])

data = pd.concat([data, concatData], axis = 1)
data.drop(['temp_ScanCount', 'ScanCount_bool'], axis = 1, inplace = True)
data['number_ScanCount'].value_counts().to_frame()
data['FinelineNumber'].value_counts().sort_values(ascending = False).to_frame()
plt.figure(figsize = (72, 10))



sns.set_style('whitegrid')

ax2 = sns.countplot(x = 'DepartmentDescription', data = data, palette = 'mako')



plt.show()
plt.figure(figsize = (12, 10))

sns.set_style('whitegrid')



ax3 = sns.stripplot(x = 'Weekday', y = 'TripType', data = data.loc[data['TripType'] < 999], palette = 'mako')

ax3.set(title = 'The Correlation with Weekday and TripTypes', xlabel = 'Weekday', ylabel = 'Trip Types')
plt.figure(figsize = (12, 10))

sns.set_style('dark')



ax4 = sns.stripplot(x = 'Weekday', y = 'FinelineNumber', data = data, palette = 'mako')

# ax4.set(title = 'The relationship between the Weekday and FinelineNumber', xlabel = 'Week Day', ylabel = 'FinelineNumber')

plt.title('The relationship between the Weekday and FinelineNumber', fontsize = 15)

plt.xticks(rotation = 45)

plt.xlabel('Week Day')

plt.ylabel('FinelineNumber')
print('The unique of VisitNumber is {}'.format(data['VisitNumber'].nunique()))

data.DepartmentDescription.nunique()
print('The missing data information'.center(50, '-'))

data.isnull().sum().sort_values(ascending = False)
data['DepartmentDescription'].fillna( 'None', inplace = True)
data['FinelineNumber'].fillna(data['FinelineNumber'].mean(), inplace = True)
data.isnull().sum().sort_values(ascending = False).to_frame()
tempData1 = pd.get_dummies(data[['DepartmentDescription']])

data = pd.concat([data, tempData1], axis = 1)
def deleteNan(datas):

    """

    Delete the 'nan' value of columns

    Parameters: datas is the data to delete.

    Return: cleaned data

    """

    datas == 'nan'

    datas = np.nan

    return datas
# columns = data.columns.tolist()

indexList = []

# columns = ['company_code', 'product_code']

columns = ['company_code']

for column in columns:

#     for index in range(data.shape[0]):

    indexList = data.loc[range(data.shape[0]), column] == 'nan'
data.loc[indexList, column] = '000000'
indexList.value_counts()
indexList = []

columns = ['product_code']

for column in columns:

    indexList = data.loc[range(data.shape[0]), column] == 'nan'
data.loc[indexList, column] = '000000'
data['company_code'].value_counts().sort_values(ascending = False)
data[['product_code']].sample(10)
data.loc[data['product_code'] == '', 'product_code'] = '00000'
data['product_code'].value_counts().sort_values(ascending = False).head()
data.info()
objectData = data.select_dtypes(include = ['object', 'category']).head()
objectData.columns.tolist()
data.drop(['Weekday', 'DepartmentDescription','number_ScanCount'], axis = 1, inplace = True)
print('The data information'.center(50, '-') + '\n')

print(data.shape)

data.sample(3)