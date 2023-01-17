import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import shapiro

from scipy.stats import anderson

from scipy.stats import normaltest

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import re

import warnings

from pandas.api.types import is_string_dtype

from pandas.api.types import is_numeric_dtype

warnings.filterwarnings('ignore')

%matplotlib inline



data = pd.read_csv('../input/data-scientist-jobs/DataScientist.csv')

data.head()
data = data.drop('Unnamed: 0', 1)

data = data.drop('index', 1)



print(data.shape)

print(data.columns)
def count_missing_values():

    for column in data:

        nullAmount = None

        if (is_numeric_dtype(data[column])):

            nullAmount = data[data[column] == -1].shape[0]

        else:

            nullAmount = data[data[column] == "-1"].shape[0]

        print('{}{},  \t{:2.1f}%'.format(column.ljust(20),nullAmount, nullAmount*100/data[column].shape[0]))

    

count_missing_values()
data = data.drop('Competitors', 1)

data = data.drop('Easy Apply', 1)
data = data.replace(-1, np.nan)

data["Rating"].interpolate(method='linear', direction = 'forward', inplace=True) 



data.drop(data[data['Headquarters'] == "-1"].index, inplace=True)

data.drop(data[data['Size'].str.contains("-1")].index, inplace=True)

data.drop(data[data['Type of ownership'].str.contains("-1")].index, inplace=True)

data.drop(data[data['Revenue'].str.contains("-1")].index, inplace=True)

print(data.shape)

count_missing_values()
data.drop(data[data['Sector'].str.contains("-1")].index, inplace=True)

data.drop(data[data['Industry'].str.contains("-1")].index, inplace=True)

print(data.shape)

count_missing_values()
data['Job Title'].value_counts()
data =  data[data['Job Title'].str.contains("Data Scientist") | data['Job Title'].str.contains("Data Analyst")]

print(data.shape)
HOURS_PER_WEEK = 40

WEEKS_PER_YEAR = 52

THOUSAND = 1000



def return_digits(x):

    result = re.findall(r'\d+', str(x))

    result = int(result[0]) if result else 0

    return result



def return_salary(string, isFrom):

    patternMain = None

    patternPerHour = None

    if(isFrom):

        patternMain = r'^\$\d+K';

        patternPerHour = r'^\$\d+';

    else:

        patternMain = r'-\$\d+K';

        patternPerHour = r'-\$\d+';

    

    result = None

    if('Per Hour' in string):

        result = re.findall(patternPerHour, str(string))

        result = return_digits(result[0]) if result else 0

        result = result * HOURS_PER_WEEK * WEEKS_PER_YEAR

    else:

        result = re.findall(patternMain, str(string))

        result = return_digits(result[0]) if result else 0

        result = result * THOUSAND

    return result



def return_average_salary(x):

    from_salary = return_salary(x, True)

    to_salary = return_salary(x, False)

    result = (from_salary+to_salary)/2

    return result



data['SalaryAverage'] =  data['Salary Estimate'].apply(return_average_salary)

print(data['SalaryAverage'].describe())

print(sns.distplot(data['SalaryAverage']))
#SalaryAverage/Rating plot

print(sns.pairplot(x_vars=["Rating"], y_vars=["SalaryAverage"],data=data,  size=5))
#SalaryAverage/Sector plot

print(sns.pairplot(x_vars=["SalaryAverage"], y_vars=["Sector"],data=data,  size=5))
#SalaryAverage/Location plot

print(sns.pairplot(x_vars=["Location"], y_vars=["SalaryAverage"],data=data,  size=5))
def return_state(string):

    patternMain = r',\s[A-Z]{2}';    

    result = re.findall(patternMain, str(string))

    if result:

        result = re.findall(r'[A-Z]{2}', str(result[0]))[0]

    else:

        result = string.split(r', ')[1]

    return result



data['State'] =  data['Location'].apply(return_state)

print(data['State'].head())

print(data['State'].value_counts())

print(sns.pairplot(x_vars=["SalaryAverage"], y_vars=["State"],data=data,  size=5))
dataBiggerSalary = data[data['State'].isin(['NY', 'NJ', 'CA'])] 

print(sns.distplot(dataBiggerSalary['SalaryAverage'], fit=norm))

print(dataBiggerSalary.shape)
from scipy.stats import norm, expon, cauchy

dataSmallerSalary = data[~data['State'].isin(['TX', 'NY', 'NJ', 'CA'])] 

print(dataSmallerSalary.shape)

print(sns.distplot(dataSmallerSalary['SalaryAverage']))
print(sns.pairplot(x_vars=["SalaryAverage"], y_vars=["State"],data=dataBiggerSalary,  size=5))

print(dataBiggerSalary.boxplot(by ='State', column =['SalaryAverage']))

print(dataBiggerSalary["SalaryAverage"].describe())
dataBiggerSalary.drop(dataBiggerSalary[dataBiggerSalary['SalaryAverage'] < 75000].index, inplace=True)

print(dataBiggerSalary.shape)

print(sns.distplot(dataBiggerSalary['SalaryAverage'], fit=norm))
def testNormality(data):

    stat, p = shapiro(data)

    print('Statistics=%.3f, p=%.3f' % (stat, p))

    alpha = 0.05

    if p > alpha:

        print('Sample looks Gaussian (fail to reject H0)')

    else:

        print('Sample does not look Gaussian (reject H0)')

        

testNormality(dataBiggerSalary['SalaryAverage'])
print(data.columns)

print(sns.countplot(y='Sector',data=data, order = data['Sector'].value_counts().index))
print(sns.countplot(y='State',data=data, order = data['State'].value_counts().index))
print(sns.countplot(y='Size',data=data, order = data['Size'].value_counts().index))

print(data["Company Name"].value_counts())
plt.figure(figsize=(15,16))

print(sns.countplot(x='Rating',data=data, order = data['Rating'].value_counts().index))