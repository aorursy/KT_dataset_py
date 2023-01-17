import pandas as pd

import matplotlib.pyplot as plt

%pylab inline

pylab.rcParams['figure.figsize'] = (15, 6)

data = pd.read_excel('../input/obes-phys-acti-diet.xls')
data
xls_file = pd.ExcelFile('../input/obes-phys-acti-diet.xls')

xls_file.sheet_names
col_name = ['Year', 'Total', 'Males', 'Females', 'Nan', 'None']

type(col_name)
df1 = pd.read_excel(xls_file, '7.1',skipfooter=14, skiprows=[0,1,2,3,4,5], names= col_name)

df1
df1 =df1.drop(['Nan', 'None'], axis=1)
print(df1)
df1.dropna()
df1.set_index('Year')
plt.scatter(df1.Males, df1.Females)

plt.xlabel('Male')

plt.ylabel('Female')

plt.scatter(df1.Year, df1.Females,color='r')

plt.bar(df1.Year, df1.Males)

# Read 2nd section, by age by skipping first four rows and skipfooter as 14

data_age = pd.read_excel(xls_file, '7.2',skipfooter=14, skiprows=[0,1,2,3])

data_age
# Rename unames to year

print(data_age.columns)

data_age.index
data_age = data_age.rename(columns={'Unnamed: 0':'Years'})
data_age = data_age.drop(['Unnamed: 10','Unnamed: 11'], axis=1)

data_age
data_age = data_age.set_index('Years')

print(data_age.isnull().sum())

data_age.dropna()
data_age.plot()
data_age[data_age.columns.difference(['Total'])].plot()
#data_age.plot(data_age['Under 16'])Under 16

data_age[["Under 16", "25-34"]].plot()