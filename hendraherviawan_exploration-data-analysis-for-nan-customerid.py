# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

color = sns.color_palette()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/data.csv',encoding="ISO-8859-1",dtype={'CustomerID': str,'InvoiceID': str})

df.InvoiceDate = pd.to_datetime(df.InvoiceDate, format="%m/%d/%Y %H:%M")

df['yearmonth'] = df['InvoiceDate'].apply(lambda x: (100*x.year) + x.month)



df.info() 
label = []

values = []



for col in df.columns:

    label.append(col)

    values.append(df[col].isnull().sum())

    print(col, values[-1])

null_CustomerID = df.loc[df['CustomerID'].isnull()] 



#remove the negative values and replace with nan

null_CustomerID.loc[null_CustomerID['Quantity'] <= 0, 'Quantity'] = np.nan

null_CustomerID.loc[null_CustomerID['UnitPrice'] < 0, 'UnitPrice'] = np.nan



#get the total spent for each line item

null_CustomerID['total_dollars'] = null_CustomerID['Quantity']*null_CustomerID['UnitPrice']



null_CustomerID.describe()
notnull_CustomerID = df.loc[~df['CustomerID'].isnull()] 



#remove the negative values and replace with nan

notnull_CustomerID.loc[notnull_CustomerID['Quantity'] <= 0, 'Quantity'] = np.nan

notnull_CustomerID.loc[notnull_CustomerID['UnitPrice'] < 0, 'UnitPrice'] = np.nan



notnull_CustomerID['total_dollars'] = notnull_CustomerID['Quantity']*notnull_CustomerID['UnitPrice']

notnull_CustomerID.describe()
#

pie_data = []

pie_data.append(len(null_CustomerID))

pie_data.append(len(notnull_CustomerID))

plt.pie(pie_data, labels=['Null', 'Non Null'], autopct='%1.1f%%',)

plt.show()
# number of Null by country

null_CustomerID.groupby(['Country']).size()
# number of Null by country

notnull_CustomerID.groupby(['Country']).size()
x2312 = null_CustomerID.Quantity.describe()

notx2312 = notnull_CustomerID.Quantity.describe()

pd.DataFrame([x2312, notx2312], index=['Null', 'Not Null'])
nullyearmonth = (null_CustomerID.groupby(['yearmonth'])['Quantity'].sum())

notnullyearmonth  = (notnull_CustomerID.groupby(['yearmonth'])['Quantity'].sum())



yearmonth = pd.DataFrame([nullyearmonth , notnullyearmonth]).transpose()



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,9))



yearmonth.plot.bar(stacked=True, ax=axes[0])

yearmonth.plot.box(ax=axes[1])
# Total Sum of Quantity

pie_data = []

#pie_data.append(null_CustomerID[null_CustomerID['Quantity'] <= 32]['Quantity'].sum())

#pie_data.append(notnull_CustomerID[notnull_CustomerID['Quantity'] <= 120]['Quantity'].sum())



pie_data.append(null_CustomerID['Quantity'].sum())

pie_data.append(notnull_CustomerID['Quantity'].sum())



plt.pie(pie_data, labels=['Null', 'Non Null'], autopct='%1.1f%%',)

plt.show()
#Quantity frequency

q99n = null_CustomerID.Quantity.quantile(0.99)

q99 = notnull_CustomerID.Quantity.quantile(0.99)



#plt.figure(figsize=(14,4))

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14,9))



null_CustomerID[null_CustomerID['Quantity'] <= q99n]['Quantity'].value_counts().sort_index().plot.bar(ax=axes[0])

notnull_CustomerID[notnull_CustomerID['Quantity'] <= q99]['Quantity'].value_counts().sort_index().plot.bar(ax=axes[1])
x2312 = null_CustomerID.total_dollars.describe()

notx2312 = notnull_CustomerID.total_dollars.describe()



pd.DataFrame([x2312, notx2312], index=['Null', 'Not Null'])
x2312 = null_CustomerID.groupby(['InvoiceNo'])['total_dollars'].sum()

notx2312 = notnull_CustomerID.groupby(['InvoiceNo'])['total_dollars'].sum()



pd.DataFrame([x2312.describe(), notx2312.describe()], index=['Null', 'Not Null'])
# Total Sale

pie_data = []

#pie_data.append(null_CustomerID[null_CustomerID['Quantity'] <= 32]['Quantity'].sum())

#pie_data.append(notnull_CustomerID[notnull_CustomerID['Quantity'] <= 120]['Quantity'].sum())



pie_data.append(null_CustomerID['total_dollars'].sum())

pie_data.append(notnull_CustomerID['total_dollars'].sum())



plt.pie(pie_data, labels=['Null', 'Non Null'], autopct='%1.1f%%',)

plt.show()
nullyearmonth = (null_CustomerID.groupby(['yearmonth'])['total_dollars'].sum())

notnullyearmonth  = (notnull_CustomerID.groupby(['yearmonth'])['total_dollars'].sum())



yearmonth = pd.DataFrame([nullyearmonth , notnullyearmonth]).transpose()

total_dollars

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,9))



yearmonth.plot.bar(stacked=True, ax=axes[0])

yearmonth.plot.box(ax=axes[1])
# Number of Invoice

Labels = ['Null', 'Not Null']



pie_data = []

pie_data.append(len(null_CustomerID['InvoiceNo'].unique()))

pie_data.append(len(notnull_CustomerID['InvoiceNo'].unique()))



pd.DataFrame(pie_data , index= Labels , columns=['Number of Invoice'])
#Cek if Invoice that intersect

intersect = pd.Series(np.intersect1d(null_CustomerID['InvoiceNo'].values,notnull_CustomerID['InvoiceNo'].values))

print (intersect.values)
plt.pie(pie_data, labels=Labels , autopct='%1.1f%%',)

plt.legend(labels=Labels, loc="best")

plt.show()
nullyearmonth = (null_CustomerID.groupby(['yearmonth'])['InvoiceNo'].nunique())

notnullyearmonth  = (notnull_CustomerID.groupby(['yearmonth'])['InvoiceNo'].nunique())



yearmonth = pd.DataFrame([nullyearmonth , notnullyearmonth]).transpose()



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,9))



yearmonth.plot.bar(stacked=True, ax=axes[0])

yearmonth.plot.box(ax=axes[1])
a = pd.DataFrame({'a': [1,0,0,0,1,1,0,0,1,0,1,1,1],'b': [1,0,0,0,1,1,0,0,1,0,0,0,0]})

a.apply(pd.value_counts).plot.pie(subplots=True)