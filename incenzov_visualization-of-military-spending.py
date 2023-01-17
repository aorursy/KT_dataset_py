import numpy as np 

import pandas as pd 

pd.set_option('display.max_columns', 5000)

pd.set_option('display.max_rows', 5000)

pd.set_option('display.float_format', lambda x: '%.3f' % x)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
df_raw = pd.read_csv('/kaggle/input/military-expenditure-of-countries-19602019/Military Expenditure.csv')

df_raw.head()
df_raw.describe()
df_raw.columns
cols = ['Name', '1969', '1970', '1971',

       '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980',

       '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989',

       '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998',

       '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007',

       '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016',

       '2017', '2018']



df = df_raw[cols]
df = df.dropna()

df = df.reset_index(drop=True)

df.describe(include='all')
df
df = df[df.Name != 'Arab World']

df = df[df.Name != 'Early-demographic dividend']

df = df[df.Name != 'Euro area']

df = df[df.Name != 'European Union']

df = df[df.Name != 'High income']

df = df[df.Name != 'IDA blend']

df = df[df.Name != 'Latin America & Caribbean (excluding high income)']

df = df[df.Name != 'Latin America & Caribbean']

df = df[df.Name != 'Lower middle income']

df = df[df.Name != 'Middle East & North Africa']

df = df[df.Name != 'Middle East & North Africa (excluding high income)']

df = df[df.Name != 'North America']

df = df[df.Name != 'OECD members']

df = df[df.Name != 'Post-demographic dividend']

df = df[df.Name != 'South Asia']

df = df[df.Name != 'Sub-Saharan Africa (excluding high income)']

df = df[df.Name != 'Sub-Saharan Africa']

df = df[df.Name != 'Latin America & the Caribbean (IDA & IBRD countries)']

df = df[df.Name != 'Middle East & North Africa (IDA & IBRD countries)']

df = df[df.Name != 'South Asia (IDA & IBRD)']

df = df[df.Name != 'Sub-Saharan Africa (IDA & IBRD countries)']

df = df[df.Name != 'South Africa']

df = df[df.Name != 'Pre-demographic dividend']

df = df[df.Name != 'Heavily indebted poor countries (HIPC)']

df = df[df.Name != 'Costa Rica']

df = df[df.Name != 'Iceland']

viz1 = df.copy()

viz1['Total'] = viz1.sum(axis=1)

viz1 = viz1[['Name', 'Total']]



name = viz1['Name']

total = viz1['Total']



plt.figure(figsize=(15, 20))

plt.barh(name, total)

plt.title('Total Military Spending for The Past 50 Years')

plt.xlabel('Amount')

plt.show()

viz2 = df.copy()

viz2 = viz2[['Name', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978']]

viz2['Total1'] = viz2.sum(axis=1)



name = viz2['Name']

total1 = viz2['Total1']



plt.figure(figsize=(15, 20))

plt.barh(name, total1)

plt.title('Total Military Spending Between 1969 - 1978')

plt.xlabel('Amount')

plt.show()

viz3 = df.copy()

viz3 = viz3[['Name', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988']]

viz3['Total2'] = viz3.sum(axis=1)



name = viz3['Name']

total2 = viz3['Total2']



plt.figure(figsize=(15, 20))

plt.barh(name, total2)

plt.title('Total Military Spending Between 1979 - 1988')

plt.xlabel('Amount')

plt.show()

viz4 = df.copy()

viz4 = viz4[['Name', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998']]

viz4['Total3'] = viz4.sum(axis=1)



name = viz4['Name']

total3 = viz4['Total3']



plt.figure(figsize=(15, 20))

plt.barh(name, total3)

plt.title('Total Military Spending Between 1989 - 1998')

plt.xlabel('Amount')

plt.show()
viz5 = df.copy()

viz5 = viz5[['Name', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008']]

viz5['Total4'] = viz5.sum(axis=1)



name = viz5['Name']

total4 = viz5['Total4']



plt.figure(figsize=(15, 20))

plt.barh(name, total4)

plt.title('Total Military Spending Between 1999 - 2008')

plt.xlabel('Amount')

plt.show()
viz6 = df.copy()

viz6 = viz6[['Name', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']]

viz6['Total5'] = viz6.sum(axis=1)



name = viz6['Name']

total5 = viz6['Total5']



plt.figure(figsize=(15, 20))

plt.barh(name, total5)

plt.title('Total Military Spending Between 2009 - 2018')

plt.xlabel('Amount')

plt.show()
viz7 = df.copy()



total1_lst = []

for x in total1:

    total1_lst.append(x)

viz7['69-78'] = total1_lst



total2_lst = []

for x in total2:

    total2_lst.append(x)

viz7['79-88'] = total2_lst



total3_lst = []

for x in total3:

    total3_lst.append(x)

viz7['89-98'] = total3_lst



total4_lst = []

for x in total4:

    total4_lst.append(x)

viz7['99-08'] = total4_lst



total5_lst = []

for x in total5:

    total5_lst.append(x)

viz7['09-18'] = total5_lst



total_lst = []

for x in total:

    total_lst.append(x)

viz7['Total'] = total_lst



viz7 = viz7[['Name', '69-78', '79-88', '89-98', '99-08', '09-18', 'Total']]

viz7 = viz7.sort_values(by='Total', ascending=False)

viz7 = viz7.head(10)



row_data_lst = []

for index, row in viz7[['69-78', '79-88', '89-98', '99-08', '09-18']].iterrows():

    row_data = [row['69-78'], row['79-88'], row['89-98'], row['99-08'], row['09-18']]

    row_data_lst.append(row_data)

    

row_name_lst = []

for index, row in viz7[['Name']].iterrows():

    row_name = [row['Name']]

    row_name_lst.append(row_name)



x = 0

y = 0

z = 0

while x != len(row_data_lst):

    plt.figure(figsize=(25,15))

    plt.plot(['69-78', '79-88', '89-98', '99-08', '09-18'], row_data_lst[y], label = row_name_lst[z])

    z += 1

    y += 1

    x += 1

    plt.plot(['69-78', '79-88', '89-98', '99-08', '09-18'], row_data_lst[y], label = row_name_lst[z])

    z += 1

    y += 1

    x += 1

    plt.plot(['69-78', '79-88', '89-98', '99-08', '09-18'], row_data_lst[y], label = row_name_lst[z])

    z += 1

    y += 1

    x += 1

    plt.plot(['69-78', '79-88', '89-98', '99-08', '09-18'], row_data_lst[y], label = row_name_lst[z])

    z += 1

    y += 1

    x += 1

    plt.plot(['69-78', '79-88', '89-98', '99-08', '09-18'], row_data_lst[y], label = row_name_lst[z])

    z += 1

    y += 1

    x += 1

    plt.plot(['69-78', '79-88', '89-98', '99-08', '09-18'], row_data_lst[y], label = row_name_lst[z])

    z += 1

    y += 1

    x += 1

    plt.plot(['69-78', '79-88', '89-98', '99-08', '09-18'], row_data_lst[y], label = row_name_lst[z])

    z += 1

    y += 1

    x += 1

    plt.plot(['69-78', '79-88', '89-98', '99-08', '09-18'], row_data_lst[y], label = row_name_lst[z])

    z += 1

    y += 1

    x += 1

    plt.plot(['69-78', '79-88', '89-98', '99-08', '09-18'], row_data_lst[y], label = row_name_lst[z])

    z += 1

    y += 1

    x += 1

    plt.plot(['69-78', '79-88', '89-98', '99-08', '09-18'], row_data_lst[y], label = row_name_lst[z])

    z += 1

    y += 1

    x += 1

    

    plt.title('Top 10 Military Spending Trend')

    plt.xlabel('Period')

    plt.ylabel('Total Spent')

    plt.legend()

    plt.show()

    



viz7 = viz7.tail(9)



row_data_lst = []

for index, row in viz7[['69-78', '79-88', '89-98', '99-08', '09-18']].iterrows():

    row_data = [row['69-78'], row['79-88'], row['89-98'], row['99-08'], row['09-18']]

    row_data_lst.append(row_data)

    

row_name_lst = []

for index, row in viz7[['Name']].iterrows():

    row_name = [row['Name']]

    row_name_lst.append(row_name)



x = 0

y = 0

z = 0

while x != len(row_data_lst):

    plt.figure(figsize=(25,15))

    plt.plot(['69-78', '79-88', '89-98', '99-08', '09-18'], row_data_lst[y], label = row_name_lst[z])

    z += 1

    y += 1

    x += 1

    plt.plot(['69-78', '79-88', '89-98', '99-08', '09-18'], row_data_lst[y], label = row_name_lst[z])

    z += 1

    y += 1

    x += 1

    plt.plot(['69-78', '79-88', '89-98', '99-08', '09-18'], row_data_lst[y], label = row_name_lst[z])

    z += 1

    y += 1

    x += 1

    plt.plot(['69-78', '79-88', '89-98', '99-08', '09-18'], row_data_lst[y], label = row_name_lst[z])

    z += 1

    y += 1

    x += 1

    plt.plot(['69-78', '79-88', '89-98', '99-08', '09-18'], row_data_lst[y], label = row_name_lst[z])

    z += 1

    y += 1

    x += 1

    plt.plot(['69-78', '79-88', '89-98', '99-08', '09-18'], row_data_lst[y], label = row_name_lst[z])

    z += 1

    y += 1

    x += 1

    plt.plot(['69-78', '79-88', '89-98', '99-08', '09-18'], row_data_lst[y], label = row_name_lst[z])

    z += 1

    y += 1

    x += 1

    plt.plot(['69-78', '79-88', '89-98', '99-08', '09-18'], row_data_lst[y], label = row_name_lst[z])

    z += 1

    y += 1

    x += 1

    plt.plot(['69-78', '79-88', '89-98', '99-08', '09-18'], row_data_lst[y], label = row_name_lst[z])

    z += 1

    y += 1

    x += 1 

    

    plt.title('Top 10 Military Spending Trend Excluding United States')

    plt.xlabel('Period')

    plt.ylabel('Total Spent')

    plt.legend()

    plt.show()

    plt.legend()

    plt.show()



viz8 = df.copy()



total_lst = []

for x in total:

    total_lst.append(x)

viz8['Total'] = total_lst



viz8 = viz8[['Name', 'Total']]



plt.figure(figsize=(10,5))

viz_8 = sns.violinplot(y=viz8['Total'], inner='box')

viz_8.set_title('Distribution of Total Military Spending')