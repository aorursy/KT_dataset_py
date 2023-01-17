# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd
import os

print(os.listdir("../input/immigration-to-canada-ibm-dataset"))
df_can_imm = pd.read_excel('../input/immigration-to-canada-ibm-dataset/Canada.xlsx',

                          sheet_name = "Canada by Citizenship",

                          skiprows = range(20),

                          skipfooter = 2)

df_can_imm.shape
df_can_imm.head(5)
df_can_imm.tail()
df_can_imm.info()
df_can_imm.columns.values
df_can_imm.index.values
df_can_imm.dtypes
df_can_imm.columns
print(type(df_can_imm.columns))
print(df_can_imm.columns.to_list())
print(type(df_can_imm.index))
print(type(df_can_imm.index.to_list()))
# in pandas axis =0 is rows(default) and axis = 1 represents column

df_can_imm.drop(['Type','Coverage','AREA','REG','DEV'], axis = 1, inplace = True)

df_can_imm.head(2)
df_can_imm.rename(columns={'OdName':'Country' , 'AreaName':'Continent', 'RegName':'Region'}, inplace = True)

df_can_imm.columns
df_for_imm_total = df_can_imm.drop(['Country','Continent','Region','DevName'], axis = 1)

df_for_imm_total

df_can_imm['Total'] = df_for_imm_total.sum(axis=1)

df_can_imm.head(2)
df_can_imm.isnull()
df_can_imm.isnull().sum()
df_can_imm.describe()
df_can_imm.describe(include = "all")
df_can_imm.Country
df_can_imm['Country']
df_can_imm[['Country','Total']]
df_can_imm[['Country',1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990]]

#as years are in integers
df_can_imm.set_index ('Country', inplace = True)

df_can_imm.head(2)
df_can_imm.index.name
print(df_can_imm.loc['India'])
print(df_can_imm.loc['India', 2013])
print(df_can_imm.loc['India', [2010,2011,2012,2013]])
print(df_can_imm.iloc[87])
print(df_can_imm.iloc[87,4])
print(df_can_imm.iloc[87,[4,5,6,7,8]])
print(df_can_imm[df_can_imm.index == 'India'])

print(df_can_imm[df_can_imm.index == 'India'].squeeze())
# japan for year 2013

df_can_imm.loc['Japan', 2013]
# alternate method

print(df_can_imm.iloc[87, 36]) # year 2013 is the last column, with a positional index of 36
# 3. for years 1980 to 1985

df_can_imm.loc['Japan', [1980, 1981, 1982, 1983, 1984, 1985]]
# 1. create the condition boolean series

condition = df_can_imm['Continent'] == 'Asia'

print(condition)
# 2. pass this condition into the dataFrame

df_can_imm[condition]
condition1 = (df_can_imm['Continent'] == 'Asia') & (df_can_imm['Region'] == 'Southern Asia')

df_can_imm[condition1]
print('data dimensions:', df_can_imm.shape)

print(df_can_imm.columns)

df_can_imm.head(2)
# we are using the inline backend

%matplotlib inline 



import matplotlib as mpl

import matplotlib.pyplot as plt
print ('Matplotlib version: ', mpl.__version__) # >= 2.0.0
print(plt.style.available)
mpl.style.use(['ggplot']) # optional: for ggplot-like style
#df_can_imm.columns = list(map(str, df_can_imm.columns))

#[print (type(x)) for x in df_can_imm.columns.values]


# useful for plotting

years = list(range(1980,2014))



years
haiti = df_can_imm.loc['Haiti', years] # passing in years 1980 - 2013 to exclude the 'total' column

haiti.head()
haiti.plot()
#haiti.index=haiti.index.map(int)

haiti.plot(kind = 'line')



plt.title("Immigration to canada")

plt.ylabel("NUmber of Immigrants")

plt.xlabel("Year")



# annotate the 2010 Earthquake. 

# syntax: plt.text(x, y, label)

plt.text(2000,6000,'2010 Earthquake')



plt.show() # need this line to show the updates made to the figure
df_indochina = df_can_imm.loc[['China','India'], years]

df_indochina .head()
df_indochina.plot(kind='line')
df_indochina = df_indochina.transpose()

df_indochina.head()
df_indochina.plot(kind='line')

plt.title('Immigrants from china and india')

plt.ylabel('NUmber of immigrants')

plt.xlabel('Years')

plt.show()

df_can_imm.sort_values(by = 'Total', ascending = False, axis = 0, inplace = True)

df_can_imm_top = df_can_imm.head(5)

df_can_imm_top
df_can_imm_top = df_can_imm_top[years].transpose() 

print(df_can_imm_top)

df_can_imm_top.plot(kind='line', figsize=(13, 8)) 

plt.title('Immigration Trend of Top 5 Countries')

plt.ylabel('Number of Immigrants')

plt.xlabel('Years')

plt.show()