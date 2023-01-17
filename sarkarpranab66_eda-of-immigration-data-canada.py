import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
data=pd.read_csv('../input/canada1/Canada - Canada by Citizenship.csv')
data.head()
print("Data Count :")
pd.DataFrame(data.count())
data.info()
# in pandas axis=0 represents rows (default) and axis=1 represents columns.
data.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)
data.head(2)
data.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)
data.columns
data['Total'] = data.sum(axis=1)
data.head()
data.isnull().sum()
data.describe()
data.set_index('Country',inplace=True)
data.head(3)
# 1. the full row data (all columns)
pd.DataFrame(data.loc['India']).drop(['Continent', 'Region','DevName','Total']).plot()
years = list(map(str, range(1980, 2014)))
years
# 1. create the condition boolean series
condition = data['Continent'] == 'Asia'
print (condition)
# 2. pass this condition into the dataFrame
data[condition]
# we can pass mutliple criteria in the same line. 
# let's filter for AreaNAme = Asia and RegName = Southern Asia

data[(data['Continent']=='Asia') & (data['Region']=='Southern Asia')]
df_CI = data.loc[['India', 'China'], years]
df_CI.head()
df_CI.plot(kind='line')
df_CI = df_CI.transpose()
df_CI.head()
### type your answer here

df_CI.index = df_CI.index.map(int) # let's change the index values of df_CI to type integer for plotting
df_CI.plot(kind='line')

plt.title('Immigrants from China and India')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()


data.sort_values(by='Total', ascending=False, axis=0, inplace=True)



# get the top 5 entries
df_top3 = data.head(4)


# transpose the dataframe
df_top3 = df_top3[years].transpose() 



print(df_top3)


# Step 2: Plot the dataframe. To make the plot more readeable, we will change the size using the `figsize` parameter.
df_top3.index = df_top3.index.map(int) # let's change the index values of df_top5 to type integer for plotting
df_top3.plot(kind='line', figsize=(14, 8)) # pass a tuple (x, y) size



plt.title('Immigration Trend of Top 3 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')


plt.show()

