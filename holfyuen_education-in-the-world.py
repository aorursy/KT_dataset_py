import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
data = pd.read_csv('../input/data.csv')
data.shape
data.head()
data.loc[:,'Country Name':'Indicator Code'].describe()
countries = data.loc[:,['Country Name','Country Code']].drop_duplicates()
indicators = data.loc[:,['Indicator Name','Indicator Code']].drop_duplicates()
print (countries.shape, indicators.shape)
countries.head()
indicators.head()
present = data.loc[:,'1977':'2016'].notnull().sum()/len(data)*100
future = data.loc[:,'2020':].notnull().sum()/len(data)*100
plt.figure(figsize=(10,7))
plt.subplot(121)
present.plot(kind='barh', color='green')
plt.title('Missing Data (% of Data Rows)')
plt.ylabel('Column')
plt.subplot(122)
future.plot(kind='barh', color='limegreen')
plt.title('Missing Data (% of Data Rows)')
plt.show()
data = data.drop(['Unnamed: 61'], axis=1)
data.head()
countries[countries['Country Name'].str.contains('Hong')]
hk = data.loc[data['Country Code']=='HKG']
hk.head()
hk.shape
hkx = hk.dropna('index', thresh = 5) # First 4 columns and at least 1 value should be available
hkx.shape
hkx.head()
indicators.to_csv('indicators.csv', index=False)