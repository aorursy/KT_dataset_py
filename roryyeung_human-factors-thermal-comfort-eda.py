import numpy as np # Linear Algebra
import pandas as pd # Data Processing , CSV file I/O

#List the files in the input dataset

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Lets read the CSV blindly - just to see how it handles it.

df = pd.read_csv('../input/ashrae-global-thermal-comfort-database-ii/ashrae_db2.01.csv')

df.head(1)
#Now lets try it, just selecting columns of interest, and specifying the dtype.

cols = [
'Publication (Citation)','Year','Season','Koppen climate classification','Climate','City','Country',
'Building type','Age','Sex','Thermal sensation','Thermal sensation acceptability','Thermal preference',
'Thermal comfort','SET','Met','Air temperature (C)',
'Ta_h (C)','Relative humidity (%)','Humidity preference','Humidity sensation','Subject«s height (cm)',
'Subject«s weight (kg)','Outdoor monthly air temperature (C)'
]

Dtypes = {
'Publication (Citation)':'string','Season':'category','Koppen climate classification':'category',
'Climate':'category','City':'category','Country':'category','Building type':'category',
'Sex':'category','Thermal preference':'category'
}

df = pd.read_csv('../input/ashrae-global-thermal-comfort-database-ii/ashrae_db2.01.csv',usecols=cols, dtype = Dtypes, low_memory=False, na_values=['Nan','Na','NaN','N/A','na'])

df.head(1)
#Lets check that it works as we expected.

print(df.shape)
print(df.dtypes)
print(df.info)
#How many unique publications?

print('There are ' + str(df['Publication (Citation)'].nunique()) + ' unique publications.')

#How many unique cities?

print('There are ' + str(df['City'].nunique()) + ' unique cities.')

#What range of years?

print('The range of years is from ' + str(df['Year'].min()) + ' to ' + str(df['Year'].max()) + '.')
#Let's examine some basic stats

df.describe()

#Count the number of examples of each category.
count_climates=df.Climate.value_counts(sort=False)
count_cities=df.City.value_counts(sort=False)
count_year=df.Year.value_counts(sort=False)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 20))

plt.subplot(3, 1, 1)
count_climates.plot(kind='bar')

plt.subplot(2, 1, 2)
count_year.plot(kind='bar')

plt.show()
plt.figure(figsize=(16, 20))
count_cities.plot(kind='bar')