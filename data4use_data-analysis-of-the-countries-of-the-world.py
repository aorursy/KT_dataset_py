# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns             # for visualisation

import matplotlib.pyplot as plt   # for visualisation



# for showing plot in jupyter notebook

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# loading and reading data





data = pd.read_csv('/kaggle/input/countries-of-the-world/countries of the world.csv')
# seeing the data



data.head()
data.columns
data.dtypes


for col in data.columns:

    print(col  , (data[col].isnull().sum()/len(data[col])*100 ))

    
data = data.fillna(0)
for col in data.columns:

    print(col  , (data[col].isnull().sum()/len(data[col])*100 ))
new_column_name = {'Area (sq. mi.)':'Area' , 'Pop. Density (per sq. mi.)':'Pop_density' , 

                  'Coastline (coast/area ratio)':'Coastline' , 

                  'Infant mortality (per 1000 births)':'Infant_mortality' , 'GDP ($ per capita)':'GDP_per_capita' ,

                  'Literacy (%)':'Literacy_percent' , 'Phones (per 1000)':'Phones_per_k' , 'Arable (%)':'Arable' ,

                   'Crops (%)':'Crops' ,'Other (%)':'Other'}

data = data.rename(columns = new_column_name )
data.head()
def replace_commas(columns):

    for col in columns:

        data[col] = data[col].astype(str)

        dat = []

        for val in data[col]:

            val = val.replace(',' , '.')

            val = float(val)

            dat.append(val)



        data[col] = dat

    return(data.head())
columns = data[['Pop_density' , 'Coastline' , 'Net migration' , 'Infant_mortality' , 

                   'Literacy_percent' , 'Phones_per_k' , 'Arable' , 'Crops' , 'Other' , 'Birthrate' , 'Deathrate' , 'Agriculture' ,

                   'Industry' , 'Service']]

replace_commas(columns)
data.sort_values(by = 'GDP_per_capita' , ascending = False).Country.iloc[0:5]
sns.distplot(data.Infant_mortality)
data.sort_values(by = 'Infant_mortality' , ascending = False).Country.iloc[0:5]
data.sort_values(by = 'Infant_mortality' , ascending = False).Region.iloc[0:5]
sns.lmplot(x = 'GDP_per_capita' , y = 'Infant_mortality' , data = data)
sns.jointplot(x = 'GDP_per_capita' , y = 'Infant_mortality' , kind = 'hex' , data = data)