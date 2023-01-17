# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')

print(data)
data.head(10)

data.tail(10)
data.info()
data.columns
data.shape
print(data['EU_Sales'].value_counts(dropna =False))
data.describe()
data.boxplot(column='NA_Sales',by='Year')
data_new = data.head()

data_new
melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['NA_Sales','EU_Sales'])

melted
melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Other_Sales','Global_Sales'])

melted
melted.pivot(index = 'Name', columns = 'variable',values='value')
data1 = data.head(10)

data2= data.tail(10)

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True)

conc_data_row
data1 = data['Name'].head()

data2 = data['Global_Sales'].head()

new_data_conc = pd.concat([data1,data2],axis =1)

new_data_conc
data.dtypes
data['Genre'] = data['Genre'].astype('category')
data.dtypes
data.info()
data["Other_Sales"].value_counts(dropna =False)
data["Global_Sales"].value_counts(dropna =False)