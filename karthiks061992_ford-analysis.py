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
dataset=pd.read_csv('/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv')

dataset.columns

dataset.head(5)
dataset=dataset.loc[:,['price','brand','year','country','state','mileage','color','condition','model']]

dataset.head(10)



#processing the null values

dataset.isnull().sum()

#No NaN values as expected to pre-processing condition

dataset.shape

#2499 tuples

dataset.groupby('brand')['model'].count().sort_values(ascending=False).head(4).plot.bar()

#clearly shows that ford cars have a higher model nmber and choices
#using log1p to bring down the range of prices

np.log1p(dataset.price)

dataset.groupby('brand')['price'].sum().sort_values(ascending=False).head(4).plot.line()

#price capsulation of ford 

dataset=dataset.rename(columns={'unnamed: 0':'index'})

#sale analysis using year prediction

dataset.groupby(['year','brand'])['price'].count().sort_values(ascending=False).head(4).plot.bar()

#Ford sold lot of its cars in 2019 
dataset.head()
#colorwise analysis of ford cars which sold the most in the last months

dataset.groupby(['brand','color','year'])['color'].count().sort_values(ascending=False).head(5).plot.bar()
#statewise sale analysis of ford car sales

dataset.groupby(['state','brand'])['model'].count().sort_values(ascending=False).head(5).plot.bar()
dataset.groupby('brand')['mileage'].sum().sort_values(ascending=False).head(4).plot.bar()

#Mileage tracks

#from the above we prove that ford manages to get most of it's sales in almost every corner of the world