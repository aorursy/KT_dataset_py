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
import pandas as pd

df = pd.read_csv("../input/gender-stats/e59e7c0f-3e73-4340-8638-7aa06c805647_Data.csv")

dfm = pd.read_csv("../input/gender-stats/e59e7c0f-3e73-4340-8638-7aa06c805647_Series - Metadata.csv")
#this shows what original dataset look like

df
#i'm going to set 'country name' as an index

df_index = df.set_index("Country Name")



df_index
#i'm going to keep the columns that I only want to keep

df_columns = df_index.loc[:,'2010 [YR2010]':'2019 [YR2019]']

df_columns
#To clean the data further, I am going to drop the rows if values from 2010~2019 are empty.

#I am going to convert empty value to display 'NaN' and drop the rows that does not have value (rows that has more than 1 'NaN')



nan_value = float("NaN")

df_columns.replace("..", nan_value, inplace = True)

df_nan=df_columns.dropna(thresh=1)



#print how data looks

df_nan
#convert "nan" values to "0" for numeric calculation

df0 = df_nan.fillna(0)



#print how data looks

df0
#Took me awhile to figure out that my data are not in number format

df0.dtypes
#convert data type as integer

df0 = df0.astype(int)

df0.dtypes



df0.to_csv('maternity_int.csv')
#see how many total days were used over 10 years period

df_sum = df0.sum(axis=1)

df_sum



df_sum.to_csv('maternity_total.cvs')
#graph of each country's paid maternity leave

df0.plot()
df_sum.plot()