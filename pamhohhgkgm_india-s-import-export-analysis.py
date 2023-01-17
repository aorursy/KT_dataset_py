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
import pandas as pd #pandas is for importing files,data processing

import numpy as np # linear algebra, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays

import seaborn as sns # data visulaization

import matplotlib.pyplot as plt # library for visualization 

from statsmodels.graphics.tsaplots import plot_acf #visualization for auto-correlation

from statsmodels.graphics.tsaplots import plot_pacf # partial auto correlation

imp_d= pd.read_csv('../input/india-trade-data/2018-2010_import.csv')
exp_d=pd.read_csv('../input/india-trade-data/2018-2010_export.csv')
imp_d.isnull().sum()
exp_d.isnull().sum()
imp_d.describe()
exp_d.describe()
imp_d.info()
exp_d.info()
imp_d=imp_d.dropna()

imp_d=imp_d.reset_index(drop=True)



exp_d=exp_d.dropna()

exp_d=exp_d.reset_index(drop=True)



imp_d.isnull().sum()
exp_d.isnull().sum()
import_contri=imp_d['country'].nunique()

import_contri
export_contri=exp_d['country'].nunique()

export_contri
import_gr=imp_d.groupby(['country','year']).agg({'value':'sum'})

export_gr=exp_d.groupby(['country','year']).agg({'value':'sum'})

export_gr.groupby(['country'])

import_temp=import_gr.groupby(['country']).agg({'value':'sum'})

export_temp=export_gr.groupby(['country']).agg({'value':'sum'}).loc[import_temp.index.values]



data_1=import_gr.groupby(['country']).agg({'value':'sum'}).sort_values(by='value').tail(10)

data_2=export_temp

data_3=data_2-data_1

data_1.column=['Import']

data_2.column=['Export']

data_3.column=['spend/gain']
df=pd.DataFrame(index=data_1.index.values)

df['Import']=data_1

df['Export']=data_2

df['spend/gain']=data_3
df
fig, ax = plt.subplots(figsize=(15,7))

df.plot(kind='bar',ax=ax)

ax.set_xlabel('Countries')

ax.set_ylabel('Value of transactions (in million US$)')
Deficit_=export_gr -import_gr

Time_Series=pd.DataFrame(index=import_gr.index.values)

Time_Series['Import']=import_gr

Time_Series['Export']=export_gr

Time_Series['spend / gain']=Deficit_



Time_Series



fig, ax = plt.subplots(figsize=(15,7))

Time_Series.plot(ax=ax,marker='o')

ax.set_xlabel('Years')

ax.set_ylabel('Value of transactions (in million US$)')
Time_Series
sns.barplot(x = 'year', y = 'spend / gain', data = Time_Series)

plt.show()