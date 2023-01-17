import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #statistical data visualization

import matplotlib.pyplot as plt #visualization library

from statsmodels.graphics.tsaplots import plot_acf #Auto-Correlation Plots

from statsmodels.graphics.tsaplots import plot_pacf #Partial-Auto Correlation Plots
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import_df = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_import.csv")

export_df = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_export.csv")
import_df.head()
export_df.head()
import_df.isnull().sum()
import_df =import_df.dropna()

import_df = import_df.reset_index(drop=True)
export_df.isnull().sum()
export_df = export_df.dropna()

export_df = export_df.reset_index(drop=True)
importing_countries=import_df[['country']].nunique()

exporting_countries=export_df[['country']].nunique()

print("India imports from:",importing_countries,"countries")

print("India exports to:",exporting_countries,"countries")
import_group=import_df.groupby(['country','year']).agg({'value':'sum'})

export_group=export_df.groupby(['country','year']).agg({'value':'sum'})
export_group.groupby(['country'])

import_temp=import_group.groupby(['country']).agg({'value':'sum'})

export_temp=export_group.groupby(['country']).agg({'value':'sum'}).loc[import_temp.index.values]
data_1=import_group.groupby(['country']).agg({'value':'sum'}).sort_values(by='value').tail(10)

data_2=export_temp

data_3=data_2-data_1
data_1.columns=['Import']

data_2.columns=['Export']

data_3.columns=['Loss / Profit']
df=pd.DataFrame(index=data_1.index.values)

#df=pd.concat([data_1,data_2,data_3])

df['Import']=data_1

df['Export']=data_2

df['Loss / Profit']=data_3
df
fig, ax = plt.subplots(figsize=(15,7))

df.plot(kind='bar',ax=ax)

ax.set_xlabel('Countries')

ax.set_ylabel('Value of transactions (in million US$)')
df_import = import_df.groupby('country').agg({'value':'sum'}).sort_values(by='value', ascending = False).head(10)

df_import.plot(kind='bar')

df_export = export_df.groupby('country').agg({'value':'sum'}).sort_values(by='value', ascending = False).head(10)





df_export.plot(kind='bar')
HSCode=pd.DataFrame()

HSCode['Start']=[1,6,15,16,25,28,39,41,44,47,50,64,68,71,72,84,86,90,93,94,97]

HSCode['End']=[5,14,15,24,27,38,40,43,46,49,63,67,70,71,83,85,89,92,93,96,98]

HSCode['Sections']=['Animals & Animal Products',

'Vegetable Products',

'Animal Or Vegetable Fats',

'Prepared Foodstuffs',

'Mineral Products',

'Chemical Products',

'Plastics & Rubber',

'Hides & Skins',

'Wood & Wood Products',

'Wood Pulp Products',

'Textiles & Textile Articles',

'Footwear, Headgear',

'Articles Of Stone, Plaster, Cement, Asbestos',

'Pearls, Precious Or Semi-Precious Stones, Metals',

'Base Metals & Articles Thereof',

'Machinery & Mechanical Appliances',

'Transportation Equipment',

'Instruments - Measuring, Musical',

'Arms & Ammunition',

'Miscellaneous',

'Works Of Art',]
HSCode
import_df['Sections']=import_df["HSCode"]

export_df['Sections']=export_df["HSCode"]

for i in range(0,len(HSCode)):

    import_df.loc[(import_df["Sections"] >= HSCode['Start'][i]) & (import_df["Sections"] <= HSCode['End'][i]),"Sections"]=i

    export_df.loc[(export_df["Sections"] >= HSCode['Start'][i]) & (export_df["Sections"] <= HSCode['End'][i]),"Sections"]=i

    
import_group=import_df.groupby(['Sections','year']).agg({'value':'sum'})

export_group=export_df.groupby(['Sections','year']).agg({'value':'sum'})
import_temp=import_group.groupby(['Sections']).agg({'value':'sum'})

export_temp=export_group.groupby(['Sections']).agg({'value':'sum'}).loc[import_temp.index.values]
data_1=import_group.groupby(['Sections']).agg({'value':'sum'}).sort_values(by='value').tail(10)

data_2=export_temp

data_3=data_2-data_1

data_1.columns=['Import']

data_2.columns=['Export']

data_3.columns=['Loss / Profit']

df=pd.DataFrame(index=data_1.index.values)

#df=pd.concat([data_1,data_2,data_3])

df['Import']=data_1

df['Export']=data_2

df['Loss / Profit']=data_3
HSCode['Sections'][data_1.index.values]
df.index=HSCode['Sections'][data_1.index.values]

fig, ax = plt.subplots(figsize=(15,7))

df.plot(kind='bar',ax=ax)

ax.set_xlabel('Sections')

ax.set_ylabel('Value of transactions (in million US$)')
data_1.index=HSCode['Sections'][data_1.index.values]

data_1.plot(kind='bar')
data_2=export_group.groupby(['Sections']).agg({'value':'sum'}).sort_values(by='value').tail(10)

data_2.index=HSCode['Sections'][data_2.index.values]

data_2.plot(kind='bar')
Import_ =import_df.groupby(['year']).agg({'value':'sum'})

Export_ =export_df.groupby(['year']).agg({'value':'sum'})

Deficit_=Export_ -Import_

Time_Series=pd.DataFrame(index=Import_.index.values)

Time_Series['Import']=Import_

Time_Series['Export']=Export_

Time_Series['Loss / Profit']=Deficit_
Time_Series
fig, ax = plt.subplots(figsize=(15,7))

Time_Series.plot(ax=ax,marker='o')

ax.set_xlabel('Years')

ax.set_ylabel('Value of transactions (in million US$)')
Time_Series.index.name = 'Year'

Time_Series.reset_index(inplace=True)
Time_Series
# Plotting bar plot for yearwise Trend

sns.barplot(x = 'Year', y = 'Loss / Profit', data = Time_Series)

plt.show()
China_df=import_df.groupby(['country'])

China_df=China_df.get_group('CHINA P RP') 

USA_df=export_df.groupby(['country'])

USA_df=USA_df.get_group('U S A')
import pylab as pl

China=China_df.groupby(['year']).agg({'value':'sum'})

USA=USA_df.groupby(['year']).agg({'value':'sum'})

contribution=pd.DataFrame(index=China.index.values)

contribution["USA's export value"]=USA

contribution["China's import value"]=China

contribution.plot(marker='o')

pl.suptitle("China's import and USA's export contributions trend")

USA_export=USA_df.groupby(['year','Commodity']).agg({'value':'sum'}).sort_values(by='value').tail(10)

China_import=China_df.groupby(['year','Commodity']).agg({'value':'sum'}).sort_values(by='value').tail(10)
China_import.plot.barh()#(kind='bar')

pl.suptitle("China's Top imported product Yearwise")
USA_export.plot.barh()#(kind='bar')

pl.suptitle("USA's Top imported product Yearwise")