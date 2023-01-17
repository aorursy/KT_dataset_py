



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

import plotly.graph_objs as go

%matplotlib inline

plt.rcParams['figure.figsize'] = (9.0,9.0)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_export = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_export.csv')

data_import = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_import.csv')

print("export data:",data_export.shape,'\n',"import data:",data_import.shape)
#export data

data_export.head()
#import data

data_import.head()
data_export['value'].describe() 
data_import['value'].describe() 
data_export.info()
data_import.info()
print("total null values in export data:",data_export['value'].isnull().sum())

print("total null values in export data:",data_import['value'].isnull().sum())

# Droping all the rows with null values by writing a function :cleanup()

def filling_null(data_df):

    #data_df = data_df[data_df.value!=0]

    data_df["value"].fillna(data_df['value'].mean(),inplace = True)

    data_import.year = pd.Categorical(data_import.year)

    return data_df
data_import = filling_null(data_import)

data_export = filling_null(data_export)

print("total null values in export data;",data_export['value'].isnull().sum())

print("total null values in export data;",data_import['value'].isnull().sum())

def drop_un(dat):

        dat['country']= dat['country'].apply(lambda x : np.NaN if x == "UNSPECIFIED" else x)

        dat.dropna(inplace=True)

        return dat
data_import = drop_un(data_import)

data_export = drop_un(data_export)

print("total number of countries india exporting commodity:",len(data_export['country'].unique()))

print("total number of countries india importing commodity:",len(data_import['country'].unique()))
df2 = data_import.groupby('country').agg({'value':'sum'}).sort_values(by='value', ascending = False)

df2 = df2.head()



df3 = data_export.groupby('country').agg({'value':'sum'}).sort_values(by='value', ascending = False)

df3 = df3.head()



sb.set_style('whitegrid')

sb.barplot(df2.index,df2.value, palette = 'dark')

plt.title('country with highest value (import trade)', fontsize = 20)

plt.show()
sb.set_style('whitegrid')

sb.barplot(df3.index,df3.value, palette = 'bright')

plt.title('country with highest value (export trade)', fontsize = 20)



plt.show()
# top 5 most exported commodity 

df6 = data_export.groupby('Commodity').agg({'value':'sum'}).sort_values(by='value', ascending=False)

df6 = df6.head(5)

df6
sb.barplot(df6['value'],df6.index, palette = 'bright')

plt.title('Top 5 exporting Commodities', fontsize = 30)

plt.show()
# top5 most imported commmodit 

df7 = data_import.groupby('Commodity').agg({'value':'sum'}).sort_values(by='value', ascending=False)

df7 = df7.head(5)

df7
sb.barplot(df7['value'],df7.index, palette = 'dark')

plt.title('Top 5 importing Commodities',fontsize=30)

plt.show()
# import yearwise

df4 = data_import.groupby('year').agg({'value':'sum'})

# export yearwise

df5 = data_export.groupby('year').agg({'value':'sum'})

# deficite
sb.barplot(df4.index,df4.value, palette = 'Reds_d')

plt.title('Yearly Import', fontsize =30)

plt.show()
sb.barplot(df5.index,df5.value, palette = 'Blues_d')

plt.title('Yearly Export', fontsize =30)
df_import = data_import[data_import.value>1000]

df_export = data_export[data_export.value>1000]



df_import.head(10)
df_export.head(10)
f1 = df_import.groupby(['country']).agg({'value': 'sum'}).sort_values(by='value')

f2 = df_export.groupby(['country']).agg({'value': 'sum'}).sort_values(by='value')

sb.heatmap(f1)

plt.title('highest trade import countrywise', fontsize = 20)

plt.show()
sb.heatmap(f2)

plt.title('highest trade export countrywise', fontsize = 20)

plt.show()