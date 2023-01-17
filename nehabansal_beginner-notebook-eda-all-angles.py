# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from ipywidgets import interact ,widgets
import seaborn as sns
%matplotlib inline


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/who-suicide-statistics/who_suicide_statistics.csv")
print(data.shape,data.drop_duplicates().shape)
print(data.head())
print(data.info())
print(data.isnull().sum())
data_country_year = data.fillna(0).groupby(['country','year'],as_index=False).agg({'suicides_no':'sum','population':'sum'})
data_country_year['suicide%'] = data_country_year['suicides_no']*100/data_country_year['population']

data_country_year_sex = data.fillna(0).groupby(['country','year','sex'],as_index=False).agg({'suicides_no':'sum','population':'sum'})
data_country_year_sex['suicide%'] = data_country_year_sex['suicides_no']*100/data_country_year['population']

data_country_year_age = data.fillna(0).groupby(['country','year','age'],as_index=False).agg({'suicides_no':'sum','population':'sum'})
data_country_year_age['suicide%'] = data_country_year_age['suicides_no']*100/data_country_year_age['population']

data_country = data.fillna(0).groupby(['country'],as_index=False).agg({'suicides_no':['min','mean','median','max'],
                                                                       'population' :['min','mean','median','max']
                                                                      })
data_country.columns = ['country','min','mean','median','max','pop_min','pop_mean','pop_median','pop_max']


@interact
def year_wise_sucides(country = data_country_year.country.unique()):
#     print(data[data.country==country].isnull().sum())
    plot_df = data_country_year[data_country_year.country==country]
    plt.figure(figsize=(15,8))
    plt.subplots_adjust(hspace=0.7)
    plt.subplot(3,1,1)
    sns.lineplot("year","suicide%",data=plot_df,marker='o')
    plt.xticks(plot_df.year.unique(),rotation=90)
    plt.title(country)
#     plt.xlabel("Year")
    plt.ylabel("%Suicide")
    plt.grid()

    plt.subplot(3,1,2)
    plot_df_sex = data_country_year_sex[data_country_year_sex.country==country]
    sns.lineplot("year","suicides_no",hue='sex',data=plot_df_sex,marker='o')
    plt.xticks(plot_df.year.unique(),rotation=90)
#     plt.title(country)
#     plt.xlabel("Year")
    plt.ylabel("%Suicide")
    plt.grid()
    
    
    plt.subplot(3,1,3)
    plot_df_age = data_country_year_age[data_country_year_age.country==country]
    sns.lineplot("year","suicides_no",hue='age',data=plot_df_age,marker='o')
    plt.xticks(plot_df.year.unique(),rotation=90)
#     plt.title(country)
    plt.xlabel("Year")
    plt.ylabel("%Suicide")
    plt.grid()

top_num = 20
plot_df = data_country.sort_values(by=['mean'],ascending=False).head(top_num)
plt.figure(figsize=(15,5))

plt.plot("country","mean",data =plot_df,label='mean')
plt.plot("country","median",data =plot_df,label='median',linestyle='--',marker='o')
plt.fill_between(plot_df['country'],plot_df['min'],plot_df['max'],alpha=0.3 )
plt.legend()
plt.xticks(rotation=90)
plt.ylabel("#Suicides ")
plt.twinx()
plt.plot("country","pop_median",data =plot_df,label='pop_median',linestyle='-.',color='g')
plt.legend(loc='upper left')
plt.title("Country Level Suicides")
plt.xlabel("Country")
plt.ylabel("Population ")
@interact
def var_dist(col = ['age','sex','year']):
    plot_df = data.fillna(0).groupby([col],as_index=False).agg({'suicides_no':'sum','population':'sum'})
    plot_df['suicide%'] = plot_df['suicides_no']*100/plot_df['population']
    plt.figure(figsize=(10,5))
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(1,2,1)
    plt.barh(col,"suicide%",data=plot_df.sort_values(by=['suicide%']))
    plt.xlabel("suicide%")
    plt.ylabel("age")
    plt.subplot(1,2,2)
    plt.barh(col,"suicides_no",data=plot_df.sort_values(by=['suicides_no']))
    plt.xlabel("suicides_no")
    
    for col2 in ['age','sex']:
        if col!=col2:
            plot_df_bi = data.fillna(0).groupby([col,col2],as_index=False).agg({'suicides_no':'sum','population':'sum'})
            plot_df_bi['suicide%'] = plot_df['suicides_no']*100/plot_df['population']
            plt.figure(figsize=(15,4))
            sns.barplot(col,'suicides_no',hue=col2,data=plot_df_bi)
            plt.xticks(rotation=90)
            plt.title(f"{col} vs {col2}")
    
        
