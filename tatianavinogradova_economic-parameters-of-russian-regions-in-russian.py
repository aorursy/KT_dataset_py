import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import squarify
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import base64
import io
import scipy.misc
import codecs
from IPython.display import HTML
from matplotlib_venn import venn2
from subprocess import check_output

from matplotlib.ticker import ScalarFormatter
import itertools
#import Translator
import time
data = pd.read_csv("../input/sberbank-open-data-eng/df.csv", index_col="region", parse_dates=True)
data.head()
pd.unique(data['name'])
data = data.pivot_table(index=["region", "date"], columns="name", values="value", aggfunc="mean", fill_value=0).reset_index()
data.head()
data = data[data["region"] != "Russia"]
data['year'] = data["date"].apply(lambda t: time.strptime(t, '%Y-%m-%d').tm_year)
plt.subplots(figsize=(12,5))
ax = sns.countplot(y=data['year'])

for i, year in enumerate(pd.unique(data['year'])):
    ax.text(15, i, data['year'][data['year'] == year].count())

plt.xlabel('Количество записей')
plt.ylabel('Год')
plt.show()
#Исследуем, в каких регионах самые высокие заработные платы (среднее)
region_val = data.groupby("region")["Average salary"].agg(np.median).sort_values(ascending=False)[:20].to_frame()
a = sns.barplot(region_val["Average salary"],region_val.index,palette='inferno')
plt.title('Топ 20 : Регионы с самой высокой средней заработной платой')
plt.xlabel('Средняя заработная плата')
plt.ylabel('Регион')
fig=plt.gcf()
fig.set_size_inches(15,15)


for i, region in enumerate(region_val.index):
    val = region_val.reset_index()
    a.text(5000, i, str(int(val[val["region"] == region]['Average salary'])) + ' руб.', color='white', horizontalalignment='center',
          fontsize=18)

plt.show();
#среднее количество заявок
region_val = data.groupby("region")["The number of applications for consumer loans"].agg(np.mean).sort_values(ascending=False)[:20].to_frame()
ax = sns.barplot(region_val.index, region_val["The number of applications for consumer loans"], palette='inferno')
plt.title('Топ 20 : Регионы с самым большим количеством заявок на потребительские кредиты')
plt.xlabel('Регион')
plt.ylabel('Количество заявок на кредит')
fig=plt.gcf()
fig.set_size_inches(20,10)

for axis in [ax.yaxis]:
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    axis.set_major_formatter(formatter)


plt.xticks(rotation=60, horizontalalignment='right', fontsize=15)
#a.barplot

plt.show()
region_val = data.groupby("region")["Average consumer loan application"].agg(np.mean).sort_values(ascending=False)[:20].to_frame()
ax = sns.barplot(region_val.index, region_val["Average consumer loan application"], palette='inferno')
plt.title('Топ 20 : Регионы с самыми высокими размерами потребительских кредитов')
plt.xlabel('Регион')
plt.ylabel('Среднее значение потребительского кредита')
fig=plt.gcf()
fig.set_size_inches(20,10)

for axis in [ax.yaxis]:
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    axis.set_major_formatter(formatter)


plt.xticks(rotation=60, horizontalalignment='right', fontsize=15)

plt.show()
region_val = data.groupby("region")["Average spending on cards"].agg(np.mean).sort_values(ascending=False)[:20].to_frame()
ax = sns.barplot(region_val.index, region_val["Average spending on cards"], palette='inferno')
plt.title('Топ 20 : Регионы с самыми высокими расходами по карте')
plt.xlabel('Регион')
plt.ylabel('Среднее значение расходов по карте')
fig=plt.gcf()
fig.set_size_inches(20,10)

for axis in [ax.yaxis]:
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    axis.set_major_formatter(formatter)


plt.xticks(rotation=60, horizontalalignment='right', fontsize=15)

plt.show()
new_data = data[data["region"] == "Primorsky Krai"]
new_data = new_data.rename(columns={'Average check in Restaurant format' : 'Average check in Restaurant',
                        'Average Fast Food format Check' : 'Average check in Fast Food'})
plt.subplots(figsize=(20,10))
ax = sns.countplot(y=new_data['year'])


plt.xlabel('Количество записей', fontsize=15)
plt.ylabel('Год', fontsize=15)
plt.title('Количество записей по Приморскому краю за каждый год', fontsize=20)
plt.show()
new_data.loc[new_data["year"]==2019, "year"] = 2018
plt.subplots(figsize=(30,40))
col=new_data.columns.drop(['region', 'date', 'year', 'Average spending in a fast food restaurant'])
#translator= Translator(to_lang="Russian")
length=len(col)

for i,j in itertools.zip_longest(col,range(length)):
    plt.subplot((length/2),2,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    
    aa = new_data.groupby(["year"])[i].agg(np.mean).to_frame().reset_index()
    sns.barplot(y=aa[i], x=aa["year"])
    #new_data[i].hist(bins=10,edgecolor='black')
    #plt.axvline(new_data.groupby(["year"])[i].agg(np.mean).mean(),linestyle='dashed',color='r')
    plt.title(i,size=30)
    plt.xlabel('')
    plt.ylabel('')
    

plt.show()