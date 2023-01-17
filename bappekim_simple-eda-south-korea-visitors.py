import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
data = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        data.append(os.path.join(dirname, filename))
data
# age
df = pd.read_csv(data[0])
df
df.shape
df = pd.read_csv(data[1])
df
df.shape
df = pd.read_csv(data[2])
df
df.shape
# using age data
df = pd.read_csv(data[2])
df.head()
df.info()
df.describe()
df.groupby('nation').date.count()
df.groupby('nation').mean()['visitor'].sort_values(ascending=False)
def all_graph(df, x, y, length):
    fig,axes = plt.subplots(1,1,figsize=(20, 16))
    axes.set_title(y)
    axes.set_ylabel(y)
    axes.set_xlabel(x)
    axes.set_xticklabels(df[x].unique(), rotation=45)
    qualitative_colors = sns.color_palette("Paired", length)
    sns.lineplot(x, y, ci=None, hue='nation', 
                 marker='o', data=df, linewidth=2, palette=qualitative_colors)
    axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
all_graph(df, 'date', 'visitor', 60)
top_countries = ['China', 'Japan', 'Taiwan', 'USA']
def time_visitor_graph(name):
    fig,axes = plt.subplots(1,1,figsize=(10, 8))
    x = df[df['nation']==name].date
    y = df[df['nation']==name].visitor
    axes.set_title(name)
    axes.set_ylabel("The number of visitors")
    axes.set_xlabel("Date")
    axes.set_xticklabels(x, rotation=45)
    axes.plot(x, y, linewidth=3.0)
for country in top_countries:
    time_visitor_graph(country)
def month_compare_graph(name):
    fig,axes = plt.subplots(1,1,figsize=(10, 8))
    x = [1, 2, 3, 4]
    y = df[(df['date'].str.endswith(('-1', '-2', '-3', '-4'))) & (df['nation'] == name)].visitor
    
    axes.set_title(name)
    axes.set_ylabel("The number of visitors")
    axes.set_xlabel("Month")
    axes.plot(x, y[:4], c='b', linewidth=5.0, label='2019')
    axes.plot(x, y[4:], c='r', linewidth=5.0, label='2020')
    axes.legend(loc=3)
for country in top_countries:
    month_compare_graph(country)
all_graph(df, 'date', 'growth', 60)
top_countries = df.groupby('nation').mean()['visitor'].sort_values(ascending=False)[:10].index
top_countries
df_top = df[df['nation'].isin(top_countries)]
df_top
all_graph(df_top, 'date', 'growth', 10)
df = pd.read_csv(data[1])
df.sort_values('growth', ascending=False).head(5)
df.sort_values('growth').head(5)
df_2019 = df[df['date'].str.startswith('2019')]
df_2019.sort_values('growth', ascending=False).head(5)
df_2019.sort_values('growth').head(5)
all_graph(df_top, 'date', 'share', 10)