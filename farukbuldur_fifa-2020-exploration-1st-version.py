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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/fifa20_data.csv",index_col="ID")

df.head()
df.describe(include='all')
from IPython.core.display import HTML



# convert your links to html tags 

def path_to_image_html(path):

    return '<img src="'+ path + '" width="60" >'
pd.set_option('display.max_colwidth', -1)

df_1=df.head()

HTML(df_1.to_html(escape=False ,formatters=dict(Image=path_to_image_html)))
df_strongest=df[["Name","Image","Age","Overall","Value","Club","Country"]][df["Overall"]==df["Overall"].max()]



HTML(df_strongest.to_html(escape=False ,formatters=dict(Image=path_to_image_html)))
df_weakest=df[["Name","Image","Age","Overall","Value","Club","Country"]][df["Overall"]==df["Overall"].min()]



HTML(df_weakest.to_html(escape=False ,formatters=dict(Image=path_to_image_html)))
keys=["Wage","Value","Release Clause"]

df_num = df.copy(deep=True)



for key in keys:

    df_num[key + "_Currency"] = df_num[key].str[:1]

    df_num[key + "_Multiplier"] = df_num[key].str[-1:]

    df_num[key] = df_num[key].replace('[\€,]', '', regex=True)

    df_num[key] = df_num[key].replace('[K,]', '', regex=True)

    df_num[key] = df_num[key].replace('[M,]', '', regex=True)
  

print(df_num["Wage_Currency"].unique(),

     df_num["Value_Currency"].unique(),

     df_num["Release Clause_Currency"].unique())
print(df_num["Wage_Multiplier"].unique(),

     df_num["Value_Multiplier"].unique(),

     df_num["Release Clause_Multiplier"].unique())
df_num[["Wage_Currency","Wage_Multiplier","Wage","Value","Value_Currency","Value_Multiplier","Release Clause",

       "Release Clause_Currency","Release Clause_Multiplier"]].head()
for key in keys:

    df_num[key]=df_num[key].astype(float)



for key in keys:

    df_num[key + "_Multiplier"] = df_num[key + "_Multiplier"].replace('[K,]', '1000', regex=True)

    df_num[key + "_Multiplier"] = df_num[key + "_Multiplier"].replace('[M,]', '1000000', regex=True)

    df_num[key + "_Multiplier"] = df_num[key + "_Multiplier"].astype(float)
keys=["Wage","Value","Release Clause"]



for key in keys:

    df[key + "_Numeric"] = df_num[key]*df_num[key + "_Multiplier"]
from tabulate import tabulate



print(tabulate(df[["Name","Wage","Wage_Numeric","Value","Value_Numeric"]].head(3) \

               .append(df[["Name","Wage","Wage_Numeric","Value","Value_Numeric"]].tail(3)), 

               headers='keys', tablefmt='grid',numalign="right",floatfmt=".0f"))
f, ax = plt.subplots(figsize=(16, 6))

ax.set(xscale="log")

plt.xlim(100, 1000000)

sns.scatterplot(x=df["Wage_Numeric"],y=df["Overall"],ax=ax)

plt.show()
f, ax = plt.subplots(figsize=(16, 6))

ax.set(xscale="log",)

sns.scatterplot(x=df["Value_Numeric"],y=df["Overall"],ax=ax)

plt.xlim(9000, 200000000)

plt.show()
f, ax = plt.subplots(figsize=(16, 6))

sns.distplot(a=df["Age"],bins=10,kde=False,hist_kws=dict(edgecolor="k", linewidth=2),ax=ax)

plt.show()
new_df = df.groupby('Country', as_index=False)['Overall'].max()
new_df.sort_values(by=["Overall"],ascending=False).head()
idx = df.groupby(['Country'])['Overall'].transform(max) == df['Overall']

HTML(df[idx].head().to_html(escape=False ,formatters=dict(Image=path_to_image_html)))
new_df_2 = df.groupby('Club', as_index=False)['Overall'].mean()

new_df_2 = new_df_2.sort_values(by=["Overall"],ascending=False)

new_df_2.head()

#new_df_2.shape
new_df_3 = new_df_2.sample(n=20)

new_df_3 = new_df_3.sort_values(by=["Overall"],ascending=False)
f, ax = plt.subplots(figsize=(16, 6))

sns.scatterplot(x=new_df_3["Club"],y=new_df_3["Overall"],ax=ax)

plt.xticks(rotation='vertical')

#plt.xlim(9000, 200000000)

plt.show()
plt.figure(figsize=(16,6))

sns.swarmplot(x=df.loc[(df['Country'] == 'Spain') | (df["Country"] == 'Argentina') | \

                  (df["Country"] == 'Portugal') | (df["Country"] == 'Turkey') ]["Country"],

              y=df.loc[(df['Country'] == 'Spain') | (df["Country"] == 'Argentina') | \

                  (df["Country"] == 'Portugal') | (df["Country"] == 'Turkey') ]["Overall"])

plt.show()
sns.distplot(a=df["Age"],bins=10,kde=False,hist_kws=dict(edgecolor="k", linewidth=2))

plt.show()
bins = pd.IntervalIndex.from_tuples([(10, 15), (15, 20), (20, 25), (25,30),(30,35),(35,40),(40,45)],closed="left")

df['Age_Bins']=pd.cut(df["Age"],bins=bins,retbins=False)

df['Age_Bins'].dtypes
import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected = True)

import plotly.graph_objs as go
idx = df.groupby(['Country'])['Overall'].transform(max) == df['Overall']



Max_Player = df[idx]



trace = [go.Choropleth(

            colorscale = 'Blues',

            locationmode = 'country names',

            locations = Max_Player['Country'],

            text = Max_Player['Name'],

            z = Max_Player['Overall'])]



layout = go.Layout(title = 'Country vs Their Top Players')





fig = go.Figure(data = trace, layout = layout)

py.iplot(fig)
sns.set(style="darkgrid")

ax = sns.countplot(x="foot", data=df)
plt.figure(figsize=(16,6))

sns.set(style="darkgrid")

ax = sns.countplot(x="BP", data=df)