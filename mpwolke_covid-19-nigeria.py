#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSQwWWqYwGxI8LvqJLmLTnQ_bguyWX0sh_7gCCvkdBtltg8PMms&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go

import plotly.offline as py

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)

from bokeh.io import output_file, output_notebook

from bokeh.plotting import figure, show, reset_output

from bokeh.models import ColumnDataSource, HoverTool

from bokeh.layouts import row, column, gridplot

from bokeh.models.widgets import Tabs, Panel

import bokeh.palettes

from bokeh.transform import cumsum

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

from plotly.offline import iplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/covid19-nigeria-dataset-eco/NIGERIA COVID-19 ECO.csv', encoding='ISO-8859-2')



df.head().style.background_gradient(cmap='Greens')
df_grp = df.groupby(["DATE","ID"])[["CONFIRMED CASE","NEW CASE","DEATH CASE"]].sum().reset_index()

df_grp.head()
df_grp = df_grp.rename(columns={"CONFIRMED CASE":"confirmed","NEW CASE":"new", "DEATH CASE": "death"})
plt.figure(figsize=(15, 5))

plt.title('ID')

df_grp.ID.value_counts().plot.bar();
df_grp_plot = df_grp.tail(80)
fig=px.bar(df_grp_plot,x='ID', y="confirmed", animation_frame="DATE", 

           animation_group="ID", color="ID", hover_name="ID")

fig.update_yaxes(range=[0, 1500])

fig.update_layout(title='Confirmed vs ID')
df_grp_r = df_grp.groupby("ID")[["confirmed","new","death"]].sum().reset_index()
df_grp_r.head()
df_grp_rl20 = df_grp_r.tail(20)
fig = px.bar(df_grp_rl20[['ID', 'confirmed']].sort_values('confirmed', ascending=False), 

             y="confirmed", x="ID", color='ID', 

             log_y=True, template='ggplot2', title='Confirmed Cases vs ID')

fig.show()
df_grp_rl20 = df_grp_rl20.sort_values(by=['confirmed'],ascending = False)
plt.figure(figsize=(40,15))

plt.bar(df_grp_rl20.ID, df_grp_rl20.confirmed,label="confirmed")

plt.bar(df_grp_rl20.ID, df_grp_rl20.new,label="new")

plt.bar(df_grp_rl20.ID, df_grp_rl20.death,label="death")

plt.xlabel('DATE')

plt.ylabel("Count")

plt.xticks(fontsize=13)

plt.yticks(fontsize=15)



plt.legend(frameon=True, fontsize=12)

plt.title('Confirmed vs News vs Deaths',fontsize=30)

plt.show()



f, ax = plt.subplots(figsize=(40,15))

ax=sns.scatterplot(x="ID", y="confirmed", data=df_grp_rl20,

             color="black",label = "Confirm")

ax=sns.scatterplot(x="ID", y="new", data=df_grp_rl20,

             color="red",label = "new")

ax=sns.scatterplot(x="ID", y="death", data=df_grp_rl20,

             color="blue",label = "death")

plt.plot(df_grp_rl20.ID,df_grp_rl20.confirmed,zorder=1,color="black")

plt.plot(df_grp_rl20.ID,df_grp_rl20.new,zorder=1,color="red")

plt.plot(df_grp_rl20.ID,df_grp_rl20.death,zorder=1,color="blue")

plt.xticks(fontsize=13)

plt.yticks(fontsize=15)

plt.legend(frameon=True, fontsize=12)
df_grp_d = df_grp.groupby("DATE")[["confirmed","new","death"]].sum().reset_index()
df_grp_dl20 = df_grp_d.tail(20)
df_grp_d['confirmed_new'] = df_grp_d['confirmed']-df_grp_d['confirmed'].shift(1)

df_grp_d['new_new'] = df_grp_d['new']-df_grp_d['new'].shift(1)

df_grp_d['death_new'] = df_grp_d['death']-df_grp_d['death'].shift(1)
new = df_grp_d

new = new.tail(14)
f, ax = plt.subplots(figsize=(23,10))

ax=sns.scatterplot(x="DATE", y="confirmed", data=df_grp_dl20,

             color="black",label = "Confirmed Patients")

ax=sns.scatterplot(x="DATE", y="new", data=df_grp_dl20,

             color="red",label = "News")

ax=sns.scatterplot(x="DATE", y="death", data=df_grp_dl20,

             color="blue",label = "Deaths")

plt.plot(df_grp_dl20.DATE,df_grp_dl20.confirmed,zorder=1,color="black")

plt.plot(df_grp_dl20.DATE,df_grp_dl20.new,zorder=1,color="red")

plt.plot(df_grp_dl20.DATE,df_grp_dl20.death,zorder=1,color="blue")
pred_cnfrm = df_grp_d.loc[:,["DATE","confirmed"]]
from fbprophet import Prophet

pr_data = pred_cnfrm.tail(10)

pr_data.columns = ['ds','y']

m=Prophet()

m.fit(pr_data)

future=m.make_future_dataframe(periods=15)

forecast=m.predict(future)

forecast
cnfrm = forecast.loc[:,['ds','trend']]

cnfrm = cnfrm[cnfrm['trend']>0]

cnfrm.columns = ['DATE','confirmed']

cnfrm.head(10)
from fbprophet.plot import plot_plotly, add_changepoints_to_plot

fig = plot_plotly(m, forecast)

py.iplot(fig) 



fig = m.plot(forecast,xlabel='DATE',ylabel='Confirmed Count')
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSSopYF5sz9mGFf0XxCJ0F36YuopCfWwjt-s-qvlUnwP_IjFv36&usqp=CAU',width=400,height=400)