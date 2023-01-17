#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSNUhYmLtqvT6JQ0KG8Pskt9vM1X82IQbWehEajxiz7IjVJ2nay&usqp=CAU',width=400,height=400)
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
df = pd.read_csv('../input/covid-19-banten-province/banten_covid19v1.csv', encoding='ISO-8859-2')

df.head()
df = df.rename(columns={'Unnamed: 0':'date'})
sns.barplot(x=df['SEMBUH'].value_counts().index,y=df['SEMBUH'].value_counts())
df["ODP"].plot.hist()

plt.show()
fig=sns.lmplot(x="POSITIF", y="ODP",data=df)
df.plot.area(y=['ODP', 'PDP','POSITIF','SEMBUH','MENINGGAL'],alpha=0.4,figsize=(12, 6));
df_grp = df.groupby(["date","ODP"])[["PDP","POSITIF","SEMBUH", "MENINGGAL"]].sum().reset_index()

df_grp.head()
df_grp_plot = df_grp.tail(80)
df_grp_r = df_grp.groupby("ODP")[['PDP',"POSITIF","SEMBUH","MENINGGAL"]].sum().reset_index()
df_grp_r.head()
df_grp_rl20 = df_grp_r.tail(20)
fig = px.bar(df_grp_rl20[['ODP', 'POSITIF']].sort_values('POSITIF', ascending=False), 

             y="POSITIF", x="ODP", color='ODP', 

             log_y=True, template='ggplot2', title='POSITIF Cases vs ODP')

fig.show()
df_grp_rl20 = df_grp_rl20.sort_values(by=['POSITIF'],ascending = False)
df_grp_d = df_grp.groupby("date")[["POSITIF","SEMBUH","MENINGGAL"]].sum().reset_index()
df_grp_dl20 = df_grp_d.tail(20)
pred_cnfrm = df_grp_d.loc[:,["date","POSITIF"]]
from fbprophet import Prophet

pr_data = pred_cnfrm.tail(10)

pr_data.columns = ['ds','y']

m=Prophet()

m.fit(pr_data)

future=m.make_future_dataframe(periods=15)

forecast=m.predict(future)

forecast
from fbprophet.plot import plot_plotly, add_changepoints_to_plot

fig = plot_plotly(m, forecast)

py.iplot(fig) 



fig = m.plot(forecast,xlabel='date',ylabel='Positif Count')
#sample codes from Mikey_Mtk @motokinakamura https://www.kaggle.com/motokinakamura/treemap-with-plotly

fig = go.Figure(go.Treemap(

    labels = ["Eve","Cain", "Seth", "Enos", "Noam", "Abel", "Awan", "Enoch", "Azura"],

    parents = ["", "Eve", "Eve", "Seth", "Seth", "Eve", "Eve", "Awan", "Eve"]

))



fig.show()
#codes from Mikey_Mtk @motokinakamura https://www.kaggle.com/motokinakamura/treemap-with-plotly

#make a df it's grouped by "Genre"

gb_MENINGGAL =df.groupby("MENINGGAL").sum()



gb_MENINGGAL.head()
# codes from Mikey_Mtk @motokinakamura https://www.kaggle.com/motokinakamura/treemap-with-plotly

MENINGGAL = list(gb_MENINGGAL.index)

score = list(gb_MENINGGAL.POSITIF)



print(MENINGGAL)

print(score)
#codes from Mikey_Mtk @motokinakamura https://www.kaggle.com/motokinakamura/treemap-with-plotly

#second treemap

test_tree_blue = go.Figure(go.Treemap(

    labels =  MENINGGAL,

    parents=[""]*len(MENINGGAL),

    values =  score,

    textinfo = "label+value",

    marker_colorscale = 'portland'

))



test_tree_blue.show()
#code from andre Sionek

import plotly.express as px



# Grouping it by Genre and track

plot_data = df.groupby(['ODP', 'SEMBUH'], as_index=False).MENINGGAL.sum()



fig = px.bar(plot_data, x='ODP', y='MENINGGAL', color='SEMBUH')

fig.update_layout(

    title_text='Covid-19 in Banten Province ',

    height=500, width=1000)

fig.show()
#code from Andre Sionek

import plotly.express as px



# Grouping it by Genre and track

plot_data = df.groupby(['ODP', 'POSITIF'], as_index=False).PDP.sum()



fig = px.line_polar(plot_data, theta='ODP', r='PDP', color='POSITIF')

fig.update_layout(

    title_text='Covid-19 Banten Province',

    height=500, width=1000)

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/png;base64,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',width=400,height=400)