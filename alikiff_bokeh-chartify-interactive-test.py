# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from bokeh.io import output_notebook,show

from bokeh.plotting import figure

from bokeh.models.widgets import Select,Div

from bokeh.models import ColumnDataSource,CustomJS

from bokeh.layouts import widgetbox,column,row

output_notebook()
df=pd.read_csv("../input/master.csv")
df.head()
df_country=df.groupby(["country","year","country-year"],as_index=False)["population","suicides_no"].sum()

df_country["suicides_per_100k"]=df_country["suicides_no"]/(df_country["population"]/100000)

df_country.head()
df_source=df_country.pivot(index="year",columns="country",values="suicides_per_100k").reset_index()

df_source=df_source.loc[df_source["year"].isin(range(1990,2016))]

df_source.dropna(axis=1,inplace=True)

df_source["Germany"]



df_source["y1"]=df_source["Germany"]

df_source["y2"]=df_source["Brazil"]



df_source.head(1)
laender_liste=[country for country in df_source.columns if not country in ["year","y1","y2"]]

source=ColumnDataSource(data=df_source)
callback=CustomJS(args=dict(source=source),

                  code="""

                        var indices=[]; data=source.data; 

                        if (cb_obj.title=='Country 1') {data["y1"]=data[cb_obj.value]; }

                        else {data["y2"]=data[cb_obj.value]; }

                        source.change.emit(); """)

select1=Select(options=laender_liste, value="Germany", callback=callback, title="Country 1:")

select2=Select(options=laender_liste, value="Brazil", callback=callback, title="Country 2:")

p=figure()



p.line(x="year",y="y1",source=source, line_width=3,legend="Country 1", line_color="green")

p.line(x="year",y="y2",source=source, line_width=3, line_color="grey", legend="Country 2")



show(column(Div(text="<h2 width=max_content>Suicides per selected Country</h2>"),

            row(widgetbox(select1),widgetbox(select2)),

            p))