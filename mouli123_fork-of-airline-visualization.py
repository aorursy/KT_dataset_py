import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly
import plotly.graph_objs as go
import plotly.plotly as py
plotly.offline.init_notebook_mode(connected=True)
airline_data_india=pd.read_csv('../input/Table_1_Airlinewise_DGCA_Q4_OCT-DEC_2017.csv')
airline_data_india.head()
airline_data_india.tail()
airline_data_india=airline_data_india[airline_data_india.Category!='TOTAL (DOMESTIC & FOREIGN CARRIERS)']
sns.set
size=sns.countplot(x='Category',data=airline_data_india)
size.figure.set_size_inches(10,8)
plt.show()
airline_category=airline_data_india.groupby(['Category'], as_index=False).agg({"PASSENGERS TO INDIA": "sum"})
fig = {
  "data": [
    {
      "values":airline_category['PASSENGERS TO INDIA'],
      "labels": airline_category.Category,
      "hoverinfo":"label+percent",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Flyers to India"
  }
}
plotly.offline.iplot(fig, filename='airline_pie_chart')
airline_foreign=airline_data_india[airline_data_india.Category=='FOREIGN CARRIERS']
import ipywidgets as widgets
g=widgets.Dropdown(
    options=list(airline_foreign['NAME OF THE AIRLINE'].unique()),
    value='AEROFLOT',
    description='Option:'
)
x_value=widgets.IntSlider(min=0,max=0)
y_value=widgets.IntSlider(min=1,max=4)
ui = widgets.HBox([g, x_value, y_value])
def on_change(change,change_x,change_y):
    sample_data=airline_foreign[airline_foreign['NAME OF THE AIRLINE']==change]
    numeric_data=sample_data.select_dtypes(exclude='object')
    x=numeric_data.columns.tolist(),
    y=numeric_data.values.tolist()
    trace = [go.Bar(
            x=x[0][change_x:change_y],
            y=y[0][change_x:change_y],
            marker=dict(
        color=['rgba(158, 21, 30, 1)', 'rgba(26, 118, 255,0.8)',
               'rgba(107, 107, 107,1)', 'rgba(255, 140, 0, 1)',
               'rgba(0, 191, 255, 1)']),
    )]
    plotly.offline.iplot(trace, filename='basic-bar')
out=widgets.interactive_output(on_change,{'change':g,'change_x':x_value,'change_y':y_value})
display(ui, out)
sorted_data=airline_data_india.sort_values(by=['PASSENGERS TO INDIA'],ascending=False)
sorted_data=sorted_data[sorted_data['NAME OF THE AIRLINE']!="TOTAL (FOREIGN CARRIERS)"]
sorted_data=sorted_data[sorted_data['NAME OF THE AIRLINE']!="TOTAL (DOMESTIC CARRIERS)"]
gs=widgets.Dropdown(
    options=sorted_data.iloc[:,3:].columns,
    value='PASSENGERS TO INDIA',
    description='Option:'
)
Max=widgets.IntSlider(min=2,max=40)
us = widgets.HBox([Max,gs])
def ms(x,y):
    sorteds=sorted_data.iloc[:x,]
    bar=sorteds['NAME OF THE AIRLINE'].tolist()
    value=sorteds[y].tolist()
    trace = [go.Bar(
            x=bar,
            y=value,
    )]
    plotly.offline.iplot(trace, filename='basic-bar')
out=widgets.interactive_output(ms,{'x':Max,'y':gs})
display(us, out)