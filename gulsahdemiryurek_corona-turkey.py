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
data=pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
data.head()
turkey= data[data["Country/Region"]=="Turkey"]
turkey.tail()
import plotly.graph_objects as go

fig=go.Figure()



fig.add_trace(go.Bar(

                x=turkey.ObservationDate,

                y=turkey["Confirmed"],

                name="Confirmed Cases",

                text=turkey["Confirmed"],

                textposition="outside",

                textfont_size=16,

                marker=dict(line=dict(width=2,

                                        color='DarkSlateGrey')),

                opacity=0.8))



# Use date string to set xaxis range

fig.update_layout(title_text="Türkiye Confirmed Cases",plot_bgcolor='azure',width=1000)

fig.update_xaxes(showline=True, linewidth=2, linecolor='dimgray', mirror=True,gridcolor='silver',title="Date")

fig.update_yaxes(showline=True, linewidth=2, linecolor='dimgray', mirror=True,gridcolor='silver' )



fig.show()
import seaborn as sns

import matplotlib.pyplot as plt

plt.loglog(turkey.index,turkey.Confirmed)

plt.title("Logarithmic Cases")

plt.show()
fig=go.Figure()

fig.add_trace(go.Scatter(x=turkey.ObservationDate,

                         y=turkey.Deaths,

                         name="Deaths",

                         text=turkey.Deaths,

                         textposition="top center",

                            

                         mode="markers+lines+text",marker=dict(size=10,symbol=22,

                              line=dict(width=2,

                                    color='DarkSlateGrey')),

                

                opacity=0.8))

fig.update_layout(title_text="Turkey Deaths",plot_bgcolor='ghostwhite',width=1000)

fig.update_xaxes(showline=True, linewidth=2, linecolor='dimgray', mirror=True,gridcolor='silver',title="Date")

fig.update_yaxes(showline=True, linewidth=2, linecolor='dimgray', mirror=True,gridcolor='silver' )



fig.show()

fig = go.Figure()





fig.add_trace(go.Scatter(

                x=turkey.ObservationDate,

                y=turkey['Recovered'],

                text=turkey.Recovered,

             

                 textposition="top center",

                     mode="markers+lines+text",marker=dict(size=10,symbol=15,

                              line=dict(width=2,

                                    color='green')),

                    name="Recovered",

                       

                opacity=0.8))

fig.update_layout(title_text="Turkey Recovered",plot_bgcolor='linen',width=1000)

fig.update_xaxes(showline=True, linewidth=2, linecolor='dimgray', mirror=True,gridcolor='silver',title="Date")

fig.update_yaxes(showline=True, linewidth=2, linecolor='dimgray', mirror=True,gridcolor='silver' )



fig.show()

                