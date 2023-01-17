# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/BlackFriday.csv")
df.info()
df.head()
df.columns
ageData = sorted(list(zip(df.Age.value_counts().index, df.Age.value_counts().values)))
age, productBuy = zip(*ageData)
age, productBuy = list(age), list(productBuy)
ageSeries = pd.Series((i for i in age))

data = [go.Bar(x=age, 
               y=productBuy, 
               name="How many products were sold",
               marker = dict(color=['#EA4A28', '#D3EA28', '#28EA4E', '#28EAE2', '#2008B9', '#E511E1', '#C4061D'],
                            line = dict(color='#7C7C7C', width = .5)),
              text="Age: " + ageSeries)]
layout = go.Layout(title= "How many products were sold by ages")
fig = go.Figure(data=data, layout=layout)
iplot(fig)
age0_17 = len(df[df.Age == "0-17"].User_ID.unique())
age18_25 = len(df[df.Age == "18-25"].User_ID.unique())
age26_35 = len(df[df.Age == "26-35"].User_ID.unique())
age36_45 = len(df[df.Age == "36-45"].User_ID.unique())
age46_50 = len(df[df.Age == "46-50"].User_ID.unique())
age51_55 = len(df[df.Age == "51-55"].User_ID.unique())
age55 = len(df[df.Age == "55+"].User_ID.unique())
agesBuyerCount = [age0_17,age18_25,age26_35,age36_45,age46_50,age51_55,age55]
               
trace1 = go.Bar(x = age,
                y = agesBuyerCount,
                name = "People count",
                marker = dict(color=['#F3B396', '#F3F196', '#A7F9AD', '#D5F0EF', '#AAADEE', '#EAC1E8', '#DF8787'],
                             line = dict(color='#7C7C7C', width = 1)),
                text = "Age: " + ageSeries)
data = [trace1]
layout = go.Layout(title= "How many people did shopping by ages")
fig = go.Figure(data=data, layout=layout)
iplot(fig)
totalPurchase = df.Purchase.sum()
y_percentage = [i/totalPurchase*100 for i in df.groupby("Age").sum()["Purchase"].values]
y_purchase = [i for i in df.groupby("Age").sum()["Purchase"].values]
x_percentage = [i for i in age]
x_purchase = [i for i in age]

trace0 = go.Bar(x = y_percentage,
                y = x_percentage,
                marker = dict(color = '#BF5959', 
                              line = dict(
                                  color = '#FFFEA6', width = 1) 
                             ),
               name = "Percentage of purchases amount in dollars",
               orientation = "h"
               )
trace1 = go.Scatter(x = y_purchase,
                    y = x_purchase,
                    mode='lines+markers',
                    line=dict(
                        color='#5079DC'),
                    name='Purchases amount in dollars ',)
layout = dict(
                title='Purchases in $',
                yaxis=dict(showticklabels=True,domain=[0, 0.85]),
                yaxis2=dict(showline=True,showticklabels=False,linecolor='rgba(102, 102, 102, 0.8)',linewidth=2,domain=[0, 0.85]),
                xaxis=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0, 0.42]),
                xaxis2=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0.47, 1],side='top',dtick=25),
                legend=dict(x=0.029,y=1.038,font=dict(size=10) ),
                margin=dict(l=200, r=20,t=70,b=70),
                paper_bgcolor='rgb(248, 248, 255)',
                plot_bgcolor='rgb(248, 248, 255)',
)

annotations = []

y_s = np.round(y_percentage, decimals=2)
y_nw = np.rint(y_purchase)

for ydn, yd, xd in zip(y_nw, y_s, x_percentage):
    annotations.append(dict(xref='x2', yref='y2',
                            y=xd, x=ydn - 20000,
                            text='{:,}'.format(ydn),
                            font=dict(family='Arial', size=12,
                                      color='rgb(128, 0, 128)'),
                            showarrow=False))
    annotations.append(dict(xref='x1', yref='y1',
                            y=xd, x=yd + 3,
                            text=str(yd) + '%',
                            font=dict(family='Arial', size=12,
                                      color='rgb(50, 171, 96)'),
                            showarrow=False))
annotations.append(dict(xref='paper', yref='paper',
                        x=-0.2, y=-0.109,
                        font=dict(family='Arial', size=10,
                                  color='rgb(150,150,150)'),
                        showarrow=False))

layout['annotations'] = annotations

fig = tools.make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
                          shared_yaxes=False, vertical_spacing=0.001)

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(layout)
iplot(fig)
data = [go.Box(y = df.Purchase, marker = dict(
        color = 'rgb(0, 128, 128)',
    ))]
iplot(data)

labels = sorted(df.City_Category.unique())
values = df.groupby("City_Category").sum()["Purchase"]
colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']

trace = go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))
iplot([trace])
labels = sorted(df.Stay_In_Current_City_Years.unique())
values = df.Stay_In_Current_City_Years.value_counts().sort_index()

trace = go.Pie(labels=labels, values=values)

iplot([trace])
data = [go.Bar(
            x=df.Product_Category_1.value_counts().sort_index().index,
            y=df.Product_Category_1.value_counts().sort_index().values
    )]
layout = go.Layout(
    title='Most Purchased Product Category',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
x_MaritalStatus = ['Single', 'Married']
y_PurchaseAmountAccordingToMaritalStatus = [int(df[df.Marital_Status == 0].Purchase.sum()), int(df[df.Marital_Status == 1].Purchase.sum())]

data = [go.Bar(x = x_MaritalStatus, 
                y = y_PurchaseAmountAccordingToMaritalStatus,
              marker = dict(color=['#0239E3','#E36502']))]
layout = go.Layout(title = 'Purchased Amount According To Marital Status (in US Dollars)')
fig = go.Figure(data=data, layout=layout)
iplot(fig)

x_Gender = ['Male', 'Female']
y_PurchaseAmountAccordingToGender = [df[df.Gender == 'M'].Purchase.sum(), df[df.Gender == 'F'].Purchase.sum()]

data = [go.Bar(x = x_Gender, 
                y = y_PurchaseAmountAccordingToGender,
              marker = dict(color=['#A6000D','#2ECEE7']))]
layout = go.Layout(title = 'Purchased Amount According To Gender (in US Dollars)')
fig = go.Figure(data=data, layout=layout)
iplot(fig)
x_Status = ['Single & Male', 'Single & Female', 'Married & Male', 'Married & Female']
y_Purchases = [df[(df.Gender == 'M') & (df.Marital_Status == 0)].Purchase.sum(),
              df[(df.Gender == 'F') & (df.Marital_Status == 0)].Purchase.sum(),
              df[(df.Gender == 'M') & (df.Marital_Status == 1)].Purchase.sum(),
              df[(df.Gender == 'F') & (df.Marital_Status == 1)].Purchase.sum()]

data = [go.Bar(x = x_Status, 
                y = y_Purchases,
              marker = dict(color=['rgb(0,212,65)','rgb(54,10,95)','rgb(5,22,205)','rgb(50,62,1)']))]
layout = go.Layout(title = 'Purchased Amount According To Gender and Marital Status (in US Dollars)')
fig = go.Figure(data=data, layout=layout)
iplot(fig)