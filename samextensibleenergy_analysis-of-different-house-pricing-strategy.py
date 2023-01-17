# Calling libraries

import math

import pandas as pd 

import datetime as dt



# Data Visualization Tools

import seaborn as sns

from matplotlib import pyplot

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.offline as offline

offline.init_notebook_mode()

from plotly import tools

import plotly.tools as tls
#Given Data (CONSTANTS)



Zillow_CC = 6174

US_CC     = 38000

Leads = 4.00

Median_SP = 350000

Marketing_Cost_Ratio = 0.01

Profit_Margin = 0.05

Price_pLead = 40

Flat_Price = 400



Leads_Growth_2018_MOM = 0.05

Leads_Growth_2019_MOM = 0.04

Leads_Growth_2020_MOM = 0.01





Z_CC_Growth_2018_MOM = 0.06

Z_CC_Growth_2019_MOM = 0.04

Z_CC_Growth_2020_MOM = 0.02
lead = []



for y in range (2018, 2021):

    if y == 2018: 

        for m in range (1,13):

            Leads = Leads*((1 + Leads_Growth_2018_MOM)**1)

            Leads = float(round(Leads,2))

            lead.append(Leads);

    if y == 2019:  

        for m in range (1,13):

            Leads = Leads*((1 + Leads_Growth_2019_MOM)**1)

            Leads = float(round(Leads,2))

            lead.append(Leads);

    if y == 2020:  

        for m in range (1,13):

            Leads = Leads*((1 + Leads_Growth_2020_MOM)**1)

            Leads = float(round(Leads,2))

            lead.append(Leads);

       

print((lead))
Z_CC = []



for y in range (2018, 2021):

    if y == 2018: 

        for m in range (1,13):

            Zillow_CC = Zillow_CC*((1 + Z_CC_Growth_2018_MOM)**1)

            Zillow_CC = float(round(Zillow_CC,2))

            Z_CC.append(Zillow_CC);

    if y == 2019:  

        for m in range (1,13):

            Zillow_CC = Zillow_CC*((1 + Z_CC_Growth_2019_MOM)**1)

            Zillow_CC = float(round(Zillow_CC,2))

            Z_CC.append(Zillow_CC);

    if y == 2020:  

        for m in range (1,13):

            Zillow_CC = Zillow_CC*((1 + Z_CC_Growth_2020_MOM)**1)

            Zillow_CC = float(round(Zillow_CC,2))

            Z_CC.append(Zillow_CC);

       

print((Z_CC))
Price_Flat = 6147

flat = []



for y in range (2018, 2021):

    if y == 2018: 

        for m in range (1,13):

            Price_Flat = Price_Flat*((1 + (Z_CC_Growth_2018_MOM*0.9))**1)

            Price_Flat = float(round(Price_Flat,2))

            flat.append(Price_Flat);

    if y == 2019:  

        for m in range (1,13):

            Price_Flat = Price_Flat*((1 + (Z_CC_Growth_2019_MOM*0.9))**1)

            Price_Flat = float(round(Price_Flat,2))

            flat.append(Price_Flat);

    if y == 2020:  

        for m in range (1,13):

            Price_Flat = Price_Flat*((1 + (Z_CC_Growth_2020_MOM*0.9))**1)

            Price_Flat = float(round(Price_Flat,2))

            flat.append(Price_Flat);

       

print((flat))
# Creating the Data Frame from the Above Lists



df = pd.DataFrame(list(zip(lead, Z_CC, flat)), 

               columns =['Avg_Leads', 'Zillow_Communites_Leads', 'Zillow_Communities_Flat'])



# Calculating the revenues

df['Leads_Revenue'] = df['Avg_Leads'] * df['Zillow_Communites_Leads'] * Price_pLead

df['Flat_Revenue'] =   Flat_Price * df['Zillow_Communities_Flat'] 



# Generating a DataTime Index

df['Date'] = pd.date_range(start='Jan-2018', periods=36, freq='M')

df['Year'] = df['Date'].dt.year

df['Month'] = df['Date'].dt.month

df = df.set_index(pd.DatetimeIndex(df['Date']))

# df = df.drop(columns=['Date'])



print(df)
list(df.columns.values)
trace03 = go.Bar(

    x = df.index.to_pydatetime(),

    y = df.Zillow_Communites_Leads,

    name='Leads_Revenue',

    

    marker=dict(

        color='black', 

        line=dict(

            color='rgb(8,48,107)',

            width=1.5),

        ),

    opacity=0.5

)



trace04 = go.Bar(

    x = df.index.to_pydatetime(),

    y = df.Zillow_Communities_Flat,

    name='Flat_Revenue',

    

    marker=dict(

        color='Blue', 

        line=dict(

            color='rgb(8,48,107)',

            width=1.5),

        ),

    opacity=1

)





layout = go.Layout(

    title='Total Communities Per Month'

)



data = [trace03, trace04]

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename="Monthly_Count_Communities")
trace01 = go.Bar(

    x = df.index.to_pydatetime(),

    y = df.Leads_Revenue,

    name='Leads_Revenue',

    

    marker=dict(

        color='orange', 

        line=dict(

            color='rgb(8,48,107)',

            width=1.5),

        ),

    opacity=0.15

)



trace02 = go.Bar(

    x = df.index.to_pydatetime(),

    y = df.Flat_Revenue,

    name='Flat_Revenue',

    

    marker=dict(

        color='Blue', 

        line=dict(

            color='rgb(8,48,107)',

            width=1.5),

        ),

    opacity=0.15

)



trace11 = go.Scatter(

    x = df.index.to_pydatetime(),

    y = df.Leads_Revenue,

    mode = 'lines+markers',

    name = 'Leads_Revenue',

    marker=dict(

        color='Red', 

        line=dict(

            color='rgb(8,48,107)',

            width=1.5),

        ),

    opacity=1.0

)



trace12 = go.Scatter(

    x = df.index.to_pydatetime(),

    y = df.Flat_Revenue,

    mode = 'lines+markers',

    name = 'Flat_Revenue',

    marker=dict(

        color='Purple', 

        line=dict(

            color='rgb(8,48,107)',

            width=1.5),

        ),

    opacity=1.0

)





layout = go.Layout(

    title='Revenue Per Month'

)



data = [trace01, trace02, trace11, trace12]

# data = [trace11, trace12]

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename="Revenue_Plot")
#Calculating Per home profit



Median_Sale_Price = 350000

Average_Sale_Price = 400000 # [Reference: 1 (above)]

Marketing_Expense_Ratio = 0.01

Profit_Margin_Ratio = 0.05



Average_Cost_Price = Average_Sale_Price / ( 1 + Marketing_Expense_Ratio  +  Profit_Margin_Ratio  ) 

Average_Profit = (Profit_Margin_Ratio * Average_Cost_Price)

Average_Marketing_Expense = (Marketing_Expense_Ratio * Average_Cost_Price )



print('Average Cost Price: USD',  Average_Cost_Price, 'per home') 

print('Average Profit: USD', Average_Profit, 'per home')

print('Average Marketing Expense: USD', Average_Marketing_Expense , 'per home')
import matplotlib.pyplot as plt

# Data to plot

labels = 'Cost_Price', 'Profit' , 'Marketing_Expense'

data = [Average_Cost_Price, Average_Profit, Average_Marketing_Expense]

colors = [ 'yellowgreen', 'lightcoral','darkturquoise']

explode = (0.25, 0.25, 0)  # explode 1st slice

 

# Plot

plt.pie(data, explode=explode,  labels=labels, colors=colors,

        autopct= ('%1.1f%%'), shadow=True, startangle=150)

plt.axis('equal')

plt.show()