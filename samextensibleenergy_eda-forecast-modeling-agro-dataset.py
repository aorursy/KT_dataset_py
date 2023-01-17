

import os

print(os.listdir("../input"))



# Exploratory Data Analysis Tools

import numpy as np

import pandas as pd

import datetime as dt

import matplotlib.pyplot as plt

from string import ascii_letters



# Modeling Tools

import fbprophet

import warnings

import itertools

import statsmodels.api as sm

from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from math import sqrt





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
# Opening source file

org_data = pd.read_csv('../input/Monthly_data_cmo.csv')

apmc = org_data # creating a separate dataset to operate on!!

apmc.tail()
apmc.describe()
# Checking Missing Data points, if any

apmc = apmc.dropna(subset=['Year', 'arrivals_in_qtl', 'min_price', 'max_price', 'modal_price'])

apmc.describe() # cleaning, dropping those rows with no values in any one of the above columns
df3 = pd.DataFrame(apmc.groupby(['Commodity', 'Year']).agg('sum')).reset_index()

df3.tail()
trace1 = go.Bar(

    x= df3.loc[df3['Year'] == 2016].Commodity,

    y= df3.loc[df3['Year'] == 2016].arrivals_in_qtl,

    name='2016',

    marker=dict(

        color='yellow', 

        line=dict(

            color='rgb(8,48,107)',

            width=0.2),

        ),

    opacity=0.6

)



trace2 = go.Bar(

    x= df3.loc[df3['Year'] == 2015].Commodity,

    y= df3.loc[df3['Year'] == 2015].arrivals_in_qtl,

    name='2015',

    marker=dict(

        color='brown', 

        line=dict(

            color='rgb(8,48,107)',

            width=0.2),

        ),

    opacity=0.6

)



trace3 = go.Bar(

    x= df3.loc[df3['Year'] == 2014].Commodity,

    y= df3.loc[df3['Year'] == 2014].arrivals_in_qtl,

    name='2014',

    marker=dict(

        color='red', 

        line=dict(

            color='rgb(8,48,107)',

            width=0.2),

        ),

    opacity=0.6

)



layout = go.Layout(

    title='Commodities Purchased (in Volumes) per year'

)



data = [trace1, trace2, trace3]

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename="popular_commodity")
df1 = pd.DataFrame(apmc.groupby(['Commodity']).sum()).reset_index()

df1 = df1[['Commodity', 'arrivals_in_qtl']]

# df1.tail()



df1_a = df1[df1['arrivals_in_qtl'] > 1000000]

df1_a_sort = df1_a.sort_values('arrivals_in_qtl', ascending=True) # for latest python df.sort has been deprecated and updated to df.sort_values





trace = go.Bar(

    x= df1_a_sort.Commodity,

    y= df1_a_sort.arrivals_in_qtl,

    marker=dict(

        color='orange',

    ),

)



layout = go.Layout(

    title='Most Popular Commodity'

)



data = [trace]

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename="popular_Commodity")
# selecting the bottom Selling Commodity in term of < 100  Quintals purchased over 2 years span

df1_b = df1[df1['arrivals_in_qtl']<100]

df1_b_sort = df1_b.sort_values('arrivals_in_qtl', ascending=True) # for latest python df.sort has been deprecated and updated to df.sort_values



trace = go.Bar(

    x= df1_b_sort.Commodity,

    y= df1_b_sort.arrivals_in_qtl,

    marker=dict(

        color='green',

    ),

)



layout = go.Layout(

    title='Least Popular Commodity'

)



data = [trace]

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename="popular_Commodity")
#creating a new column Calculating the total market price

apmc['Total_Market_Price'] = apmc['modal_price'] * apmc['arrivals_in_qtl']
df2 = pd.DataFrame(apmc.groupby(['district_name', 'Year']).agg('mean')).reset_index()

df2.tail(n=6)



trace14 = go.Bar(

    x= df2.loc[df2['Year'] == 2014].district_name,

    y = df2.loc[df2['Year'].isin([2014])].Total_Market_Price,

    name='2014',

    

    marker=dict(

        color='orange', 

        line=dict(

            color='rgb(8,48,107)',

            width=1.5),

        ),

    opacity=1.0

)



trace15 = go.Bar(

    x= df2.loc[df2['Year'] == 2015].district_name,

    y= df2.Total_Market_Price.loc[df2['Year'] == 2015],

    name='2015',

    marker=dict(

        color='purple', 

        line=dict(

            color='rgb(8,48,107)',

            width=1.5),

        ),

    opacity=0.8

)



trace16 = go.Bar(

    x= df2.loc[df2['Year'] == 2016].district_name,

    y= df2.loc[df2['Year'] == 2016].Total_Market_Price,

    name='2016',

    marker=dict(

        color='pink', 

        line=dict(

            color='rgb(8,48,107)',

            width=1.5),

        ),

    opacity=0.6

)



layout = go.Layout(

    title='Market share per district per year'

)



data = [trace14, trace15, trace16]

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename="district_price")
# selecting the top 10 Selling Commodity in term of NUmber of Quintals purchased over 3 years

print(df1.sort_values('arrivals_in_qtl', ascending=False).head(n=10)) # for latest python df.sort has been deprecated and updated to df.sort_values
df4 = apmc

df4 = df4.loc[df4['Commodity'].isin(['Onion','Soybean', 'Potato', 'Cotton', 'Rice(Paddy-Hus)', 'Tomato', ' Coriander  ', 'Methi (Bhaji)','Pigeon Pea (Tur)', 'Maize'])]

# Getting the Sum and the mean of the values

df4_a = pd.DataFrame(df4.groupby(['Commodity', 'date']).agg('sum')).reset_index()

df4_b = pd.DataFrame(df4.groupby(['Commodity', 'date']).agg('mean')).reset_index()
trace21 = go.Scatter(

    x= df4_a.loc[df4_a['Commodity'] == 'Onion'].date,

    y= df4_a.loc[df4_a['Commodity'] == 'Onion'].arrivals_in_qtl,

    mode = 'lines+markers',

    name = 'Onion'

)



trace22 = go.Scatter(

    x= df4_a.loc[df4_a['Commodity'] == 'Soybean'].date,

    y= df4_a.loc[df4_a['Commodity'] == 'Soybean'].arrivals_in_qtl,

    mode = 'lines+markers',

    name = 'Soybean'

)



trace23 = go.Scatter(

    x= df4_a.loc[df4_a['Commodity'] == 'Cotton'].date,

    y= df4_a.loc[df4_a['Commodity'] == 'Cotton'].arrivals_in_qtl,

    mode = 'lines+markers',

    name = 'Cotton'

)



trace24 = go.Scatter(

    x= df4_a.loc[df4_a['Commodity'] == 'Potato'].date,

    y= df4_a.loc[df4_a['Commodity'] == 'Potato'].arrivals_in_qtl,

    mode = 'lines+markers',

    name = 'Potato'

)



trace25 = go.Scatter(

    x= df4_a.loc[df4_a['Commodity'] == 'Rice(Paddy-Hus)'].date,

    y= df4_a.loc[df4_a['Commodity'] == 'Rice(Paddy-Hus)'].arrivals_in_qtl,

    mode = 'lines+markers',

    name = 'Rice(Paddy-Hus)'

)



trace26 = go.Scatter(

    x= df4_a.loc[df4_a['Commodity'] == 'Tomato'].date,

    y= df4_a.loc[df4_a['Commodity'] == 'Tomato'].arrivals_in_qtl,

    mode = 'lines+markers',

    name = 'Tomato'

)



trace27 = go.Scatter(

    x= df4_a.loc[df4_a['Commodity'] == 'Coriander'].date,

    y= df4_a.loc[df4_a['Commodity'] == 'Coriander'].arrivals_in_qtl,

    mode = 'lines+markers',

    name = 'Coriander'

)



trace28 = go.Scatter(

    x= df4_a.loc[df4_a['Commodity'] == 'Methi (Bhaji)'].date,

    y= df4_a.loc[df4_a['Commodity'] == 'Methi (Bhaji)'].arrivals_in_qtl,

    mode = 'lines+markers',

    name = 'Methi (Bhaji)'

)



trace29 = go.Scatter(

    x= df4_a.loc[df4_a['Commodity'] == 'Pigeon Pea (Tur)'].date,

    y= df4_a.loc[df4_a['Commodity'] == 'Pigeon Pea (Tur)'].arrivals_in_qtl,

    mode = 'lines+markers',

    name = 'Pigeon Pea (Tur)'

)



trace20 = go.Scatter(

    x= df4_a.loc[df4_a['Commodity'] == 'Maize'].date,

    y= df4_a.loc[df4_a['Commodity'] == 'Maize'].arrivals_in_qtl,

    mode = 'lines+markers',

    name = 'Maize'

)



trace31 = go.Bar(

    x= df4_b.loc[df4_b['Commodity'] == 'Onion'].date,

    y= df4_b.loc[df4_b['Commodity'] == 'Onion'].modal_price,

    name = 'Onion_Price',

    yaxis='y2',

    opacity=0.5

)



trace32 = go.Bar(

    x= df4_b.loc[df4_b['Commodity'] == 'Soybean'].date,

    y= df4_b.loc[df4_b['Commodity'] == 'Soybean'].modal_price,

    name = 'Soybean_Price',

    yaxis='y2',

    opacity=0.5

)



trace33 = go.Bar(

    x= df4_b.loc[df4_b['Commodity'] == 'Cotton'].date,

    y= df4_b.loc[df4_b['Commodity'] == 'Cotton'].modal_price,

    name = 'Cotton_Price',

    yaxis='y2',

    opacity=0.5

)



trace34 = go.Bar(

    x= df4_b.loc[df4_b['Commodity'] == 'Potato'].date,

    y= df4_b.loc[df4_b['Commodity'] == 'Potato'].modal_price,

    name = 'Potato_Price',

    yaxis='y2',

    opacity=0.5

)



trace35 = go.Bar(

    x= df4_b.loc[df4_b['Commodity'] == 'Rice(Paddy-Hus)'].date,

    y= df4_b.loc[df4_b['Commodity'] == 'Rice(Paddy-Hus)'].modal_price,

    name = 'Rice(Paddy-Hus)_Price',

    yaxis='y2',

    opacity=0.5

)

    

trace36 = go.Bar(

    x= df4_b.loc[df4_b['Commodity'] == 'Tomato'].date,

    y= df4_b.loc[df4_b['Commodity'] == 'Tomato'].modal_price,

    name = 'Tomato_Price',

    yaxis='y2',

    opacity=0.5

)

    

trace37 = go.Bar(

    x= df4_b.loc[df4_b['Commodity'] == 'Coriander'].date,

    y= df4_b.loc[df4_b['Commodity'] == 'Coriander'].modal_price,

    name = 'Coriander_Price',

    yaxis='y2',

    opacity=0.5

)

    

trace38 = go.Bar(

    x= df4_b.loc[df4_b['Commodity'] == 'Methi (Bhaji)'].date,

    y= df4_b.loc[df4_b['Commodity'] == 'Methi (Bhaji)'].modal_price,

    name = 'Methi (Bhaji)_Price',

    yaxis='y2',

    opacity=0.5

)

    

trace39 = go.Bar(

    x= df4_b.loc[df4_b['Commodity'] == 'Pigeon Pea (Tur)'].date,

    y= df4_b.loc[df4_b['Commodity'] == 'Pigeon Pea (Tur)'].modal_price,

    name = 'Pigeon Pea (Tur)_Price',

    opacity=0.5,

    yaxis='y2'

)

    

trace30 = go.Bar(

    x= df4_b.loc[df4_b['Commodity'] == 'Maize'].date,

    y= df4_b.loc[df4_b['Commodity'] == 'Maize'].modal_price,

    name = 'Maize_Price', 

    opacity=0.5,

    yaxis='y2'

)





data = [trace21, trace22, trace23, trace24, trace25, trace26, trace27, trace28, trace29, trace20,

        trace31, trace32, trace33, trace34, trace35, trace36, trace37, trace38, trace39, trace30]

    



layout = go.Layout(

    legend=dict(orientation="h"),

    

    title='Monthly Chart : Top 10 Commodity, Economics v/s Quantity',

    yaxis=dict(

        title='Quintals_Purchased_in_Maharashtra'

    ),

    yaxis2=dict(

        title='Average_Modal_Price_Per_Quintal(INR)',

        titlefont=dict(

            color='rgb(148, 103, 189)'

        ),

        tickfont=dict(

            color='rgb(148, 103, 189)'

        ),

        overlaying='y',

        side='right'

    )

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename="popular_commodity")
print(df1.sort_values('arrivals_in_qtl', ascending=True).head()) # for latest python df.sort has been deprecated and updated to df.sort_values
df0 = apmc

df0 = df0.loc[df0['Commodity'].isin(['CASTOR SEED','LEAFY VEGETABLE', 'Baru Seed', 'Jui', 'Papnas', 'MUSTARD', 

                                           'SARSAV', 'Terda','GOATS', 'Kalvad', 'Peer', 'NOLKOL', 'Plum',

                                          'GROUNDNUT PODS (WET)', 'Karvand', 'He Buffalo'])]

# Getting the Sum and the mean of the values

df0_a = pd.DataFrame(df0.groupby(['Commodity', 'date']).agg('sum')).reset_index()

df0_b = pd.DataFrame(df0.groupby(['Commodity', 'date']).agg('mean')).reset_index()
data = [

    go.Bar(

        x=df0.loc[df0['Commodity'] == 'CASTOR SEED'].district_name, y=df0.loc[df0['Commodity'] == 'CASTOR SEED'].arrivals_in_qtl,

        name = 'Castor seed', opacity=0.5),

    go.Bar(

        x=df0.loc[df0['Commodity'] == 'GOATS'].district_name, y=df0.loc[df0['Commodity'] == 'GOATS'].arrivals_in_qtl,

        name = 'Goats', opacity=0.5

    ),

    go.Bar(

        x=df0.loc[df0['Commodity'] == 'Karvand'].district_name, y=df0.loc[df0['Commodity'] == 'Karvand'].arrivals_in_qtl,

        name = 'Karvand', opacity=0.5

    ),

    go.Bar(

        x=df0.loc[df0['Commodity'] == 'He Buffalo'].district_name, y=df0.loc[df0['Commodity'] == 'He Buffalo'].arrivals_in_qtl,

        name = 'Buffalo', opacity=0.5

    ),

    go.Bar(

        x=df0.loc[df0['Commodity'] == 'LEAFY VEGETABLE'].district_name, y=df0.loc[df0['Commodity'] == 'LEAFY VEGETABLE'].arrivals_in_qtl,

        name = 'Leafy Veggie', opacity=0.5

    ),

    go.Bar(

        x=df0.loc[df0['Commodity'] == 'Baru Seed'].district_name, y=df0.loc[df0['Commodity'] == 'Baru Seed'].arrivals_in_qtl,

        name = 'Baru Seed', opacity=0.5

    ),

    go.Bar(

        x=df0.loc[df0['Commodity'] == 'Jui'].district_name, y=df0.loc[df0['Commodity'] == 'Jui'].arrivals_in_qtl,

        name = 'Jui', opacity=0.5

    ),

    go.Bar(

        x=df0.loc[df0['Commodity'] == 'Papnas'].district_name, y=df0.loc[df0['Commodity'] == 'Papnas'].arrivals_in_qtl,

        name = 'Papnas', opacity=0.5

    ),

    go.Bar(

        x=df0.loc[df0['Commodity'] == 'MUSTARD'].district_name, y=df0.loc[df0['Commodity'] == 'MUSTARD'].arrivals_in_qtl,

        name = 'Mustard', opacity=0.5

    ),

    go.Bar(

        x=df0.loc[df0['Commodity'] == 'SARSAV'].district_name, y=df0.loc[df0['Commodity'] == 'SARSAV'].arrivals_in_qtl,

        name = 'Sarsav', opacity=0.5

    ),

    go.Bar(

        x=df0.loc[df0['Commodity'] == 'Terda'].district_name, y=df0.loc[df0['Commodity'] == 'Terda'].arrivals_in_qtl,

        name = 'Terda', opacity=0.5

    ),

    go.Bar(

        x=df0.loc[df0['Commodity'] == 'Kalvad'].district_name, y=df0.loc[df0['Commodity'] == 'Kalvad'].arrivals_in_qtl,

        name = 'Kalvad', opacity=0.5

    ),

    go.Bar(

        x=df0.loc[df0['Commodity'] == 'Peer'].district_name, y=df0.loc[df0['Commodity'] == 'Peer'].arrivals_in_qtl,

        name = 'Peer', opacity=0.5

    ),

    go.Bar(

        x=df0.loc[df0['Commodity'] == 'Plum'].district_name, y=df0.loc[df0['Commodity'] == 'Plum'].arrivals_in_qtl,

        name = 'Plum', opacity=0.5

    ),

    go.Bar(

        x=df0.loc[df0['Commodity'] == 'NOLKOL'].district_name, y=df0.loc[df0['Commodity'] == 'NOLKOL'].arrivals_in_qtl,

        name = 'Nolkol', opacity=0.5

    ),

    go.Bar(

        x=df0.loc[df0['Commodity'] == 'GROUNDNUT PODS (WET)'].district_name, y=df0.loc[df0['Commodity'] == 'GROUNDNUT PODS (WET)'].arrivals_in_qtl,

        name = 'Groundnut Pods', opacity=0.5

    )



]





layout = go.Layout(

    barmode='stack',

    title='Least Popular Commodities and their Purchase District',

    yaxis=dict(

        title='Quintals_Purchased'

    )

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename="popular_commodity")
df6 = apmc

df6_a = pd.DataFrame(df6.groupby(['Commodity', 'district_name']).agg('mean')).reset_index()
Commodity = 'Maize'



trace00 = go.Scatter(

    x= df6_a.loc[df6_a['Commodity'] == Commodity].district_name,

    y= df6_a.loc[df6_a['Commodity'] == Commodity].max_price,

    mode = 'lines+markers',

    line = dict(

        color = ('rgb(22, 96, 167)'),

        width = 2,

        dash = 'dot'),

    name = 'Price_MAX', 

    opacity=0.5

)



trace01 = go.Scatter(

    x= df6_a.loc[df6_a['Commodity'] == Commodity].district_name,

    y= df6_a.loc[df6_a['Commodity'] == Commodity].modal_price,

    mode = 'lines+markers',

    line = dict(

        color = ('Red'),

        width = 2),

    name = 'Price_MODE',

    opacity=1.0

)



trace02 = go.Scatter(

    x= df6_a.loc[df6_a['Commodity'] == Commodity].district_name,

    y= df6_a.loc[df6_a['Commodity'] == Commodity].min_price,

    mode = 'lines+markers',

    line = dict(

        color = ('rgb(22, 96, 167)'),

        width = 2,

        dash = 'dot'),

    name = 'Price_MIN', 

    opacity=0.5

)



trace03 = go.Bar(

    x= df6_a.loc[df6_a['Commodity'] == Commodity].district_name,

    y= df6_a.loc[df6_a['Commodity'] == Commodity].arrivals_in_qtl,

    name = 'Quantity', 

    opacity=0.2,

    yaxis='y2'

)





data = [trace00, trace01, trace02, trace03]



    



layout = go.Layout(

    legend=dict(orientation="v"),

    

    title='Price Range Chart district-wise',

    yaxis=dict(

        title='Price per Quintal'

    ),

    yaxis2=dict(

        title='Average_Quintal',

        titlefont=dict(

            color='rgb(148, 103, 189)'

        ),

        tickfont=dict(

            color='rgb(148, 103, 189)'

        ),

        overlaying='y',

        side='right'

    )

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename="popular_commodity")
df7 = apmc

df7_a = df7.loc[df7['Commodity'].isin(['CASTOR SEED','LEAFY VEGETABLE', 'Baru Seed', 'Jui', 'Papnas', 'MUSTARD', 

                                           'SARSAV', 'Terda','GOATS', 'Kalvad', 'Peer', 'NOLKOL', 'Plum',

                                          'GROUNDNUT PODS (WET)', 'Karvand', 'He Buffalo'])]



df7_a= df7_a[[ 'Year', 'date' , 'arrivals_in_qtl', 'modal_price', 'min_price', 'max_price', 'Total_Market_Price' ]]



sns.set(style="white")



# Generate a large random dataset

d = df7_a



# Compute the correlation matrix

corr = d.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
df7_b = df7.loc[df7['Commodity'].isin(['Onion', 'Soybean','Potato', 'Cotton', 'Rice(Paddy-Hus)'])]



df7_b= df7_b[[ 'Year', 'date' , 'arrivals_in_qtl', 'modal_price', 'min_price', 'max_price', 'Total_Market_Price' ]]



sns.set(style="white")



# Generate a large random dataset

d = df7_b



# Compute the correlation matrix

corr = d.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
df8 = apmc

df8 = df8.loc[df8['district_name'].isin(['Mumbai','Sangli', 'Wardha', 'Pune', 'Thane', 'Wasim', 'Nasik','Yewatmal', 'Gondiya'])]

df8_a = pd.DataFrame(df8.groupby(['district_name', 'Month']).agg('mean')).reset_index()
a4_dims = (15, 8.27)

fig, ax = pyplot.subplots(figsize=a4_dims)



# sns.set(style="whitegrid")

p = sns.violinplot(ax = ax,

                   data=df8_a,

                   x = 'district_name',

                   y = 'Total_Market_Price', bw=0.5, saturation = 1.25, width = 1.1

                   )

plt.show()
df5 = apmc

df5 = df5.loc[df5['Commodity'].isin(['Pigeon Pea (Tur)'])]

df5 = pd.DataFrame(df5.groupby(['date']).agg('mean')).reset_index()

df5 = df5 [['date', 'modal_price']]

df5.tail()


trace00 = go.Scatter(

    x= df5.date,

    y= df5.modal_price,

    mode = 'lines+markers',

)



layout = go.Layout(

    title='Pigeon_Pea Prices across the month'

)



data = [trace00]

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename="popular_commodity")
#Preparing the dataset:   



df5.to_csv('forecast_APMC.csv')

# df5.dtypes # our datatypes for date column is in objects, we shall convert it to datetime 

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')



df5 = pd.read_csv('forecast_APMC.csv', parse_dates=['date'], index_col='date',date_parser=dateparse) # indexed the date column

df5 = df5[['modal_price']]

df5.head()
ts = df5

ticks = ts.loc[:, ['modal_price']]

upsampled = ticks.modal_price.resample('7D', how = 'last') # upsampling it to weekly data points

interpolated = upsampled.interpolate(method='spline', order=2) # for more smoothened curve values. 

# print(interpolated )

plt.plot(interpolated, color='red', label = 'Interpolated weekly data')

plt.plot(ts, color='green', label = 'Original Monthly data')

plt.show()
# split into train and test sets

 

start_train = 0          # add variable integer for location of data points for start of training

end_train = 100          # add variable integer for location of data points for end of training

start_test = 100        # add variable integer for location of data points for start of testing

end_test = 112          # add variable integer for location of data points for end of testing



X = interpolated



train, test = X[start_train:end_train], X[start_test:end_test]

# train, test = train_test_split(interpolated, test_size=0.2)       # splitting into train and test

history = [interpolated for interpolated in train]                  # creating a historical memory bin

predictions_forecast = list()                                    # creating an empty list for prediction forecasts

predictions_CI = list()                                          # creating an empty list for 95% Confidence Intervals

predictions_STD = list()  



test = test.tolist()    # converting test to list...

train_list = train.tolist()  # converting train to list...
# walk-forward validation

for t in range(len(test)):

	# fit model

	model = ARIMA(history, order=(2,1,2)) # ideal order is taken from the AIC test above

	model_fit = model.fit()



# one step forecast

# 	yhat = model_fit.forecast()[0]

	forecast, stderr, conf = model_fit.forecast(steps = 1, alpha = 0.05) 



# store forecast and ob

	predictions_forecast.append(forecast)

	predictions_STD.append(stderr)

	predictions_CI.append(conf)

	history.append(test[t])
residual_error = [predictions_forecast-test for predictions_forecast,test in zip(predictions_forecast,test)]

plt.plot(residual_error, color='orange', label='Residual Errors')
# plot forecasts against actual outcomes

plt.figure(figsize=(8,4))

plt.plot(test, color = 'blue', label = 'Test Data')

plt.plot(predictions_forecast, color='red' , label='Predicted Data')



plt.grid(True)

plt.xticks(rotation=90)

plt.xlabel("Units")

plt.ylabel("Power Demand (kW)")

# plt.ylim(ymin=0)

plt.legend()

plt.show()
# data curating

prophet_data = pd.read_csv('forecast_APMC.csv', parse_dates=['date'], index_col='date', date_parser=dateparse)

prophet_data =  prophet_data[[ 'modal_price']]

# prophet_data.tail()
#UPSAMPLING

ticks = prophet_data.loc[:, ['modal_price']]

upsampled = ticks.modal_price.resample('D', how = 'last') # upsampling it to daily data points

data = upsampled.interpolate(method='spline', order=2) # for more smoothened curve values. 

# print(interpolated )

data.plot()

plt.show()
data=data.reset_index()

data = data[['date', 'modal_price']]

data.tail()
# Prophet requires columns ds (Date) and y (value)

prc = data.rename(columns={'date': 'ds', 'modal_price': 'y'})
# Make the prophet model and fit on the data

prc_prophet = fbprophet.Prophet(changepoint_prior_scale=0.90, seasonality_prior_scale = 0.99) # keeping High sensitivity to seasonal variability and changing points

prc_prophet.fit(prc) 
# Make a future dataframe for next 90 days (3 months)

prc_forecast = prc_prophet.make_future_dataframe(periods=90, freq= "d") 

# Make predictions

prc_forecast = prc_prophet.predict(prc_forecast)
prc_forecast = prc_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n=90) 

prc_forecast.tail()



# yhat --> forecasted Modal Price

# yhat_lower and yhat_upper --> forecasted 95% confidence interval range of modal price
prc_prophet.plot(prc_forecast, xlabel = 'Date', ylabel = 'Commodity_Price')

plt.ylim(ymin=0);

plt.title('Price Predictions');