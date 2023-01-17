# Import Libraries
import numpy as np 
import pandas as pd 
import os
from plotly.offline import init_notebook_mode, iplot
import plotly.plotly as py
import plotly.figure_factory as FF
import plotly.plotly as py
from plotly import tools
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.features import DivIcon
from folium.plugins import HeatMap
import warnings
import datetime
import squarify
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/MELBOURNE_HOUSE_PRICES_LESS.csv")
df.head()
df.isnull().sum().sort_values(ascending=False)
no_missing_df = df[df['Price'].notnull()]
no_missing_df.isnull().sum().sort_values(ascending=False)
df = no_missing_df.copy()

df['Regionname'].unique()
# Price by Region
all_regions = df['Price'].values
northern_metropolitan = df['Price'].loc[df['Regionname'] == 'Northern Metropolitan'].values
southern_metropolitan = df['Price'].loc[df['Regionname'] == 'Southern Metropolitan'].values
eastern_metropolitan = df['Price'].loc[df['Regionname'] == 'Eastern Metropolitan'].values
western_metropolitan = df['Price'].loc[df['Regionname'] == 'Western Metropolitan'].values
southeastern_metropolitan = df['Price'].loc[df['Regionname'] == 'South-Eastern Metropolitan'].values
northern_victoria = df['Price'].loc[df['Regionname'] == 'Northern Victoria'].values
eastern_victoria = df['Price'].loc[df['Regionname'] == 'Eastern Victoria'].values
western_victoria = df['Price'].loc[df['Regionname'] == 'Western Victoria'].values
gaussian_distribution = np.log(df['Price'].values)


# Histograms
overall_price_plot = go.Histogram(
    x=all_regions,
    histnorm='count', 
    name='All Regions',
    marker=dict(
        color='#6E6E6E'
    )
)


northern_metropolitan_plot = go.Histogram(
    x=northern_metropolitan,
    histnorm='count', 
    name='Northern Metropolitan',
    marker=dict(
        color='#2E9AFE'
    )
)

southern_metropolitan_plot = go.Histogram(
    x=southern_metropolitan,
    histnorm='count', 
    name='Southern Metropolitan',
    marker=dict(
        color='#FA5858'
    )
)


eastern_metropolitan_plot = go.Histogram(
    x=eastern_metropolitan,
    histnorm='count', 
    name='Eastern Metropolitan',
    marker=dict(
        color='#81F781'
    )
)

western_metropolitan_plot = go.Histogram(
    x=western_metropolitan,
    histnorm='count', 
    name='Western Metropolitan',
    marker=dict(
        color='#BE81F7'
    )
)

southeastern_metropolitan_plot = go.Histogram(
    x=southeastern_metropolitan,
    histnorm='count', 
    name='SouthEastern Metropolitan',
    marker=dict(
        color='#FE9A2E'
    )
)

northern_victoria_plot = go.Histogram(
    x=northern_victoria,
    histnorm='count', 
    name='Northern Victoria',
    marker=dict(
        color='#04B4AE'
    )
)

eastern_victoria_plot = go.Histogram(
    x=eastern_victoria,
    histnorm='count', 
    name='Eastern Victoria',
    marker=dict(
        color='#088A08'
    )
)


western_victoria_plot = go.Histogram(
    x=western_victoria,
    histnorm='count', 
    name='Western Victoria',
    marker=dict(
        color='#8A0886'
    )
)

gaussian_distribution_plot = go.Histogram(
    x=gaussian_distribution,
    histnorm='probability',
    name='Gaussian Distribution',
    marker=dict(
        color='#800000'
    )
)

fig = tools.make_subplots(rows=6, cols=2, print_grid=False, specs=[[{'colspan': 2}, None], [{}, {}], [{}, {}], [{}, {}], [{}, {}], [{'colspan': 2}, None]],
                         subplot_titles=(
                             'Overall Price Distribution',
                             'Northern Metropolitan',
                             'Southern Metropolitan',
                             'Eastern Metropolitan',
                             'Western Metropolitan',
                             'SouthEastern Metropolitan',
                             'Northern Victoria',
                             'Eastern Victoria',
                             'Western Victoria',
                             'Gaussian Distribution of Price'
                             ))
fig.append_trace(overall_price_plot, 1, 1)
fig.append_trace(northern_metropolitan_plot, 2, 1)
fig.append_trace(southern_metropolitan_plot, 2, 2)
fig.append_trace(eastern_metropolitan_plot, 3, 1)
fig.append_trace(western_metropolitan_plot, 3, 2)
fig.append_trace(southeastern_metropolitan_plot, 4, 1)
fig.append_trace(northern_victoria_plot, 4, 2)
fig.append_trace(eastern_victoria_plot, 5, 1)
fig.append_trace(western_victoria_plot, 5, 2)
fig.append_trace(gaussian_distribution_plot, 6, 1)

fig['layout'].update(showlegend=False, title="Price Distributions by Region",
                    height=1200, width=800)
iplot(fig, filename='custom-sized-subplot-with-subplot-titles')
df.head()
df['Date'].unique()
print(df['Date'].head())
df['Date'].dtype
# Convert to Datetime Format
df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
# Analyze seasonality per month (and answer the question in which month there is more demand)
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
# Subplot #1
total_sales = df['Price'].sum()

def month_sales(df, month, sales=total_sales):
    share_month_sales = df['Price'].loc[df['Month'] == month].sum()/sales
    return share_month_sales

january_sales = month_sales(df, 1)
february_sales = month_sales(df, 2)
march_sales = month_sales(df, 3)
april_sales = month_sales(df, 4)
may_sales = month_sales(df, 5)
june_sales = month_sales(df, 6)
july_sales = month_sales(df, 7)
august_sales = month_sales(df, 8)
september_sales = month_sales(df, 9)
october_sales = month_sales(df, 10)
november_sales = month_sales(df, 11)
december_sales = month_sales(df, 12)

month_total_sales = [january_sales, february_sales, march_sales, april_sales,
                     may_sales, june_sales, july_sales, august_sales, 
                     september_sales, october_sales, november_sales, december_sales]

labels = ['January', 'February', 'March', 'April',
          'May', 'June', 'July', 'August', 'September', 
          'October', 'November', 'December']


colors = ['#ffb4da', '#b4b4ff', '#daffb4', '#fbab60', '#fa8072', '#FA6006',
          '#FDB603', '#639702', '#dacde6', '#faec72', '#9ab973', '#87cefa']

pie_plot = go.Pie(labels=labels, values=month_total_sales,
               hoverinfo='label+percent',
               marker=dict(colors=colors,
                           line=dict(color='#000000', width=2)))

data = [pie_plot]

layout = go.Layout(
    title="Share of Sales by Month"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='lowest-oecd-votes-cast')
# Establish the sum for each month and year
def month_year_sales(df, month, year):
    double_conditional = df['Price'].loc[(df['Month'] == month) & (df['Year'] == year)].sum()
    return double_conditional

# Sales 2016
january_2016 = month_year_sales(df, 1, 2016)
february_2016 = month_year_sales(df, 2, 2016)
march_2016 = month_year_sales(df, 3, 2016)
april_2016 = month_year_sales(df, 4, 2016)
may_2016 = month_year_sales(df, 5, 2016)
june_2016 = month_year_sales(df, 6, 2016)
july_2016 = month_year_sales(df, 7, 2016)
august_2016 = month_year_sales(df, 8, 2016)
september_2016 = month_year_sales(df, 9, 2016)
october_2016 = month_year_sales(df, 10, 2016)
november_2016 = month_year_sales(df, 11, 2016)
december_2016 = month_year_sales(df, 12, 2016)

# Sales 2017
january_2017 = month_year_sales(df, 1, 2017)
february_2017 = month_year_sales(df, 2, 2017)
march_2017 = month_year_sales(df, 3, 2017)
april_2017 = month_year_sales(df, 4, 2017)
may_2017 = month_year_sales(df, 5, 2017)
june_2017 = month_year_sales(df, 6, 2017)
july_2017 = month_year_sales(df, 7, 2017)
august_2017 = month_year_sales(df, 8, 2017)
september_2017 = month_year_sales(df, 9, 2017)
october_2017 = month_year_sales(df, 10, 2017)
november_2017 = month_year_sales(df, 11, 2017)
december_2017 = month_year_sales(df, 12, 2017)

# Sales 2018 (Until May)
january_2018 = month_year_sales(df, 1, 2018)
february_2018 = month_year_sales(df, 2, 2018)
march_2018 = month_year_sales(df, 3, 2018)
april_2018 = month_year_sales(df, 4, 2018)
may_2018 = month_year_sales(df, 5, 2018)


# List of values
lst_2016 = [january_2016, february_2016, march_2016, april_2016, 
           may_2016, june_2016, july_2016, august_2016, 
           september_2016, october_2016, november_2016, december_2016]

lst_2017 = [january_2017, february_2017, march_2017, april_2017, 
           may_2017, june_2017, july_2017, august_2017, 
           september_2017, october_2017, november_2017, december_2017]


lst_2018 = [january_2018, february_2018, march_2018, april_2018, 
           may_2018]


plot_2016 = go.Scatter(
    x=lst_2016,
    y=labels,
    xaxis='x2',
    yaxis='y2',
    mode='markers',
    name='2016',
    marker=dict(
        color='rgba(0, 128, 128, 0.95)',
        line=dict(
            color='rgba(56, 56, 56, 1)',
            width=1.5,
        ),
        symbol='circle',
        size=16,
    )
)


plot_2017 = go.Scatter(
    x=lst_2017,
    y=labels,
    xaxis='x2',
    yaxis='y2',
    mode='markers',
    name='2017',
    marker=dict(
        color='rgba(255, 72, 72, 0.95)',
        line=dict(
            color='rgba(56, 56, 56, 1)',
            width=1.5,
        ),
        symbol='circle',
        size=16,
    )
)

plot_2018 = go.Scatter(
    x=lst_2018,
    y=labels,
    xaxis='x2',
    yaxis='y2',
    mode='markers',
    name='2018',
    marker=dict(
        color='rgba(72, 255, 72, 0.95)',
        line=dict(
            color='rgba(56, 56, 56, 1)',
            width=1.5,
        ),
        symbol='circle',
        size=16,
    )
)

data = [plot_2016, plot_2017, plot_2018]

layout = go.Layout(
    title="Sales by Month for the Years <br> (2016, 2017, 2018)",
    xaxis=dict(
        showgrid=False,
        showline=True,
        linecolor='rgb(102, 102, 102)',
        titlefont=dict(
            color='rgb(204, 204, 204)'
        ),
        tickfont=dict(
            color='rgb(102, 102, 102)',
        ),
        autotick=False,
        dtick=10,
        ticks='outside',
        tickcolor='rgb(102, 102, 102)',
    ),
    margin=dict(
        l=140,
        r=40,
        b=50,
        t=80
    ),
    legend=dict(
        font=dict(
            size=10,
        ),
        yanchor='left',
        xanchor='left',
    ),
    width=800,
    height=600,
    paper_bgcolor='rgb(255, 255, 224)',
    plot_bgcolor='rgb(255, 255, 246)',
    hovermode='closest',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='multiple-subplots')
# Create the Mean for each of the months!
def avg_price_sold(df, month, year):
    avg_p = round(np.mean(df['Price'].loc[(df['Month'] == month) & (df['Year'] == year)].values), 2)
    return avg_p

# 2016 months
jan_2016 = avg_price_sold(df, 1, 2016)
feb_2016 = avg_price_sold(df, 2, 2016)
mar_2016 = avg_price_sold(df, 3, 2016)
apr_2016 = avg_price_sold(df, 4, 2016)
may_2016 = avg_price_sold(df, 5, 2016)
june_2016 = avg_price_sold(df ,6, 2016)
july_2016 = avg_price_sold(df, 7, 2016)
aug_2016 = avg_price_sold(df, 8, 2016)
sep_2016 = avg_price_sold(df, 9, 2016)
oct_2016 = avg_price_sold(df, 10, 2016)
nov_2016 = avg_price_sold(df, 11, 2016)
dec_2016 = avg_price_sold(df, 12, 2016)

# 2017 months
jan_2017 = avg_price_sold(df, 1, 2017)
feb_2017 = avg_price_sold(df, 2, 2017)
mar_2017 = avg_price_sold(df, 3, 2017)
apr_2017 = avg_price_sold(df, 4, 2017)
may_2017 = avg_price_sold(df, 5, 2017)
june_2017 = avg_price_sold(df ,6, 2017)
july_2017 = avg_price_sold(df, 7, 2017)
aug_2017 = avg_price_sold(df, 8, 2017)
sep_2017 = avg_price_sold(df, 9, 2017)
oct_2017 = avg_price_sold(df, 10, 2017)
nov_2017 = avg_price_sold(df, 11, 2017)
dec_2017 = avg_price_sold(df, 12, 2017)

# 2018 Months (Until May only)
jan_2018 = avg_price_sold(df, 1, 2018)
feb_2018 = avg_price_sold(df, 2, 2018)
mar_2018 = avg_price_sold(df, 3, 2018)
apr_2018 = avg_price_sold(df, 4, 2018)
may_2018 = avg_price_sold(df, 5, 2018)

# Lists for each year
# 2016
lst_2016_avg = [jan_2016, feb_2016, mar_2016, apr_2016, may_2016, june_2016, july_2016, aug_2016, sep_2016,
               oct_2016, nov_2016, dec_2016]
# 2017
lst_2017_avg = [jan_2017, feb_2017, mar_2017, apr_2017, may_2017, june_2017, july_2017, aug_2017, sep_2017,
               oct_2017, nov_2017, dec_2017]
# 2018
lst_2018_avg = [jan_2018, feb_2018, mar_2018, apr_2018, may_2018]



# Replace Nan for Zero
lst_2016_avg[2] = 0
lst_2017_avg[0] = 0
lst_2018_avg[0] = jan_2018
# Radar Charts (Three of them) with the distribution per Month of Sales
month_labels = ['January', 'February', 'March', 'April',
                'May', 'June', 'July', 'August', 'September', 
                'October', 'November', 'December']

data = [
    go.Scatterpolar(
        mode='line+markers',
        r = lst_2016_avg,
        theta = month_labels,
        fill = 'toself',
        name="2016",
        line=dict(
            color="rgba(0, 128, 128, 0.95)"
        ),
        marker=dict(
            color="rgba(0, 74, 147, 1)",
            symbol="square",
            size=8
        ),
        subplot = "polar"
    ),
    go.Scatterpolar(
        mode='line+markers',
        r = lst_2017_avg,
        theta = month_labels,
        fill = 'toself',
        name="2017",
        line=dict(
            color="rgba(255, 72, 72, 0.95)"
        ),
        marker=dict(
            color="rgba(219, 0, 0, 1)",
            symbol="square",
            size=8
        ),
        subplot = "polar2"
    ),
    go.Scatterpolar(
        mode='line+markers',
        r = lst_2018_avg,
        theta = month_labels,
        fill = 'toself',
        name="2018",
        line=dict(
            color="rgba(72, 255, 72, 0.95)"
        ),
        marker=dict(
            color="rgba(0, 147, 74, 1)",
            symbol="square",
            size=8
        ),
        subplot = "polar3"
    )
]

layout = go.Layout(
    title="Average House Price <br> (Distribution per Month)",
    showlegend = False,
     paper_bgcolor = "rgb(255, 255, 224)",
    polar = dict(
      domain = dict(
        x = [0,0.3],
        y = [0,1]
      ),
      radialaxis = dict(
        tickfont = dict(
          size = 6
        )
      ),
      angularaxis = dict(
        tickfont = dict(
          size = 6
        ),
        rotation = 90,
        direction = "counterclockwise"
      )
    ),
    polar2 = dict(
      domain = dict(
        x = [0.35,0.65],
        y = [0,1]
      ),
      radialaxis = dict(
        tickfont = dict(
          size = 6
        )
      ),
      angularaxis = dict(
        tickfont = dict(
          size = 6
        ),
        rotation = 85,
        direction = "clockwise"
      ),
    ),
    polar3 = dict(
      domain = dict(
        x = [0.7, 1],
        y = [0,1]
      ),
      radialaxis = dict(
        tickfont = dict(
          size = 6
        )
      ),
      angularaxis = dict(
        tickfont = dict(
          size = 6
        ),
        rotation = 90,
        direction = "clockwise"
      ),
    ))

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='polar/directions')
df['Month'].value_counts()
df['Season'] = np.nan
lst = [df]

for column in lst:
    column.loc[(column['Month'] > 2) & (column['Month'] <= 5), 'Season'] = 'Spring'
    column.loc[(column['Month'] > 5) & (column['Month'] <= 8), 'Season'] = 'Summer'
    column.loc[(column['Month'] > 8) & (column['Month'] <= 11), 'Season'] = 'Autumn'
    column.loc[column['Month'] <= 2, 'Season'] = 'Winter'
    column.loc[column['Month'] == 12, 'Season'] = 'Winter'
    
df['Season'].value_counts()
# Perform a ViolinPlot with the distribution of price for each Season...
# Types: 
# h- house, cottage, villa, semi, terrace
# u - unit, duplex
# t - townhouse,


fig = {
    "data": [
        {
            "type": 'violin',
            "x": df['Type'] [ df['Season'] == 'Spring' ],
            "y": df['Price'] [ df['Season'] == 'Spring' ],
            "legendgroup": 'Spring',
            "scalegroup": 'Spring',
            "name": 'Spring',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": '#6cff6c'
            }
        },
        {
            "type": 'violin',
            "x": df['Type'] [ df['Season'] == 'Summer' ],
            "y": df['Price'] [ df['Season'] == 'Summer' ],
            "legendgroup": 'Summer',
            "scalegroup": 'Summer',
            "name": 'Summer',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": '#ff6961'
            }
        },
        {
            "type": 'violin',
            "x": df['Type'] [ df['Season'] == 'Autumn' ],
            "y": df['Price'] [ df['Season'] == 'Autumn' ],
            "legendgroup": 'Autumn',
            "scalegroup": 'Autumn',
            "name": 'Autumn',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": '#9a5755'
            }
        },
        {
            "type": 'violin',
            "x": df['Type'] [ df['Season'] == 'Winter' ],
            "y": df['Price'] [ df['Season'] == 'Winter' ],
            "legendgroup": 'Winter',
            "scalegroup": 'Winter',
            "name": 'Winter',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": '#70c8cb'
            }
        },
    ],
    "layout" : {
        "title": "Distribution of Price by Type <br> <sub> Measuring Seasonality Activity </sub>",
        "yaxis": {
            "zeroline": False,
        },
        "violinmode": "group"
    }
}


iplot(fig, filename = 'violin/grouped', validate = False)
# Distribution of Price per Season and Year (2016, 2017 and Both)
def distribution_seasonality(df, season, year):
    dis_season = np.around(np.sum(df['Price'].loc[(df['Season'] == season) & (df['Year'] == year)].values), 2)
    per_season =  round((dis_season/df['Price'].loc[df['Year'] == year].sum()) * 100, 2)
    return per_season

# year 2016
spring_2016 = distribution_seasonality(df, 'Spring', 2016)
summer_2016 = distribution_seasonality(df, 'Summer', 2016)
autumn_2016 = distribution_seasonality(df, 'Summer', 2016)
winter_2016 = distribution_seasonality(df, 'Summer', 2016)

# year 2017 
spring_2017 = distribution_seasonality(df, 'Spring', 2017)
summer_2017 = distribution_seasonality(df, 'Summer', 2017)
autumn_2017 = distribution_seasonality(df, 'Summer', 2017)
winter_2017 = distribution_seasonality(df, 'Summer', 2017)


# Overall distribution
overall_spring = (df['Price'].loc[df['Season'] == 'Spring'].sum()/ df['Price'].sum()) * 100
overall_summer = (df['Price'].loc[df['Season'] == 'Summer'].sum() / df['Price'].sum()) * 100
overall_autumn = (df['Price'].loc[df['Season'] == 'Autumn'].sum()/df['Price'].sum()) * 100
overall_winter = (df['Price'].loc[df['Season'] == 'Winter'].sum()/df['Price'].sum()) * 100
colors = ['#6cff6c', '#ff6961', '#9a5755', '#70c8cb']

fig = {
  "data": [
    {
      "values": [overall_spring, overall_summer, overall_autumn, overall_winter],
      "labels": [
        "Spring",
        "Summer",
        "Autumn",
        "Winter"
      ],
    'marker': {'colors': colors},
      "domain": {"x": [0, .28]},
      "name": "Overall Sales",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },
      {
      "values": [spring_2016, summer_2016, autumn_2016, winter_2016],
      "labels": [
        "Spring",
        "Summer",
        "Autumn",
        "Winter"
      ],
    'marker': {'colors': colors},
      "text":"2016",
      "textposition":"inside",
      "domain": {"x": [.33, .61]},
      "name": "2016 <br> Sales",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },
      {
      "values": [spring_2017, summer_2017, autumn_2017, winter_2017],
      "labels": [
        "Spring",
        "Summer",
        "Autumn",
        "Winter"
      ],
        'marker': {'colors': colors},
      "text":"2017",
      "textposition":"inside",
      "domain": {"x": [.66, .94]},
      "name": "2017<br> Sales",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
     "layout": {
        "title":"Sales per Season",
        "annotations": [
            {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "Overall<br>Sales",
                "x": 0.105,
                "y": 0.5
            },
            {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "2016<br>Sales",
                "x": 0.47,
                "y": 0.5
            },
            {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "2017<br>Sales",
                "x": 0.825,
                "y": 0.5
            }
        ]
    }
}

iplot(fig, filename='donut')
df.head()
# Let's make percent change for each of the month during the different years subplot 
# Second subplot will be percent change of the total during the years.

# Percent Change formula: (New Value - Old Value) / Old Value
# We start with the value for year 2016 and see how it change towards the year 2017
from numpy import inf
from scipy import *

def percent_change(df, old_val, new_val):
    per_change = (new_val - old_val)/old_val * 100
    rounded_per_change = float("{0:.2f}".format(per_change))
    return rounded_per_change

# # Percent Change for the years (2016 - 2017)
# jan_2016_2017 = percent_change(df, january_2016, january_2017)
# feb_2016_2017 = percent_change(df, february_2016, february_2017)
# mar_2016_2017 = percent_change(df, march_2016, march_2017)
# mar_2016_2017 = percent_change(df, march_2016, march_2017)

first_perchange_lst = []
second_perchange_lst = []

# For the months between the years 2016-2017
for old, new in zip(lst_2016, lst_2017):
    per_change = (new - old)/ old * 100
    rounded_per_change = float("{0:.2f}".format(per_change))
    first_perchange_lst.append(rounded_per_change)
    

per_2016_2017 = np.array(first_perchange_lst)
per_2016_2017[2] = 100

for old, new in zip(lst_2017, lst_2018):
    per_change = (new - old)/ old * 100
    rounded_per_change = float("{0:.2f}".format(per_change))
    second_perchange_lst.append(rounded_per_change)
    
per_2017_2018 = np.array(second_perchange_lst)
per_2017_2018 = np.concatenate([per_2017_2018, np.zeros(7)])
per_2017_2018[0] = 100

# Here we should create a line plot and something else for Percent change.
trace0 = go.Scatter(
    x = month_labels,
    y = per_2016_2017,
    name = 'Percent Change (2016-2017)',
    text = '%',
    line = dict(
        color = ('rgb(205, 12, 24)'),
        width = 4,
        dash = 'dash'
    )
)

trace1 = go.Scatter(
    x = month_labels,
    y = per_2017_2018,
    name = 'Percent Change (2017-2018)',
    text = '%',
    line = dict(
        color = ('rgb(0, 192, 98)'),
        width = 4,
        dash = 'dash'
    )
)

data = [trace0, trace1]

layout = dict(title = 'Percent Change in House Prices',
              xaxis = dict(title = 'Month'),
              yaxis = dict(title = 'Percent Change (%)'),
              paper_bgcolor='rgb(255, 255, 224)'
              )

fig = dict(data=data, layout=layout)
iplot(fig, filename='styled-line')

# Small Function for Sales Percentage
def region_sales_percentage(df, region, sales=total_sales):
    sales_percentage = (df['Price'].loc[df['Regionname'] == region].sum()/sales) * 100
    return sales_percentage

# Sales percentage per Region
northernmet_salesper = region_sales_percentage(df, region='Northern Metropolitan')
westernmet_salesper = region_sales_percentage(df, region='Western Metropolitan')
southernmet_salesper = region_sales_percentage(df, region='Southern Metropolitan')
easternmet_salesper = region_sales_percentage(df, region='Eastern Metropolitan')
south_easternmet_salesper = region_sales_percentage(df, region='South-Eastern Metropolitan')
northernvic_salesper = region_sales_percentage(df, region='Northern Victoria')
westernvic_salesper = region_sales_percentage(df, region='Western Victoria')
easternvic_salesper = region_sales_percentage(df, region='Eastern Victoria')

# Total Sales Sum per Region
nothernmet_total_sales = df['Price'].loc[df['Regionname'] == 'Northern Metropolitan'].sum()
westernmet_total_sales = df['Price'].loc[df['Regionname'] == 'Western Metropolitan'].sum()
southernmet_total_sales = df['Price'].loc[df['Regionname'] == 'Southern Metropolitan'].sum()
easternmet_total_sales = df['Price'].loc[df['Regionname'] == 'Eastern Metropolitan'].sum()
south_easternmet_total_sales = df['Price'].loc[df['Regionname'] == 'South-Eastern Metropolitan'].sum()
northernvic_total_sales = df['Price'].loc[df['Regionname'] == 'Northern Victoria'].sum()
westernvic_total_sales = df['Price'].loc[df['Regionname'] == 'Western Victoria'].sum()
easternvic_total_sales = df['Price'].loc[df['Regionname'] == 'Eastern Victoria'].sum()

labels = ['Northern <br> Metropolitan', 'Western <br> Metropolitan', 'Southern <br> Metropolitan', 'Eastern <br> Metropolitan', 
         'South-Eastern <br> Metropolitan', 'Northern <br> Victoria', 'Western <br> Victoria', 'Eastern <br> Victoria']

salesper_data = [northernmet_salesper, westernmet_salesper, southernmet_salesper, easternmet_salesper, 
              south_easternmet_salesper, northernvic_salesper, westernvic_salesper, easternvic_salesper]

total_sales_data = [nothernmet_total_sales, westernmet_total_sales, southernmet_total_sales, easternmet_total_sales, 
                   south_easternmet_total_sales, northernvic_total_sales, westernvic_total_sales, easternvic_total_sales]


sales_percent_plot = go.Bar(
    x=salesper_data,
    y=labels,
    marker=dict(
        color='rgba(152, 251, 152, 0.6)',
        line=dict(
            color='rgba(12, 218, 12, 1)',
            width=1),
    ),
    name='Sales Percentage of Houses per Region',
    orientation='h'
)

total_sales_plot = go.Scatter(
    x=total_sales_data,
    y=labels,
    mode='markers',
    marker=dict(
        color='rgb(34, 178, 178)',
    size=8),
    name='Total Sales per Region / (in Australian Dollars ($AUD))'
)

layout = go.Layout(
    title='Total Sales in Melbourne <br> Real Estate Market',
    yaxis1=dict(
        showgrid=True,
        showline=True,
        showticklabels=True,
        domain=[0, 0.85],
        tickangle=360,
        tickcolor=dict(
            color='rgb(255, 108, 108)'
        ),
    ),
    yaxis2=dict(
        showgrid=True,
        showline=True,
        showticklabels=False,
        linecolor='rgba(102, 102, 102, 0.8)',
        linewidth=2,
        domain=[0, 0.85],
        tickangle=360
    ),
    xaxis1=dict(
        zeroline=True,
        showline=True,
        showticklabels=False,
        showgrid=True,
        domain=[0, 0.42],
    ),
    xaxis2=dict(
        zeroline=True,
        showline=True,
        showticklabels=False,
        showgrid=True,
        domain=[0.47, 1],
        side='top',
        dtick=25000,
    ),
    legend=dict(
        x=0.029,
        y=1.038,
        font=dict(
            size=10,
        ),
    ),
    margin=dict(
        l=100,
        r=220,
        t=70,
        b=70,
    ),
    paper_bgcolor='rgb(255, 255, 224)',
    plot_bgcolor='rgb(255, 255, 246)',
)

y_s = np.round(salesper_data, decimals=2)
y_nw = np.rint(total_sales_data)

annotations = []

# Adding labels
for ydn, yd, xd in zip(y_nw, y_s, labels):
    # labeling the scatter savings
    annotations.append(dict(xref='x2', yref='y2',
                            y=xd, x=ydn*2,
                            text='{:,}'.format(ydn) + 'M',
                            ax=50,
                            ay=20,
                            xanchor='left',
                            yanchor='middle',
                            font=dict(family='Arial', size=12,
                                      color='rgb(17, 87, 87)'),
                            showarrow=False,
                           arrowhead=3))
    # labeling the bar net worth
    annotations.append(dict(xref='x1', yref='y1',
                            y=xd, x=yd + 6,
                            text=str(yd) + '%',
                            font=dict(family='Arial', size=12,
                                      color='rgb(7, 143, 7)'),
                            showarrow=False))
# Source
annotations.append(dict(xref='paper', yref='paper',
                        x=-0.1, y=-0.109,
                        text='Source: Plotly Documentation (Horizontal Bar Charts)',
                        font=dict(family='Arial', size=10,
                                  color='rgb(102, 102, 102)'),
                        showarrow=False))

layout['annotations'] = annotations

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, print_grid=False, specs=[[{}, {}]], shared_xaxes=True,
                          shared_yaxes=False, vertical_spacing=0.001)

fig.append_trace(sales_percent_plot, 1, 1)
fig.append_trace(total_sales_plot, 1, 2)

fig['layout'].update(layout)
iplot(fig, filename='oecd-networth-saving-bar-line')
x = 0
y = 0
width = 100
height = 100

region_names = df['Regionname'].value_counts().index
values = df['Regionname'].value_counts().tolist()

normed = squarify.normalize_sizes(values, width, height)
rects = squarify.squarify(normed, x, y, width, height)

colors = ['rgb(200, 255, 144)','rgb(135, 206, 235)',
          'rgb(235, 164, 135)','rgb(220, 208, 255)',
          'rgb(253, 253, 150)','rgb(255, 127, 80)', 
         'rgb(218, 156, 133)', 'rgb(245, 92, 76)']

shapes = []
annotations = []
counter = 0

for r in rects:
    shapes.append(
        dict(
            type = 'rect',
            x0 = r['x'],
            y0 = r['y'],
            x1 = r['x'] + r['dx'],
            y1 = r['y'] + r['dy'],
            line = dict(width=2),
            fillcolor = colors[counter]
        )
    )
    annotations.append(
        dict(
            x = r['x']+(r['dx']/2),
            y = r['y']+(r['dy']/2),
            text = values[counter],
            showarrow = False
        )
    )
    counter = counter + 1
    if counter >= len(colors):
        counter = 0
    
# For hover text
trace0 = go.Scatter(
    x = [ r['x']+(r['dx']/2) for r in rects],
    y = [ r['y']+(r['dy']/2) for r in rects],
    text = [ str(v) for v in region_names],
    mode='text',
)

layout = dict(
    title='Number of Houses Sold <br> <i>(Segmented by Region)</i>',
    height=700, 
    width=700,
    xaxis=dict(showgrid=False,zeroline=False),
    yaxis=dict(showgrid=False,zeroline=False),
    shapes=shapes,
    annotations=annotations,
    hovermode='closest'
)

# With hovertext
figure = dict(data=[trace0], layout=layout)

iplot(figure, filename='squarify-treemap')
# Average Distance per Region from the City Center
def distance_per_region(df, region):
    distances = df['Distance'].loc[df['Regionname'] == region].values.tolist()
    return distances

northern_met = distance_per_region(df, 'Northern Metropolitan')
southern_met = distance_per_region(df, 'Southern Metropolitan')
eastern_met = distance_per_region(df, 'Eastern Metropolitan')
western_met = distance_per_region(df, 'Western Metropolitan')
southeast_met = distance_per_region(df, 'South-Eastern Metropolitan')
northern_vic = distance_per_region(df, 'Northern Victoria')
eastern_vic = distance_per_region(df, 'Eastern Victoria')
western_vic = distance_per_region(df, 'Western Victoria')


# DataFrame to show the minimum and maximum distance from each region
regions = df['Regionname'].unique().tolist()

# Minimum for each region in distance
low_northmet = min(northern_met)
low_southmet = min(southern_met)
low_easternmet = min(eastern_met)
low_westmet = min(western_met)
low_southeastmet = min(southeast_met)
low_northvic = min(northern_vic)
low_eastvic = min(eastern_vic)
low_westvic = min(western_vic)

# Max Values
high_northmet = max(northern_met)
high_southmet = max(southern_met)
high_easternmet = max(eastern_met)
high_westmet = max(western_met)
high_southeastmet = max(southeast_met)
high_northvic = max(northern_vic)
high_eastvic = max(eastern_vic)
high_westvic = max(western_vic)

regions_data = {'Regions': regions, 'Minimum Distance (km)': [low_northmet, low_westmet, low_southmet, 
                                                        low_southeastmet, low_easternmet, low_northvic,
                                                        low_eastvic, low_westvic],
               'Maximum Distance (km)': [high_northmet, high_westmet, high_southmet, high_southeastmet,
                                   high_easternmet, high_northvic, high_eastvic, high_westvic]}

regions_distance = pd.DataFrame(data=regions_data)
regions_distance.style.bar(subset=['Minimum Distance (km)', 'Maximum Distance (km)'], color='#d65f5f')
northmet_trace = go.Area(
    r= northern_met,
    t=['Northern Metropolitan', 'Northern Victoria', 'Eastern Metropolitan', 
       'SouthEast Metropolitan', 'Southern Metropolitan', 'Western Victoria', 'West Metropolitan',
       'Eastern Victoria'],
    name='0-4 km',
    marker=dict(
        color='rgb(255, 36, 36)'
    )
)
northvic_trace = go.Area(
    r=northern_vic,
    t=['Northern Metropolitan', 'Northern Victoria', 'Eastern Metropolitan', 
       'SouthEast Metropolitan', 'Southern Metropolitan', 'Western Victoria', 'West Metropolitan',
       'Eastern Victoria'],
    name='5-9 km ',
    marker=dict(
        color='rgb(255, 144, 144)'
    )
)
easternmet_trace = go.Area(
    r=eastern_met,
    t=['Northern Metropolitan', 'Northern Victoria', 'Eastern Metropolitan', 
       'SouthEast Metropolitan', 'Southern Metropolitan', 'Western Victoria', 'West Metropolitan',
       'Eastern Victoria'],
    name='10-14 km',
    marker=dict(
        color='rgb(255, 200, 144)'
    )
)
southeast_trace = go.Area(
    r=southeast_met,
    t=['Northern Metropolitan', 'Northern Victoria', 'Eastern Metropolitan', 
       'SouthEast Metropolitan', 'Southern Metropolitan', 'Western Victoria', 'West Metropolitan',
       'Eastern Victoria'],
    name='15-19 km',
    marker=dict(
        color='rgb(255, 255, 144)'
    )
)

southernmet_trace = go.Area(
    r=southern_met,
    t=['Northern Metropolitan', 'Northern Victoria', 'Eastern Metropolitan', 
       'SouthEast Metropolitan', 'Southern Metropolitan', 'Western Victoria', 'West Metropolitan',
       'Eastern Victoria'],
    name='20-25 km',
    marker=dict(
        color='rgb(255, 255, 36)'
    )
)
westvic_trace = go.Area(
    r=western_vic,
    t=['Northern Metropolitan', 'Northern Victoria', 'Eastern Metropolitan', 
       'SouthEast Metropolitan', 'Southern Metropolitan', 'Western Victoria', 'West Metropolitan',
       'Eastern Victoria'],
    name='26-31 km',
    marker=dict(
        color='rgb(182, 255, 108)'
    )
)
westmet_trace = go.Area(
    r=western_met,
    t=['Northern Metropolitan', 'Northern Victoria', 'Eastern Metropolitan', 
       'SouthEast Metropolitan', 'Southern Metropolitan', 'Western Victoria', 'West Metropolitan',
       'Eastern Victoria'],
    name='32-42 km',
    marker=dict(
        color='rgb(108, 255, 108)'
    )
)
eastvic_trace = go.Area(
    r=eastern_vic,
    t=['Northern Metropolitan', 'Northern Victoria', 'Eastern Metropolitan', 
       'SouthEast Metropolitan', 'Southern Metropolitan', 'Western Victoria', 'West Metropolitan',
       'Eastern Victoria'],
    name='43-52 km',
    marker=dict(
        color='rgb(0, 219, 0)'
    )
)

data = [northmet_trace, northvic_trace, easternmet_trace, southeast_trace, southernmet_trace, westvic_trace, 
       westmet_trace, eastvic_trace]

layout = go.Layout(
    title='Distances from the City Center by Region',
    width=800,
    height=500,
    paper_bgcolor="rgb(255, 255, 224)",
    font=dict(
        size=16,
        color="#262626"
    ),
    legend=dict(
        font=dict(
            size=16
        )
    ),
    radialaxis=dict(
        ticksuffix='km'
    ),
    orientation=270
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='polar-area-chart')
# Add Periodic Table Data
symbol = [['Ba', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Ra'],
         ['Ne', 'Ho', '', '', '', '', '', '', '', '', '', '', 'Bux', 'Je', 'Mar', 'Ho', 'Bux', 'Ho'],
         ['Sw', 'Ja', '', '', '', '', '', '', '', '', '', '', 'Ne', 'Ra', 'Ga', 'Woo', 'Ba', 'Ob'],
         ['Br', 'Yp', 'Ho', 'Rai', 'Ba', 'Je', 'No', 'McG', 'Ha', 'Ne', 'Bu', 'St', 'Ba', 'Hod', 'Fl', 'Rt', 'Ha', 'Hod'],
         ['Gr ', 'Ra', 'Fn', 'Pr', 'Ra', 'Fl', 'Mi', 'Ho', 'Ph', 'Mo', 'Bux', 'Ca', 'Ra', 'Je', 'Gr', 'No', 'C', 'Is' ],
         ['Bi', 'Do', 'Ry', 'Yp', 'Ra', 'Fl', 'C', 'McG', 'M', 'L', 'iT', 'Bo', 'Ho', 'St', 'Yp', 'Bi', 'Ev', 'W' ],
         ['Vi', 'Wi', 'Ra', 'Re', 'Ba', 'Ob', 'Ha', 'Ho', 'Pe', 'Je', 'Ev', 'Hos', 'Ha', 'Lo', 'Br', 'Har', 'St', 'Hoh'],
         ['', '', 'Bi', 'Ba', 'Rai', 'Ra', 'Mo', 'Pr', 'Bu', 'Rt', 'Je', 'Ha', 'Bi', 'Rw', 'McG', 'Woo', 'Rai', ''],
         ['', '', 'Har', 'Le', 'Ba', 'Ho', 'Ma', 'Har', 'Fn', 'L', 'Mil', 'Rw', 'Da', 'Co', 'Bu', 'Ev', 'Mo', '' ],
         ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
         ['', 'West Met.', '', '', '', 'Eastern Met.', '', '', '', '', 'Northern Met.', '', '', '', '', 'Southern Met.', '', ''],
         ['', 'West Vic.', '', '', '', 'Eastern Vic.', '', '', '', '', 'Northern Vic.', '', '', '', '', 'South-East Met.', '', '']]

element = [['Barry', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Ray'],
           ['Nelson', 'hockingstuart', '', '', '', '', '', '', '', '', '', '', 'Buxton', 'Jellis', 'Marshall', 'hockingstuart', 'Buxton', 'hockingstuart'],
           ['Sweeney', 'Jas', '', '', '', '', '', '', '', '', '', '', 'Nelson', 'Ray', 'Gary', 'Woodards', 'Barry', "O'Brien"],
           ['Brad', ' YPA', ' hockingstuart', ' Raine', ' Barry', ' Jellis',  'Noel', 'McGrath', 'Harcourts', 'Nelson', 'Buckingham', 'Stockdale', 'Barry', 'Hodges', 'Fletchers', 'RT', 'Harcourts', 'Hodges'],
           ['Greg', 'Ray', 'FN', 'PRDNationwide', 'Ray', 'Fletchers', 'Miles', 'hockingstuart', 'Phillip', 'Morrison', 'Buxton', 'Carter', 'Ray', 'Jellis', 'Greg', 'Noel', 'C21', 'iSell'],
           ['Biggin', ' Douglas', 'Ryder',  'YPA', 'Ray', 'Fletchers', 'C21', 'McGrath', 'Max', 'LJ', 'iTRAK', 'Bowman', 'hockingstuart', 'Stockdale', 'YPA', 'Biggin', 'Eview', 'Win'],
           [' Village', ' Williams', 'Ray', 'Reliance','Barry',"O'Brien",'Harcourts','hockingstuart','Peake','Jellis','Eview','Hoskins','Harcourts','Love','Brad','HAR','Stockdale','hockingstuart/hockingstuart'],
           ['', '',  'Biggin', 'Barry', 'Raine', 'Ray', 'Morrison', 'PRDNationwide', 'Buckingham', 'RT', 'Jellis', 'Harcourts', 'Biggin', 'RW', 'McGrath', 'Woodards', 'Raine', ''],
           ['', '', 'HAR', 'Len', 'Barry', 'hockingstuart', 'Mason', 'HAR', 'FN', 'LJ', 'Millership', 'RW', 'Darren','Collins' ,'Buckingham', 'Eview', 'Morrison', '' ],
           ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
           ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
           ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']]

atomic_mass = [[ 1307, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0,  516],
     [ 1038, 589, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0,  1580, 1465, 1434, 1224, 511, 302],
     [ 510, 484, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0,  2014, 433, 396, 383, 280, 260], 
     [ 408, 403, 36, 31, 1106, 786, 477, 254, 233, 198, 108, 77, 1246, 376, 349, 262, 172, 172],
     [ 236, 236, 16, 16, 656, 574, 404, 250, 220, 121, 83, 71, 1156, 952, 260, 255, 140, 84],
     [ 226, 216, 14, 13, 48, 32, 23, 19, 13, 8, 7, 5, 592, 397, 374, 218, 63, 55],
     [ 197, 150, 8, 7, 32, 30, 21, 19, 9, 8, 6, 3, 365, 363, 356, 307, 54, 53],
     [.0, .0, 6, 4, 94, 33, 22, 17, 11, 8, 7, 5, 300, 243, 229, 227, 207, .0],
     [.0, .0, 4, 1, 50, 24, 22, 17, 9, 7, 6, 4, 177, 116, 166, 90, 77, .0],
     [.0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0],
     [.0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0],
     [.0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0]]

z = [[.15, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .85],
     [.15, .15, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .75, .75, .75, .75, .85, .85],
     [.15, .15, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .55, .75, .75, .75, .85, .85], 
     [.15, .15, .25, .25, .35, .35, .35, .35, .35, .35, .35, .35, .55, .75, .75, .75, .85, .85],
     [.15, .15, .25, .25, .35, .35, .35, .35, .35, .35, .35, .35, .55, .55, .75, .75, .85, .85],
     [.15, .15, .25, .25, .45, .45, .45, .45, .45, .45, .45, .45, .55, .55, .55, .75, .85, .85],
     [.15, .15, .25, .25, .45, .45, .45, .45, .45, .45, .45, .45, .55, .55, .55, .55, .85, .85],
     [.0, .0, .25, .25, .65, .65, .65, .65, .65, .65, .65, .65, .55, .55, .55, .55, .55, .0],
     [.0, .0, .25, .25, .65, .65, .65, .65, .65, .65, .65, .65, .55, .55, .55, .55, .55, .0],
     [.0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0],
     [.15, .15, .15, .0, .35, .35, .35, .0, .0, .55, .55, .55, .0, .0, .75, .75, .75, .0],
     [.25, .25, .25, .0, .45, .45, .45, .0, .0, .65, .65, .65, .0, .0, .85, .85, .85, .0]]

# Display element name and atomic mass on hover
hover=list(range(len(symbol)))
for x in range(len(symbol)):
    hover[x] = [i + '<br>' + 'Houses Sold: ' + str(j) for i, j in zip(element[x], atomic_mass[x])]

# Invert Matrices
symbol = symbol[::-1]
hover = hover[::-1]
z = z[::-1]

# Set Colorscale
colorscale=[[0.0, 'rgb(255,255,255)'], [.25, 'rgb(255, 255, 153)'], 
            [.45, 'rgb(153, 255, 204)'], [.65, 'rgb(0, 110, 219)'], 
            [.85, 'rgb(255, 96, 96)'],[1.0, 'rgb(255, 77, 148)']]

# Make Annotated Heatmap
pt = FF.create_annotated_heatmap(z, annotation_text=symbol, text=hover,
                                 colorscale=colorscale, font_colors=['black'], hoverinfo='text')
pt.layout.title = 'Best Seller Brokers <br> Distributed By Region'

iplot(pt, filename='periodic_table')
# Graphic Top Income Cities with a Population above 50 records

# data we wish to analize:
cols  = ['CouncilArea','Distance','Rooms','Price'];
col_analysis  = ['Distance','Rooms','Price']

# define titles for plot:
title = {
    'Distance':  'Distance Distribution <br><sub>Top 10 Council Areas</sub>',
     'Rooms':'# of Rooms Distribution <br><sub>Top 10 Council Areas</sub>',
    'Price':  'House Price Distribution <br><sub>Top 10 Council Areas</sub>',
}

# drop down names:
Keys = {'Distance':  'Distance from Downtown',
        'Rooms':'# of Rooms',
        'Price':'House Price'}

dropdown_names = {'Distance':  'Distance from Downtown',
                  'Rooms':'# of Rooms', 'Price':'House Price'}

colors = {'Distance':  'rgb(38, 38, 38)',
          'Rooms':'rgb(255, 99, 71)', 'Price':'rgb(102, 221, 170)'}

# drop down names:
drop_buttons = {
    'Distance':  [True, False, False],
    'Rooms':[False, True, False],
    'Price':[False, False, True],
}

# will be used for new plots
CA_count = df.CouncilArea.value_counts()
excluded_councils = CA_count[CA_count.values < 1820].index.tolist();


# group data & filter data
df_councils = df[cols].groupby(['CouncilArea']).mean().dropna()
df_councils = df_councils[~df_councils.index.isin(excluded_councils)]



buttons = []; data = []

for col in col_analysis:
    cas = df_councils.sort_values(col,ascending=0).index.tolist()[:10]
    data.append(go.Box(x=df[df.CouncilArea.isin(cas)]['CouncilArea'],
                       y=df[df.CouncilArea.isin(cas)][col],
                       visible= drop_buttons[col][0],
                       marker = dict(color = colors[col]),
                       name=dropdown_names[col],
                           showlegend=True))
    
    buttons.append(dict(label = dropdown_names[col],
                        method = 'restyle',
             args = ['visible', drop_buttons[col], 
                     'title', title[col]]))


updatemenus = list([dict(y=1.12, x= .98,buttons=buttons)])
layout = dict(title='Distributions by CouncilArea <br><sub>Select your type of distribution in drop down menu</sub>', 
              width=800,height=700, margin=go.Margin( l=50, r=25, b=100, t=100, pad=4),
              font=dict(family='Open Sans', size=12),
              showlegend=False,
              paper_bgcolor='rgb(255, 255, 255)',
              plot_bgcolor='rgb(255, 255, 255)',
              updatemenus=updatemenus)

fig = dict(data=data, layout=layout)
iplot(fig, filename='dropdown',config={'displayModeBar':False,'showLink': False,
                      'shape':{'layer':'below','hoverinfo':'none'}})
print('The main dataframe has: {} samples'.format(len(df)))
import cufflinks as cf

df_corr = df.corr()

trace = [go.Heatmap( z=df_corr.values.tolist(), 
                   x=df_corr.columns,
                   y=df_corr.columns,
                   colorscale='Pearl')]

layout = go.Layout(
    title="Heatmap <br> Detecting Correlations"
)

fig = dict(data=trace, layout=layout)


iplot(fig, filename='pandas-heatmap')
# Ill take a sample of the dataset since plotly is pretty small
# Our aim is to visualize plus we will randomly choose the data

shuffle_df = df.sample(frac=1).reset_index(drop=True)
sample_df = shuffle_df[:200]
sample_df.head()

colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98,0.98,0.98)]

distance = sample_df['Distance'].values
rooms = sample_df['Rooms'].values
price = sample_df['Price'].values


fig = FF.create_2d_density(
    distance, price, colorscale='YlOrRd',
    hist_color='rgb(255, 99, 71)', point_size=3
)

fig['layout'].update(showlegend=False, title="Price vs Distance <br> (Negative Correlation)",
                    height=500, width=800)

iplot(fig, filename='histogram_subplots')
fig2 = FF.create_2d_density(
    rooms, price, colorscale='YlOrRd',
    hist_color='rgb(255, 99, 71)', point_size=3
)

fig2['layout'].update(showlegend=False, title="Price vs Rooms <br> (Positive Correlation)",
                    height=500, width=800)

iplot(fig2, filename='histogram_subplots')