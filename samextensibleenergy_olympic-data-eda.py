import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime as dt
from sklearn.metrics import mean_squared_error
from math import sqrt

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool

athlete = pd.read_csv('../input/athlete_events.csv')
print ("Athlete Dataset: Rows, Columns: ", athlete.shape)

a_df = athlete # creating a separate dataset to operate on!!
a_df = a_df.dropna(subset=['Age', 'Height', 'Weight', 'Year'])
a_df.describe() # cleaning, dropping those rows with no values in any one of these columns
year = pd.DataFrame(a_df.groupby(['Year', 'Season']).count()).reset_index()
year = year[['Year','ID', 'Season']]
years = year[year.Season == 'Summer']
yearw = year[year.Season == 'Winter']

output_file('Count_bars.html')

sourceS = ColumnDataSource(
        data=dict(
            xs= years.Year,
            ys= years.ID,
        )
    )

sourceW = ColumnDataSource(
        data=dict(
            xw= yearw.Year,
            yw= yearw.ID,
        )
    )

hoverS = HoverTool(
        tooltips=[
            ("Year:", "@xs"), # use @ for fixed values and $ for continuous hovering values
            ("Athlete_Counts:", "@ys")
        ]
    )


hoverW = HoverTool(
        tooltips=[
            ("Year:", "@xw"), # use @ for fixed values and $ for continuous hovering values
            ("Athlete_Counts:", "@yw")
        ]
    )

p = figure(plot_width=1000, plot_height=500, tools=[hoverW, hoverS],
           title="Total Participation versus Year")


# p.line('x', 'y', legend="Counts", line_color="red", source=source)
p.circle('xs', 'ys', legend="Summer", size=8, fill_color="orange", line_color="green", 
         line_width=3, source=sourceS)
p.vbar(x='xs', width=0.1, bottom=0, top = 'ys', color="firebrick", source = sourceS)

p.circle('xw', 'yw', legend="Winter", size=8, fill_color="magenta", line_color="purple", 
         line_width=3, source=sourceW)
p.vbar(x='xw', width=0.1, bottom=0, top = 'yw', color="indigo", source = sourceW)

show(p)
a_df['BMI'] = a_df['Weight'] /  ((a_df['Height']/100)**2) ## Creating A BMI Column For Normalizing Weight and Heights for Athletes
# plot
sns.set_style('ticks')
fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(20, 8.27)

bplot = sns.boxplot(y='BMI', x='Year', 
                 data=a_df, 
                 width=0.5,
                 palette="colorblind")

# bplot = sns.violinplot(y='BMI', x='Year', data=a_df, inner="points", ax=ax)    
# sns.despine()

bplot.axes.set_title("BMI v/s Olympic Years",
                    fontsize=16)
 
bplot.set_xlabel("Years", 
                fontsize=14)
 
bplot.set_ylabel("Body Mass Index (BMI)",
                fontsize=14)
 
bplot.tick_params(labelsize=10)
bmi = pd.DataFrame(a_df.groupby(['Season','Year']).mean()).reset_index()
# bmi = bmi[bmi.Season == 'Summer']
# bmi.tail(50)
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool


bmis = bmi[bmi.Season == 'Summer']
bmiw = bmi[bmi.Season == 'Winter']

output_file('bmi_avg.html')

sourceS = ColumnDataSource(
        data=dict(
           
            xs= bmis.Year,
            ys= bmis.BMI,
            
        )
    )

sourceW = ColumnDataSource(
        data=dict(
           
            xw= bmiw.Year,
            yw= bmiw.BMI,
            
        )
    )

hoverS = HoverTool(
        tooltips=[
            ("Year:", "@xs" ), # use @ for fixed values and $ for continuous hovering values
            ("BMI:", "@ys")
        ]
    )

hoverW = HoverTool(
        tooltips=[
            ("Year:", "@xw" ), # use @ for fixed values and $ for continuous hovering values
            ("BMI:", "@yw")
        ]
    )

p = figure(plot_width=1000, plot_height=500, tools=[hoverW, hoverS],
           title="Average BMI versus Year")


p.line('xs', 'ys', legend="Summer", line_color="lightcoral", source=sourceS)
p.line('xw', 'yw', legend="Winter", line_color="lightseagreen", source=sourceW)
p.circle('xs', 'ys', legend="Summer", size=5, fill_color="red", line_color="darkred", 
         line_width=2, source=sourceS)
p.circle('xw', 'yw', legend="Winter", size=5, fill_color="teal", line_color="midnightblue", 
          line_width=2, source=sourceW)
# p.vbar(x='x', width=0.1, bottom=0, top = 'y', color="firebrick", source = source)

show(p)
mf = pd.DataFrame(a_df.groupby(['Season','Sex']).count()).reset_index()
mf.tail()
plt.figure(0)
labels = 'Summer_Male', 'Summer_Female'
sizes = [113325, 53381]
colors = ['yellowgreen', 'lightcoral']
explode = (0, 0.1)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode,  labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')


plt.figure(1)
labels = 'Winter_Male', 'Winter_Female'
sizes = [26129, 13330]
colors = ['yellowgreen', 'lightcoral']
explode = (0, 0.1)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode,  labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')

plt.show() #show all figures
medal = pd.DataFrame(a_df.groupby(['Year', 'Medal']).count()).reset_index()
medal.tail(50)
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
from bokeh.transform import factor_cmap

output_file("bar_nested_medal.html")

years = ['1896', '1900', '1904', '1906', '1908', '1912', '1920', '1924', '1928', '1932', '1936', '1948',
          '1952', '1956', '1960', '1964', '1968', '1972', '1976', '1980', '1984', '1988', '1992',
          '1994', '1996', '1998', '2000', '2002', '2004', '2006', '2008', '2010', '2012', '2014', '2016']

medals = ['Gold', 'Silver', 'Bronze']

data = {'years' : years,
        'Gold'   : [2, 1, 4, 3, 2, 4, 2, 1, 4, 3, 2, 4,2, 1, 4, 3, 2, 4,2, 1, 4, 3, 2, 4,2, 1, 4, 3, 2, 4,2, 1, 4, 3, 2],#medal.ID,
        'Silver'   : [2, 1, 4, 3, 2, 4, 2, 1, 4, 3, 2, 4,2, 1, 4, 3, 2, 4,2, 1, 4, 3, 2, 4,2, 1, 4, 3, 2, 4,2, 1, 4, 3, 2],#medal.ID,
        'Bronze'   : [2, 1, 4, 3, 2, 4, 2, 1, 4, 3, 2, 4,2, 1, 4, 3, 2, 4,2, 1, 4, 3, 2, 4,2, 1, 4, 3, 2, 4,2, 1, 4, 3, 2]}#medal.ID}


palette = ["#c9d9d3", "#718dbf", "#e84d60"]

# this creates [ ("Apples", "2015"), ("Apples", "2016"), ("Apples", "2017"), ("Pears", "2015), ... ]
x = [ (year, medal) for year in years for medal in medals ]
counts = sum(zip(data['Gold'], data['Silver'], data['Bronze']), ()) # like an hstack

source = ColumnDataSource(data=dict(x=x, counts=counts))

p = figure(x_range=FactorRange(*x), plot_height=350, title="Fruit Counts by Year",
           toolbar_location=None, tools="")

p.vbar(x='x', top='counts', width=0.9, source=source, line_color="yellow",
       fill_color=factor_cmap('x', palette=palette, factors=medals))

# p.y_range.start = 0
# p.x_range.range_padding = 0.1
# p.xaxis.major_label_orientation = 1
# p.xgrid.grid_line_color = None

show(p)


from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls

cnt_srs = a_df['Sport'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Picnic',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Most Popular Sport'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="sport")


medal = pd.DataFrame(a_df.groupby(['Team']).count()).reset_index()
medal = pd.DataFrame(medal.sort_values(by='Medal', ascending=False)).reset_index()
medal = medal[['Team', 'ID', 'Medal']]
medal.head(20)
# adding the data for West and East Germany to combine into a single country Germany
germany = medal.loc[medal['Team'].isin(['Germany','East Germany','West Germany' ])].sum()
russia = medal.loc[medal['Team'].isin(['Soviet Union','Russia'])].sum()

print (germany, russia)
addition = [{'Team': 'Germany', 'ID': 12667, 'Medal': 3007},
            {'Team': 'Russia', 'ID': 9371, 'Medal': 3294}]
df = pd.DataFrame(addition)
medal_new = medal.append(df) # adding new  combined data for 2 countries
medal_new_ = medal_new.sort_values(by='Medal', ascending=False)
medal_new_ = medal_new_.drop([3, 4 ,6, 11, 19]) # dropping common Columns
medal_new_.head(20)
