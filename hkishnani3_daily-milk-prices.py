# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
from plotly.offline import iplot, plot, init_notebook_mode
init_notebook_mode(connected=True)
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import random
from calendar import month_abbr

from statsmodels.tsa import stattools

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Packages required for data visualization and modelling
# Set your own project id here
PROJECT_ID = 'your-google-cloud-project'
from google.cloud import bigquery
bigquery_client = bigquery.Client(project=PROJECT_ID)
from google.cloud import storage
storage_client = storage.Client(project=PROJECT_ID)
df = pd.read_csv("/kaggle/input/milk-price/daily_milk_rate.csv")
df.info()
df.dropna(0, how='any', inplace = True)
print(df.info())

df['date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.drop(['Date'], axis=1, inplace=True)
df.set_index(['date'], drop=True, inplace=True)
df
# thus the data has 5 day a week resolution.
cities = df['Centre_Name'].value_counts()
print('\n',cities.keys)

commodities = df['Commodity_Name'].value_counts()
print('\n',commodities.keys)

# Sorting in descending fashion according to number of available data points.
# Since data points are of "daily or so" resolution thus It gives us a hint to what should be the resolution of plots to capture the pattern.
# since there are many cities So we decide to visualize data just for DELHI
# And later on compare with other cities later on in the code.
city = 'DELHI'
df.loc[df['Centre_Name']==city]

# why to carry Commodity_Name since right now it is Redundant column due to the fact that the dataset only comprices of Milk prices.
df.drop(['Commodity_Name'], axis=1, inplace=True)

# segregate data of a particular city Eg: Delhi
City_df = df.loc[df['Centre_Name']== city]

# and now we are dropping the Centre Name as well, since that too becomes a redundant column now.
# sO DROP IT LIKE ITS HOT 🪔
City_df.drop(['Centre_Name'], axis=1, inplace=True )
City_df = pd.Series(City_df['Price'])
print(City_df.head(15))
# the fashion in which date time is printed in the output is YYYY-MM-DD. Because of a certain reason.

years = City_df.index.year.value_counts().index
# This line is the James bond 🔫 of our code. Learning from mistakes.

City_df.index.year.value_counts()
# just to make sure we're on the right track!! This gives us a clue to the TS bias towards Year.
# Though it looks biased but let's move on.
# Line and scatter plot both 
fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x = City_df.index, y = City_df.values,
                         mode = 'markers',
                         name = 'markers'))

fig.add_trace(go.Scatter(x = City_df.index, y = City_df.values,
                         mode = 'lines+markers',
                         name = 'lines+markers'))

fig.add_trace(go.Scatter(x = City_df.index, y = City_df.values,
                         mode = 'lines',
                         name = 'lines'))

fig.update_layout(title = "Price v/s Date",
                  xaxis_title = "Date_Month_Year",
                  yaxis_title = "Price (₹)",
                  font = dict(
                          family = "Courier New, monospace",
                          size = 18,
                          color = "#7f7f7f"
                  )
                 )
fig.show()
# Color map in scatter plot to show gradient
fig = go.Figure(data = go.Scatter(x = City_df.index, y = City_df.values,
                                 mode = 'markers',
                                 marker = dict(
                                     size = 10,
                                     color = City_df.values,
                                     colorscale = 'Viridis',
                                 showscale = True
                                 )
                            )
               )

fig.update_layout(title = "Cmap Price v/s Date",
                  xaxis_title = "Date_Month_Year",
                  yaxis_title = "Prices (₹)",
                  font = dict(
                          family = "Courier New, monospace",
                          size = 18,
                          color = "#7f7f7f"
                  )
                 )

fig.show()
# Histograms

fig = go.Figure()
fig.add_trace(go.Histogram(histfunc = "avg", x = City_df.index,
                           y = City_df.values,
                          name = "avg"))

fig.update_layout(title = "Avg.Price v/s Date_span",
                  xaxis_title = "Date_Month_Year",
                  yaxis_title = "Prices (₹)",
                  font = dict(
                          family = "Courier New, monospace",
                          size = 18,
                          color = "#7f7f7f"
                  )
                 )
fig.show()


''' Here if is getting clumsy.
But on a side note Histograms do not give us a meaningful insight on yearly resolution and above all the fact that
milk prices are more about change than absolute values so to prove it we, need to get a distribution plot.'''
''' A box plot is a statistical representation of numerical data through their quartiles.
The ends of the box represent the lower and upper quartiles,
while the median (second quartile) is marked by a line inside the box.'''

fig = go.Figure()
fig.add_trace(go.Box(x = City_df.index.year, y = City_df.values,
             boxmean = 'sd',
             name = 'Mean & SD',
             marker_color = 'royalblue'))

fig.update_layout(
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
    yaxis=dict(zeroline=False, gridcolor='white'),
    paper_bgcolor='rgb(233,233,233)',
    plot_bgcolor='rgb(233,233,233)',
    title_text="Box Plot of Yearly Prices"
)

fig.show()

# Now you could see why. Why can't yearly resolution give us a meaningful insight of the data just have a look at the value of sigma (SD)


City_df_after_2010 = City_df.loc[City_df.index.year >= 2010]
City_df_after_2010.head(10)

fig = go.Figure()
fig.add_trace(go.Box(x = City_df_after_2010.index.year, y = City_df_after_2010.values,
             boxmean = 'sd',
             name = 'Mean & SD',
             marker_color = 'royalblue'))

fig.update_layout(
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
    yaxis=dict(zeroline=False, gridcolor='white'),
    paper_bgcolor='rgb(233,233,233)',
    plot_bgcolor='rgb(233,233,233)',
    title_text="Box Plot of Yearly Prices"
)

fig.show()


# Now we'll visualize data in a different sense. In terms of a batch of days or maybe weeks.
# But before that we Could have an exploded view of the boxplot of trimmed dataset.
# Box for 2015 is flat due to the fact that it has only 78 points and all of them have a single value that is ₹38/lit.

# Random set visualization on daily resolution keep N <= 60

N =[10,20,30,40,50,60]
for N_points in N:
    random_date_time = random.randint(0,len(City_df_after_2010.index)-N_points)
    input(f"\n\nShowing graph of {N_points} points")
    print(f"randomly chosen date and time is {City_df_after_2010.index[random_date_time]}")

    # actually what you could see as Hour in th output is something that pops out by default.
    # this does not imply that the point corresponds to 12 in the midnight since the data has day's resolution so we do not have to worry about the timestamp

    random_day_set = []
    for day in range(N_points):
      random_day_set.append(City_df_after_2010.index[random_date_time + day])

    # Which plot could give us a deep and accurate insight to the pattern if it exists?
    # answer to this question is quite debateable but let us go with line plots.
    # let us give it a hit. For  N = 10,20,30,40,50 and 60 if we do not find any meaningful insight then it would imply that pattern does not have a daily resolution.
    # Once we detect a pattern then we could go about quantifying it using Auto-correlation function.
    # But the essential question remains the same and that is when does prices start deviating from their lagging values and in what pattern if any.
    
    #print(random_day_set,'\n')
    #City_df_after_2010.loc[random_day_set]
    #print (City_df_after_2010,'\n')
    fig = go.Figure()
    # Add Traces
    fig.add_trace(go.Scatter(x=City_df_after_2010.loc[random_day_set].index, y=City_df_after_2010.loc[random_day_set].values,
                             mode='lines+markers',
                             name='lines+markers'))
    fig.show()

# Click on output screen to enlarge. and this is what has been happening most of the times that with N=60 prices fluctuate. 
# With this what we could conclude is that the data is not varying over a resolution of days. So let us check once again for resolution of months
N =[3,4,5,6] # in months roughly around 20 entries each month
for N_points in N:
    random_date_time = random.randint(0,len(City_df_after_2010.index)-N_points*20)
    input(f"\n\nShowing graph of {N_points} points")
    print(f"randomly chosen date and time is {City_df_after_2010.index[random_date_time]}")

    # actually what you could see as Hour in th output is something that pops out by default.
    # this does not imply that the point corresponds to 12 in the midnight since the data has day's resolution so we do not have to worry about the timestamp

    random_day_set = []
    for day in range(N_points*20):
      random_day_set.append(City_df_after_2010.index[random_date_time + day])

    # Which plot could give us a deep and accurate insight to the pattern if it exists?
    # answer to this question is quite debateable but let us go with line plots.
    # let us give it a hit. For  N = 10,20,30,40,50 and 60 if we do not find any meaningful insight then it would imply that pattern does not have a daily resolution.
    # Once we detect a pattern then we could go about quantifying it using Auto-correlation function.
    # But the essential question remains the same and that is when does prices start deviating from their lagging values and in what pattern if any.
    
    #print(random_day_set,'\n')
    #City_df_after_2010.loc[random_day_set]
    #print (City_df_after_2010,'\n')
    fig = go.Figure()
    # Add Traces
    fig.add_trace(go.Scatter(x=City_df_after_2010.loc[random_day_set].index, y=City_df_after_2010.loc[random_day_set].values,
                             mode='lines+markers',
                             name='lines+markers'))
    fig.show()
    

# With this what we could conclude is that the data is varying over a resolution of months not days. So let us check once again
# Thus now with logical reasoning and quantitative plots we conclude that having a violon plot of monthly resolution will serve the purpose.
# Half the work is done.
list_years = [City_df_after_2010.index.year.values]
list_years = np.unique(np.array(list_years))

list_month = [City_df_after_2010.index.month.values]
list_month = np.unique(np.array(list_month))



for year in list_years:
    fig = go.Figure()
    for month in list_month:
        y = City_df_after_2010.loc[City_df_after_2010.index.month == month]
        y = y.loc[y.index.year == year]
        print(y)
        x = month_abbr[month]
        print(x)
        type(y)

# pERRRRR.... fectO
        