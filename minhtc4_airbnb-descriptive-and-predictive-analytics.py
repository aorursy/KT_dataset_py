import numpy as np

import pandas as pd
calendar = pd.read_csv("../input/calendar.csv")

calendar.head()
calendar.info()
listing = pd.read_csv("../input/listings.csv")

listing.head(1)
reviews = pd.read_csv("../input/reviews.csv")

reviews.head()
df_listing = listing[listing.applymap(np.isreal)]

df_listing.dropna(how = "all", axis = 1, inplace = True)

df_listing.head()
df_listing.fillna(listing.mean(), inplace = True)

df_listing.drop(["latitude", "longitude"], axis = 1, inplace= True)

df_listing.head()
calendar.head()
calendar["price"] = calendar["price"].apply(lambda x: str(x).replace("$", ""))

calendar["price"] = pd.to_numeric(calendar["price"] , errors="coerce")

df1  = calendar.groupby("date")[["price"]].sum()

df1["mean"]  = calendar.groupby("date")[["price"]].mean()

df1.columns = ["Total", "Average"]

df1.head()
df2 = calendar.set_index("date")

df2.index = pd.to_datetime(df2.index)

df2 =  df2[["price"]].resample("M").mean()

df2.head()
import plotly as py

from plotly.offline import iplot, plot, init_notebook_mode, download_plotlyjs

import plotly.graph_objs as go

init_notebook_mode(connected=True)

import plotly.offline as offline
trace1 = go.Scatter(

    x = df1.index,

    y = df1["Total"]

)

data = [trace1]

layout = go.Layout(

    title = "Price by each time",

    xaxis  = dict(title = "Time"),

    yaxis = dict(title = "Total ($)")

)

trace2 = go.Scatter(

    x = df1.index,

    y = df1["Average"]

)



data2 = [trace2]

layout2 = go.Layout(

    title = "Price by each time",

    xaxis  = dict(title = "Time"),

    yaxis = dict(title = "Mean ($)")

)

fig = go.Figure(data = data, layout = layout)

fig2 = go.Figure(data = data2, layout = layout2)

offline.iplot(fig)
offline.iplot(fig2)
trace3 = go.Scatter(

    x = df2.index[:-1],

    y = df2.price[:-1]

)

layout3 = go.Layout(

    title = "Average price by month",

    xaxis = dict(title = "time"),

    yaxis = dict(title = "Price")

)

data3 = [trace3]

fig3 = go.Figure(data= data3, layout= layout3)

offline.iplot(fig3)
from statsmodels.tsa.seasonal import seasonal_decompose
def draw_interactive_graph(mode):

    df1.index = pd.to_datetime(df1.index)

    decomposition = seasonal_decompose(df1[[mode]])

    trace4_1 = go.Scatter(

        x = decomposition.observed.index, 

        y = decomposition.observed[mode],

        name = "Observed"

    )

    trace4_2 = go.Scatter(

        x = decomposition.trend.index,

        y = decomposition.trend[mode],

        name = "Trend"

    )

    trace4_3 = go.Scatter(

        x = decomposition.seasonal.index,

        y = decomposition.seasonal[mode],

        name = "Seasonal"

    )

    trace4_4 = go.Scatter(

        x = decomposition.resid.index,

        y = decomposition.resid[mode],

        name = "Resid"

    )



    fig = py.tools.make_subplots(rows=4, cols=1, subplot_titles=('Observed', 'Trend',

                                                              'Seasonal', 'Residiual'))

    # append trace into fig

    fig.append_trace(trace4_1, 1, 1)

    fig.append_trace(trace4_2, 2, 1)

    fig.append_trace(trace4_3, 3, 1)

    fig.append_trace(trace4_4, 4, 1)



    fig['layout'].update( title='Descompose with TimeSeri')

    offline.iplot(fig)
draw_interactive_graph("Average")
draw_interactive_graph("Total")
def loc_city(x):

    if "," not in str(x):

        return x

    if "live" in str(x) or "Next door to" in str(x) or "live" in str(x) or "having" in str(x):

        return "USA"

    return str(x).split(",")[0]

a = listing["host_location"].apply(lambda x: loc_city(x))
df_listing["City"]  = a

df_listing.head(1)
df_seattle = df_listing[df_listing["City"] == "Seattle"]

df_seattle.head()
calendar_clean = calendar.dropna()

calendar_clean.set_index("date", inplace = True)

calendar_clean.head()
calendar_clean.index = pd.to_datetime(calendar_clean.index)

number_hire_room = calendar_clean.resample("M")[["price"]].count()

total_price_each_month  = calendar_clean.resample("M")[["price"]].sum()
trace5 = go.Scatter(

    x = number_hire_room.index[:-1],

    y = number_hire_room.price[:-1]

)

data5 = [trace5]

layout5 = go.Layout(

    title = "Number of Hire Room by Month in Seattle",

    xaxis = dict(title = "Month"),

    yaxis = dict(title = "Number hirde")

)

fig5  = go.Figure(data = data5, layout = layout5)
trace6 = go.Scatter(

    x = number_hire_room.index[:-1],

    y = number_hire_room.price[:-1]/number_hire_room.price[0]

)

data6 = [trace6]

layout6 = go.Layout(

    title = "the ratio of the number of rooms compare with the first month",

    xaxis = dict(title = "Month"),

    yaxis = dict(title = "Ratio")

)

fig6 = go.Figure(data = data6, layout = layout6)
offline.iplot(fig5)
offline.iplot(fig6)
from scipy import stats
a = calendar_clean.index.month

# calendar_clean["Month"] = a

calendar_clean = calendar_clean.assign(Month = a)

calendar_clean.head()
result = []

for i in range(1,13):

    result.append(np.array([calendar_clean[calendar_clean["Month"] == i].price]))
data_score = []

for i in range(11):

    score = stats.ttest_rel(result[i][0][:64911],result[-1][0][:64911])

    data_score.append((score[0], score[1]))
score_board = pd.DataFrame(data = data_score, columns = ["Test Statistic", "P_value"])

score_board["Month"] = range(1, 12)

score_board.set_index("Month", inplace = True)

score_board
offline.iplot(fig3)