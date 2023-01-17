import numpy as pd, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import warnings, time, gc
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)

color = sns.color_palette()
warnings.filterwarnings("ignore")
%matplotlib inline

from kaggle.competitions import twosigmanews

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

env = twosigmanews.make_env()
market_train, news_train = env.get_training_data()
import random

def generate_color():
    color = "#{:02x}{:02x}{:02x}".format(*map(lambda x: random.randint(0, 255), range(3)))
    return color
market_train.head()
market_train.describe()
market_train.isnull().sum()
market_train["time"] = market_train["time"].dt.strftime(date_format = '%Y-%m-%d')
temp = market_train["time"].value_counts()
time_count = pd.DataFrame({"date": temp.index,
                           "count": temp.values}).sort_values(by = "date")
time_count.sort_values(by = "date")
trace = go.Scatter(x = time_count["date"].values,
                   y = time_count["count"].values)

layout = dict(title = "Asset Counts Per Day", 
              xaxis = dict(title = "Day"), 
              yaxis = dict(title = "Count"))
iplot(dict(data = [trace], layout = layout))
asset_code = market_train["assetCode"].describe()
asset_name = market_train["assetName"].describe()

pd.concat([asset_code, asset_name], axis = 1, keys = ["Asset Code", "Asset Name"])
unknown_name = market_train[market_train["assetName"] == "Unknown"]
unknown_count = unknown_name["assetCode"].value_counts()

print("There are {} unique asset code with unknown asset name".format(len(unknown_count.index)))
trace = go.Bar(x = unknown_count.index[:25],
               y = unknown_count.values[:25])
layout = dict(title = "First 25 Asset Code with Unknown Asset Name", 
              xaxis = dict(title = "Asset Code"), 
              yaxis = dict(title = "Days"))
iplot(dict(data = [trace], layout = layout))
valumeByCode = market_train.groupby(market_train["assetCode"])["volume"].sum().sort_values(ascending = False)
top_trade_code = valumeByCode.index[:10]

fig = tools.make_subplots(rows = len(top_trade_code), cols = 2)

for i, c in enumerate(top_trade_code):
    temp = market_train[market_train["assetCode"] == c]
    trace = go.Scatter(x = temp["time"],
                       y = temp["volume"],
                       name = c)
    fig.append_trace(trace, int(i/2)+1, (i%2)+1)

fig["layout"].update(height = 1000, width = 800,
                     title = "Top 10 AssetCode By Trade Volume")
    
iplot(fig)
fig = tools.make_subplots(rows = len(top_trade_code), cols = 2)

for i, c in enumerate(top_trade_code):
    temp = market_train[(market_train["assetCode"] == c) & 
                        (market_train["time"] > "2008-01-01") &
                        (market_train["time"] < "2010-01-01")]
    trace = go.Scatter(x = temp["time"],
                       y = temp["volume"],
                       name = c)
    fig.append_trace(trace, int(i/2)+1, (i%2)+1)

fig["layout"].update(height = 1200, width = 800,
                     title = "Top 10 AssetCode By Trade Volume")
    
iplot(fig)
def candle_sticks(data):
    data["high"] = data["open"]
    data["low"] = data["close"]
    
    for idx, row in data.iterrows():
        if row["close"] > row["open"]:
            data.loc[idx, "high"] = row["close"]
            data.loc[idx, "low"] = row["open"]
            
    return data

for c in top_trade_code[:5]:
    temp1 = market_train[market_train["assetCode"] == c][["time", "open", "close"]]
    temp2 = candle_sticks(temp1)
    
    trace = go.Candlestick(x = temp2["time"],
                           open = temp2["open"],
                           low = temp2["low"],
                           high = temp2["high"],
                           close = temp2["close"],
                           increasing = dict(line = dict(color = generate_color())),
                           decreasing = dict(line = dict(color = generate_color())))
    
    layout = dict(title = "Candlestick Chart for {}".format(c), 
                 xaxis = dict(title = "Day"),
                 yaxis = dict(title = "Price (USD)"))
    
    iplot(dict(data = [trace], layout = layout))
c = top_trade_code[0]
temp1 = market_train[market_train["assetCode"] == c]
color = generate_color()
trace1 = go.Scatter(x = temp1["time"],
                    y = temp1["open"],
                    marker = dict(color = color),
                    name = c)

trace2 = go.Scatter(x = temp1["time"],
                    y = temp1["returnsOpenPrevRaw1"],
                    marker = dict(color = color),
                    name = c)

trace3 = go.Scatter(x = temp1["time"],
                    y = temp1["returnsOpenPrevRaw10"],
                    marker = dict(color = color),
                    name = c)

title1 = "Price Per Day"
title2 = "Price Differences (1 Day)"
title3 = "Price Differences (10 Days)"
temp2 = market_train[(market_train["assetCode"] == c) & 
                     (market_train["time"] > "2006-12-31") & 
                     (market_train["time"] < "2008-01-01")]
color = generate_color()
trace4 = go.Scatter(x = temp2["time"],
                    y = temp2["open"],
                    marker = dict(color = color),
                    name = c)

trace5 = go.Scatter(x = temp2["time"],
                    y = temp2["returnsOpenPrevRaw1"],
                    marker = dict(color = color),
                    name = c)
    
trace6 = go.Scatter(x = temp2["time"],
                    y = temp2["returnsOpenPrevRaw10"],
                    marker = dict(color = color),
                    name = c)

title4 = "Price Per Day"
title5 = "Price Differences (1 Day)"
title6 = "Price Differences (10 Days)"
temp3 = market_train[(market_train["assetCode"] == c) & 
                     (market_train["time"] > "2007-10-31") & 
                     (market_train["time"] < "2008-01-01")]
color = generate_color()
trace7 = go.Scatter(x = temp3["time"],
                    y = temp3["open"],
                    marker = dict(color = color),
                    name = c)

trace8 = go.Scatter(x = temp3["time"],
                    y = temp3["returnsOpenPrevRaw1"],
                    marker = dict(color = color),
                    name = c)

trace9 = go.Scatter(x = temp3["time"],
                    y = temp3["returnsOpenPrevRaw10"],
                    marker = dict(color = color),
                    name = c)

title7 = "Price Per Day"
title8 = "Price Differences (1 Day)"
title9 = "Price Differences (10 Days)"
fig = tools.make_subplots(rows = 3, cols = 3, subplot_titles = (title1, title2, title3,
                                                                title4, title5, title6,
                                                                title7, title8, title9))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)
fig.append_trace(trace4, 2, 1)
fig.append_trace(trace5, 2, 2)
fig.append_trace(trace6, 2, 3)
fig.append_trace(trace7, 3, 1)
fig.append_trace(trace8, 3, 2)
fig.append_trace(trace9, 3, 3)

fig["layout"]["xaxis7"].update(title = "Day")
fig["layout"]["xaxis8"].update(title = "Day")
fig["layout"]["xaxis9"].update(title = "Day")

fig["layout"].update(height = 1000, width = 800,
                     title = "{} Open Price & Its Difference Overview".format(c))
iplot(fig)
color = generate_color()
trace1 = go.Scatter(x = temp1["time"],
                    y = temp1["close"],
                    marker = dict(color = color),
                    name = c)

trace2 = go.Scatter(x = temp1["time"],
                    y = temp1["returnsClosePrevRaw1"],
                    marker = dict(color = color),
                    name = c)

trace3 = go.Scatter(x = temp1["time"],
                    y = temp1["returnsClosePrevRaw10"],
                    marker = dict(color = color),
                    name = c)

title1 = "Price Per Day"
title2 = "Price Differences (1 Day)"
title3 = "Price Differences (10 Days)"
temp2 = market_train[(market_train["assetCode"] == c) & 
                     (market_train["time"] > "2006-12-31") & 
                     (market_train["time"] < "2008-01-01")]
color = generate_color()
trace4 = go.Scatter(x = temp2["time"],
                    y = temp2["close"],
                    marker = dict(color = color),
                    name = c)

trace5 = go.Scatter(x = temp2["time"],
                    y = temp2["returnsClosePrevRaw1"],
                    marker = dict(color = color),
                    name = c)

trace6 = go.Scatter(x = temp2["time"],
                    y = temp2["returnsClosePrevRaw10"],
                    marker = dict(color = color),
                    name = c)

title4 = "Price Per Day"
title5 = "Price Differences (1 Day)"
title6 = "Price Differences (10 Days)"
temp3 = market_train[(market_train["assetCode"] == c) & 
                     (market_train["time"] > "2007-10-31") & 
                     (market_train["time"] < "2008-01-01")]
color = generate_color()
trace7 = go.Scatter(x = temp3["time"],
                    y = temp3["close"],
                    marker = dict(color = color),
                    name = c)

trace8 = go.Scatter(x = temp3["time"],
                    y = temp3["returnsClosePrevRaw1"],
                    marker = dict(color = color),
                    name = c)

trace9 = go.Scatter(x = temp3["time"],
                    y = temp3["returnsClosePrevRaw10"],
                    marker = dict(color = color),
                    name = c)
    
title7 = "Price Per Day"
title8 = "Price Differences (1 Day)"
title9 = "Price Differences (10 Days)"
fig = tools.make_subplots(rows = 3, cols = 3, subplot_titles = (title1, title2, title3,
                                                                title4, title5, title6,
                                                                title7, title8, title9))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)
fig.append_trace(trace4, 2, 1)
fig.append_trace(trace5, 2, 2)
fig.append_trace(trace6, 2, 3)
fig.append_trace(trace7, 3, 1)
fig.append_trace(trace8, 3, 2)
fig.append_trace(trace9, 3, 3)

fig["layout"]["xaxis7"].update(title = "Day")
fig["layout"]["xaxis8"].update(title = "Day")
fig["layout"]["xaxis9"].update(title = "Day")

fig["layout"].update(height = 1000, width = 800,
                     title = "{} Close Price & Its Difference Overview".format(c))
iplot(fig)
fig = tools.make_subplots(rows = len(top_trade_code), cols = 2)

for i, c in enumerate(top_trade_code):
    temp = market_train[market_train["assetCode"] == c][["time", "returnsOpenNextMktres10"]]
    trace = go.Scatter(x = temp["time"],
                       y = temp["returnsOpenNextMktres10"],
                       marker = dict(color = generate_color()),
                       name = c)
    fig.append_trace(trace, int(i//2)+1, (i%2)+1)

fig["layout"].update(height = 800, width = 800,
                     title = "Returns Open Next Mktres Top 10 Trade Value Companies (10 Days)")
iplot(fig)
fig = tools.make_subplots(rows = len(top_trade_code), cols = 2)

for i, c in enumerate(top_trade_code):
    temp = market_train[(market_train["assetCode"] == c) & 
                        (market_train["time"] > "2007-12-31") & 
                        (market_train["time"] < "2010-01-01")][["time", "returnsOpenNextMktres10"]]
    trace = go.Scatter(x = temp["time"],
                       y = temp["returnsOpenNextMktres10"],
                       marker = dict(color = generate_color()),
                       name = c)
    fig.append_trace(trace, int(i//2)+1, (i%2)+1)

fig["layout"].update(height = 800, width = 800,
                     title = "Returns Open Next Mktres Top 10 Trade Value Companies (10 Days)")
iplot(fig)
fig = tools.make_subplots(rows = len(top_trade_code), cols = 2)

for i, c in enumerate(top_trade_code):
    temp = market_train[(market_train["assetCode"] == c) & 
                        (market_train["time"] > "2008-12-31") & 
                        (market_train["time"] < "2010-01-01")][["time", "returnsOpenNextMktres10"]]
    trace = go.Scatter(x = temp["time"],
                       y = temp["returnsOpenNextMktres10"],
                       marker = dict(color = generate_color()),
                       name = c)
    fig.append_trace(trace, int(i//2)+1, (i%2)+1)

fig["layout"].update(height = 800, width = 800,
                     title = "Returns Open Next Mktres Top 10 Trade Value Companies (10 Days)")
iplot(fig)
known_open_Mktres = market_train[market_train["returnsOpenPrevMktres10"].notnull()]

fig = tools.make_subplots(rows = len(top_trade_code), cols = 2)
for i, c in enumerate(top_trade_code):
    temp = known_open_Mktres[known_open_Mktres["assetCode"] == c]
    color = generate_color()
    trace1 = go.Scatter(x = temp["time"],
                        y = temp["returnsOpenPrevMktres10"],
                        marker = dict(color = color),
                        name = c)
    fig.append_trace(trace1, i+1, 1)
    trace2 = go.Scatter(x = temp["time"],
                        y = temp["returnsOpenNextMktres10"],
                        marker = dict(color = color),
                        name = c)
    fig.append_trace(trace2, i+1, 2)
    
fig["layout"].update(height = 1200, width = 800,
                     title = "Returns Open Prev Mktres10 v.s. Returns Open Next Mktres10 ")
iplot(fig)
fig = tools.make_subplots(rows = len(top_trade_code), cols = 2)
for i, c in enumerate(top_trade_code):
    temp = known_open_Mktres[(known_open_Mktres["assetCode"] == c) & 
                             (known_open_Mktres["time"] > "2008-12-31") &
                             (known_open_Mktres["time"] < "2010-01-01")]
    color = generate_color()
    trace1 = go.Scatter(x = temp["time"],
                        y = temp["returnsOpenPrevMktres10"],
                        marker = dict(color = color),
                        name = c)
    fig.append_trace(trace1, i+1, 1)
    trace2 = go.Scatter(x = temp["time"],
                        y = temp["returnsOpenNextMktres10"],
                        marker = dict(color = color),
                        name = c)
    fig.append_trace(trace2, i+1, 2)
    
fig["layout"].update(height = 1200, width = 800,
                     title = "Returns Open Prev Mktres10 v.s. Returns Open Next Mktres10 ")
iplot(fig)
known_close_Mktres = market_train[market_train["returnsClosePrevMktres10"].notnull()]

fig = tools.make_subplots(rows = len(top_trade_code), cols = 2)
for i, c in enumerate(top_trade_code):
    temp = known_close_Mktres[known_close_Mktres["assetCode"] == c]
    color = generate_color()
    trace1 = go.Scatter(x = temp["time"],
                        y = temp["returnsClosePrevMktres10"],
                        marker = dict(color = color),
                        name = c)
    fig.append_trace(trace1, i+1, 1)
    trace2 = go.Scatter(x = temp["time"],
                        y = temp["returnsOpenNextMktres10"],
                        marker = dict(color = color),
                        name = c)
    fig.append_trace(trace2, i+1, 2)
    
fig["layout"].update(height = 1200, width = 800,
                     title = "Returns Close Prev Mktres10 v.s. Returns Open Next Mktres10 ")
iplot(fig)
fig = tools.make_subplots(rows = len(top_trade_code), cols = 2)
for i, c in enumerate(top_trade_code):
    temp = known_close_Mktres[(known_close_Mktres["assetCode"] == c) & 
                              (known_close_Mktres["time"] > "2008-12-31") &
                              (known_close_Mktres["time"] < "2010-01-01")]
    color = generate_color()
    trace1 = go.Scatter(x = temp["time"],
                        y = temp["returnsClosePrevMktres10"],
                        marker = dict(color = color),
                        name = c)
    fig.append_trace(trace1, i+1, 1)
    trace2 = go.Scatter(x = temp["time"],
                        y = temp["returnsOpenNextMktres10"],
                        marker = dict(color = color),
                        name = c)
    fig.append_trace(trace2, i+1, 2)
    
fig["layout"].update(height = 1200, width = 800,
                     title = "Returns Close Prev Mktres10 v.s. Returns Open Next Mktres10 ")
iplot(fig)
universe_count = market_train["universe"].value_counts()
trace = go.Bar(x = ["Score", "Not Score"],
               y = [universe_count.values[0], universe_count.values[1]],
               marker = dict(color = ["blue", "red"]))
layout = dict(title = "Universe Bar Plot",
              yaxis = dict(title = "Count"))
iplot(dict(data = [trace], layout = layout))
news_train.head()
news_train.describe()
news_train.isnull().sum()
news_train["time"] = news_train["time"].dt.strftime(date_format = '%Y-%m-%d')
news_train["sourceTimestamp"] = news_train["sourceTimestamp"].dt.strftime(date_format = '%Y-%m-%d')
news_train["firstCreated"] = news_train["firstCreated"].dt.strftime(date_format = '%Y-%m-%d')
temp = news_train[news_train["assetName"] == "Bank of America Corp"]
temp.head()
headline_tag_count = news_train["headlineTag"].value_counts()
trace = go.Bar(x = headline_tag_count.index,
               y = headline_tag_count.values,
               marker = dict(color = [generate_color() for _ in range(len(headline_tag_count.index))]))
layout = dict(title = "Headline Tag Bar Plot",
              xaxis = dict(title = "Headline Tag"),
              yaxis = dict(title = "Count"))
iplot(dict(data = [trace], layout = layout))
trace = go.Bar(x = headline_tag_count.index[1:],
               y = headline_tag_count.values[1:],
               marker = dict(color = [generate_color() for _ in range(len(headline_tag_count.index[1:]))]))
layout = dict(title = "Headline Tag Bar Plot",
              xaxis = dict(title = "Headline Tag"),
              yaxis = dict(title = "Count"))
iplot(dict(data = [trace], layout = layout))
urgency_count = news_train["urgency"].value_counts()
trace = go.Bar(x = urgency_count.index,
               y = urgency_count.values,
               marker = dict(color = [generate_color(), generate_color()]))
layout = dict(title = "News Urgency Count")
iplot(dict(data = [trace], layout = layout))
provider_count = news_train["provider"].value_counts()
trace = go.Bar(x = provider_count.index,
               y = provider_count.values,
               marker = dict(color = [generate_color() for _ in range(len(provider_count.index))]))
layout = dict(title = "News Provider & Their News Count")
iplot(dict(data = [trace], layout = layout))
company_count = news_train["companyCount"].unique()

for year in range(2007, 2017):
    data = []
    for c in list(sorted(company_count)):
        temp = news_train[(news_train["companyCount"] == c) & 
                          (news_train["time"] >= "{}-01-01".format(year)) & 
                          (news_train["time"] < "{}-01-01".format(year+1))]
        trace = go.Box(y = temp["bodySize"],
                       name = "Company Count = {}".format(c),
                       marker = dict(color = generate_color()),
                       boxpoints = False)
        data.append(trace)

    layout = dict(title = "Body Size v.s. Company Count Box Plot in {}".format(year),
                  xaxis = dict(title = "Company Count"),
                  yaxis = dict(title = "Body Size"))
    iplot(dict(data = data, layout = layout))
years = list(range(2007, 2017))
fig = tools.make_subplots(rows = len(years), cols = 1)

for i, year in enumerate(years):
    temp = news_train[(news_train["assetName"] == "Bank of America Corp") & 
                      (news_train["time"] >= "{}-01-01".format(year)) & 
                      (news_train["time"] < "{}-01-01".format(year+1))]
    trace = go.Scatter(x = temp["sentenceCount"],
                       y = temp["wordCount"],
                       name = year,
                       mode = "markers",
                       marker = dict(color = generate_color()))
    fig.append_trace(trace, i+1, 1)

fig["layout"]["xaxis{}".format(len(years))].update(title = "Sentence Count")
fig["layout"].update(height = 1200, width = 800,
                     title = "Sentence Count v.s. Word Count of Bank of America")
iplot(fig)
def candle_sticks(data):
    data["high"] = data["open"]
    data["low"] = data["close"]
    
    for idx, row in data.iterrows():
        if row["close"] > row["open"]:
            data.loc[idx, "high"] = row["close"]
            data.loc[idx, "low"] = row["open"]
            
    return data
temp = news_train[(news_train["assetName"] == "Bank of America Corp") &
                  (news_train["time"] > "2008-12-31") &
                  (news_train["time"] < "2009-02-01")]
trace1 = go.Scatter(x = temp["time"],
                    y = temp["relevance"],
                    mode = "markers+lines",
                    marker = dict(color = generate_color()),
                    name = "Relevance")
trace2 = go.Scatter(x = temp["time"],
                    y = temp["sentimentNegative"],
                    mode = "markers",
                    marker = dict(color = generate_color()),
                    name = "Sentiment Negative")
trace3 = go.Scatter(x = temp["time"],
                    y = temp["sentimentNeutral"],
                    mode = "markers",
                    marker = dict(color = generate_color()),
                    name = "Sentiment Neutral")
trace4 = go.Scatter(x = temp["time"],
                    y = temp["sentimentPositive"],
                    mode = "markers",
                    marker = dict(color = generate_color()),
                    name = "Sentiment Positive")
layout = dict(title = "Relevance & Sentiment over time (Bank of America Corp)",
              xaxis = dict(title = "Day"))
iplot(dict(data = [trace1, trace2, trace3, trace4], layout = layout))

temp = market_train[(market_train["assetCode"] == "BAC.N") & 
                    (market_train["time"] > "2008-12-31") & 
                    (market_train["time"] < "2009-02-01")][["time", "open", "close"]]

temp2 = candle_sticks(temp)
    
trace = go.Candlestick(x = temp2["time"],
                       open = temp2["open"],
                       low = temp2["low"],
                       high = temp2["high"],
                       close = temp2["close"],
                       increasing = dict(line = dict(color = generate_color())),
                       decreasing = dict(line = dict(color = generate_color())))
    
layout = dict(title = "Candlestick Chart for Bank of America Corp in Jan 2009", 
              xaxis = dict(title = "Day"),
              yaxis = dict(title = "Price (USD)"))
iplot(dict(data = [trace], layout = layout))

