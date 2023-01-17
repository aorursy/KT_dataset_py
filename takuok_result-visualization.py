import plotly
import pandas as pd
plotly.offline.init_notebook_mode(connected=False)
path = "../input/my-rnn-avito/my_rnnv5val.csv"

y = pd.read_csv("../input/avito-demand-prediction/train.csv", usecols=["deal_probability", "item_id"])
y = y.rename(columns={"deal_probability": "y"})

df = pd.read_csv(path)
df = df.sample(n=1000)
df['deal_probability'] = df['deal_probability'].clip(0.0, 1.0) 
df = df.merge(y, how="left", on="item_id")
y_ = df.y.values
x = df["deal_probability"].values
# make trace
data = [
    plotly.graph_objs.Scatter(x = x,  y = y_, mode = "markers"),
    plotly.graph_objs.Scatter(x=[0,1], y=[0,1], name="legend2"),
]

# define layout
layout = plotly.graph_objs.Layout(
    title="result",
    xaxis=dict(title='pred'),
    yaxis=dict(title='true'),
    showlegend=False)

fig = dict(data=data, layout=layout)

plotly.offline.iplot(fig, filename="result")