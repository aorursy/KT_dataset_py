import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

data = pd.read_csv("../input/nifty50-stock-market-data/ADANIPORTS.csv")

tbl = pd.DataFrame(data,columns=['Date','Prev Close','Open','High','Low','Last','Close'])

pd.set_option('display.max_rows', tbl.shape[0]+1)

tbl = tbl.sort_values(['Date'],ascending=False)

#print(tbl.head(5))

#Table View
labels = ['Date','Open','High','Low','Last','Close']

fig = go.Figure(data = [go.Table(header=dict(values=labels),
                 cells=dict(values=[tbl.Date,tbl.Open,tbl.High,tbl.Low,tbl.Last,tbl.Close]))
                     ])
fig.update_layout(
    title='Adani Ports Stock Data :',
)
fig.show()

#Pie Chart

fig1 = go.Figure()

fig1.add_trace(go.Scatter(x=tbl.Date, y=tbl['High'],mode='lines',name='High',marker=dict(color="green")))
fig1.add_trace(go.Scatter(x=tbl.Date, y=tbl['Low'],mode='lines',name='Low',marker=dict(color="red")))
#fig1.add_trace(go.Scatter(x=tbl.Date, y=tbl['Close'],mode='lines',name='Close',marker=dict(color="orange")))

fig1.update_layout(title='Adani Ports Stock with High and Low',template='plotly_dark')

fig1.show()