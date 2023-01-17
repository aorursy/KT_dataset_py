import pandas as pd
df = pd.read_csv('../input/amazon-stocks-lifetime-dataset/AMZN.csv')

df.head()
import plotly.graph_objects as go
df = df.set_index(pd.DatetimeIndex(df['Date'].values))

df.head()
df1 = df.head(200)

df1.shape
figure = go.Figure(



    data = [

            go.Candlestick(

            x = df1.index,

            low = df1['Low'],

            high = df1['High'],

            close = df1['Close'],

            open = df1['Open'],

            increasing_line_color = 'orange',

            decreasing_line_color = 'black'

        )

    ]

)

figure.update_layout(

        title = 'Amazon Stock Prices',

        yaxis_title = 'Amazon stock prices ($)',

        xaxis_title = 'Date')

figure.show()