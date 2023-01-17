import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

# Load the Bitcoin price data
df = pd.read_csv('/kaggle/input/392-crypto-currency-pairs-at-minute-resolution/btcusd.csv')
# Create a working copy of the last 2 hours of the data
btc_data = df.iloc[-120:].copy()

# Create an interactive candlestick plot with Plotly
fig = go.Figure(data=[go.Candlestick(x=pd.to_datetime(btc_data.time, unit='ms'),
                open=btc_data['open'],
                high=btc_data['high'],
                low=btc_data['low'],
                close=btc_data['close'])])

# Add title and format axes
fig.update_layout(
    title='Bitcoin price in the last 2 hours as candlestick chart',
    yaxis_title='BTC/USD')

fig.show()
# Create a working copy of the last 2 hours of the data
btc_data = df.iloc[-240:].copy()

# Define the parameters for the Bollinger Band calculation
ma_size = 20
bol_size = 2

# Convert the timestamp data to a human readable format
btc_data.index = pd.to_datetime(btc_data.time, unit='ms')

# Calculate the SMA
btc_data.insert(0, 'moving_average', btc_data['close'].rolling(ma_size).mean())

# Calculate the upper and lower Bollinger Bands
btc_data.insert(0, 'bol_upper', btc_data['moving_average'] + btc_data['close'].rolling(ma_size).std() * bol_size)
btc_data.insert(0, 'bol_lower', btc_data['moving_average'] - btc_data['close'].rolling(ma_size).std() * bol_size)

# Remove the NaNs -> consequence of using a non-centered moving average
btc_data.dropna(inplace=True)

# Create an interactive candlestick plot with Plotly
fig = go.Figure(data=[go.Candlestick(x = btc_data.index,
                                     open = btc_data['open'],
                                     high = btc_data['high'],
                                     low = btc_data['low'],
                                     showlegend = False,
                                     close = btc_data['close'])])

# Plot the three lines of the Bollinger Bands indicator
for parameter in ['moving_average', 'bol_lower', 'bol_upper']:
    fig.add_trace(go.Scatter(
        x = btc_data.index,
        y = btc_data[parameter],
        showlegend = False,
        line_color = 'gray',
        mode='lines',
        line={'dash': 'dash'},
        marker_line_width=2, 
        marker_size=10,
        opacity = 0.8))
    
# Add title and format axes
fig.update_layout(
    title='Bitcoin price in the last 2 hours with Bollinger Bands',
    yaxis_title='BTC/USD')

fig.show()
