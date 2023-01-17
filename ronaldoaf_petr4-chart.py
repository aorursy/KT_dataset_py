import plotly.graph_objects as go



import pandas as pd

from datetime import datetime



df = pd.read_csv('/kaggle/input/brazil-stock-exchange-historical-quote-19952019/BOVESPA.csv')





df=df[(df.CODNEG=='PETR4') & (df.DATE>=20120101) ]

df['date']=df.DATE.astype(str).str[:4]+'-'+df.DATE.astype(str).str[4:6]+'-'+df.DATE.astype(str).str[6:]





fig = go.Figure(data=[go.Candlestick(x=df['date'],

                open=df['PREABE'],

                high=df['PREMAX'],

                low=df['PREMIN'],

                close=df['PREULT'])])



fig.show()