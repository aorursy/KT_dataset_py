import plotly.graph_objs as go

import pandas as pd
df = pd.read_csv("../input/bantul-classified-spi-1981-2018/Bantul_Classified_SPI_1981_2018.csv")

df1 = pd.read_csv('../input/bantul-yearly-classified-spi-1981-2018/Bantul_Yearly_Classified_SPI_1981_2018.csv')
new_df = df[['Date', 'D0','D1','D2','D3','D4']]

new_df.iloc[:,1:6] = (new_df.iloc[:,1:6]/20)*100

new_df.head()
fig = go.Figure()

fig.add_trace(go.Scatter(x=new_df.Date, y=new_df['D0'], name="Abnormally Dry",

                         line = dict(color = '#F9F753'), fill = 'tozeroy'))



fig.add_trace(go.Scatter(x=new_df.Date, y=new_df['D1'], name="Moderate Drought",

                         line = dict(color = '#F7D38B'), fill = 'tozeroy'))



fig.add_trace(go.Scatter(x=new_df.Date, y=new_df['D2'], name="Severe Drought",

                         line = dict(color = '#F3AC3D'), fill = 'tozeroy'))



fig.add_trace(go.Scatter(x=new_df.Date, y=new_df['D3'], name="Extreme Drought",

                         line = dict(color = '#D42D1F'), fill = 'tozeroy'))



fig.add_trace(go.Scatter(x=new_df.Date, y=new_df['D4'], name="Exceptional Drought",

                         line = dict(color = '#69110A'), fill = 'tozeroy'))



fig.layout.update(title_text='Drought Tracker - Bantul',

                  xaxis_rangeslider_visible=True)

fig.show()

new_df1 = df1[['Year', 'D0','D1','D2','D3','D4']]

new_df1.iloc[:,1:6] = (new_df1.iloc[:,1:6]/20)*100

new_df1.head()
fig = go.Figure()

fig.add_trace(go.Scatter(x=new_df1.Year, y=new_df1['D0'], name="Abnormally Dry",

                         line = dict(color = '#F9F753')))



fig.add_trace(go.Scatter(x=new_df1.Year, y=new_df1['D1'], name="Moderate Drought",

                         line = dict(color = '#F7D38B')))



fig.add_trace(go.Scatter(x=new_df1.Year, y=new_df1['D2'], name="Severe Drought",

                         line = dict(color = '#F3AC3D')))



fig.add_trace(go.Scatter(x=new_df1.Year, y=new_df1['D3'], name="Extreme Drought",

                         line = dict(color = '#D42D1F')))



fig.add_trace(go.Scatter(x=new_df1.Year, y=new_df1['D4'], name="Exceptional Drought",

                         line = dict(color = '#69110A')))



fig.layout.update(title_text='Yearly Trend - Bantul',

                  xaxis_rangeslider_visible=True)

fig.show()