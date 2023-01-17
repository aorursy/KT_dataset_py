import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input/video-game-sales-with-ratings"))
df = pd.read_csv('../input/video-game-sales-with-ratings/Video_Games_Sales_as_at_22_Dec_2016.csv')
df.head()
df.shape
df.dtypes
df.isnull().sum()
df = df.dropna(how='any', subset=['Name','Year_of_Release','Publisher'])
df.shape
df.isnull().sum()
df.shape
df.dtypes
df['User_Score'].unique()
df = df[df.User_Score != 'tbd']
df['User_Score'].unique()
df['User_Score']=df['User_Score'].astype(float)
df.dtypes
df['User_Score'].fillna(df['User_Score'].mean(), inplace=True)
df['User_Score'].mean()
df['Critic_Score'].mean()
df['Critic_Score'].fillna(df['Critic_Score'].mean(), inplace=True)

df['Critic_Count'].fillna(df['Critic_Count'].mean(), inplace=True)

df['User_Score'].fillna(df['User_Score'].mean(), inplace=True)

df['User_Count'].fillna(df['User_Count'].mean(), inplace=True)
df.isnull().sum()
import plotly.express as px

globalsalesdata = df.groupby('Name', as_index=False).agg({"Global_Sales":"mean"})



ord_global_sales = globalsalesdata.sort_values(by='Global_Sales', ascending=False)

ord_global_sales = ord_global_sales[:10]

fig = px.bar(ord_global_sales, x='Name', y='Global_Sales',

             labels={'Global_Sales':'Global Sales', 'Name' : 'Name of Game'})

fig.update_traces(marker_color='rgb(255,204,153)', marker_line_color='rgb(255,102,102)',

                  marker_line_width=1.5, opacity=0.8)



fig.update_layout(title={'text': "Top 10 Best Seller Video Games", 'y': 0.95, 'x':0.5, 'xanchor': 'center', 'yanchor' : 'top'})

fig.show()
import plotly.express as px

yeardata = df.groupby('Year_of_Release', as_index=False).agg({"Name":"count"})



fig = px.bar(yeardata, x='Year_of_Release', y='Name',

             hover_data=['Name'], color='Name',

             labels={'Name':'Number of Games', 'Year_of_Release' : 'Year'})



fig.update_layout(title={'text': "Number of games every year", 'y': 0.95, 'x':0.5, 'xanchor': 'center', 'yanchor' : 'top'})

fig.show()
# libraries

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

 
sales_publisher = df.groupby('Publisher', as_index=False).agg({"Global_Sales":"sum"})

ord_sales_publisher = sales_publisher.sort_values(by='Global_Sales', ascending=False)



ord_sales_publisher = ord_sales_publisher[:15]

ord_sales_publisher = ord_sales_publisher.sort_values(by='Global_Sales', ascending=True)

my_range=range(1,len(ord_sales_publisher.index)+1)

import seaborn as sns



plt.hlines(y=my_range, xmin=0, xmax=ord_sales_publisher['Global_Sales'], color='skyblue')

plt.plot(ord_sales_publisher['Global_Sales'], my_range, "o")

# Add titles and axis names

plt.yticks(my_range, ord_sales_publisher['Publisher'])

plt.title("15 Publisher dengan Penjualan Tertinggi ", loc='left')

plt.xlabel('Penjualan Global')

plt.ylabel('Publisher')
import plotly.graph_objects as go



#ambil data (rata2)

dfGlobal = df.sort_values(by="Global_Sales", ascending=False)

dfGlobal = dfGlobal[:10].mean()



labels = ['North America Sales', 'Europe Sales', 'Japan Sales', 'Other Sales']

values = dfGlobal[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])



fig.update_layout(title={'text': "Percentage of Sales", 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor' : 'top'})

fig.show()
df_penj=df.sort_values(by='Global_Sales', ascending=False)

df_penj = df_penj[:10]

df_score= df_penj[['Name','User_Score']]

crit_score = df_penj['Critic_Score']/10

df_score=pd.concat([df_score,crit_score], axis=1)

df_score
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Bar(

    x=df_score['Name'],

    y=df_score['Critic_Score'],

    name='Critic Score',

    marker_color='indianred'

))

fig.add_trace(go.Bar(

    x=df_score['Name'],

    y=df_score['User_Score'],

    name='User Score',

    marker_color='lightsalmon'

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(barmode='group', xaxis_tickangle=-45)

fig.update_layout(title={'text': "Kritik dan Skor Penilaian Pengguna pada 5 Penjualan Teratas", 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor' : 'top'})

fig.show()