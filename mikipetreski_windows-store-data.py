import pandas as pd
import numpy as np
import plotly as py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import datetime

init_notebook_mode(connected=True)
df = pd.read_csv('../input/windows-store/msft.csv')
print(df.head())
#1. Find out the types of the columns and their names
#2. Discover the shape of the dataframe
print(df.dtypes)
print(df.shape)
#From above we can see that column Date is stored as object and we need it to be in datetime
df['Date'] = pd.to_datetime(df['Date'])
print(df.dtypes)
#Using the info method we can see that this dataframe is 99% complete
df.info()
#In each column there is 1 row that contains nan values - so let's remove them.
df = df.dropna()
#The column name "No of people Rated" is not good so let's change it
df.rename(columns={'No of people Rated':'No_Ratings'}, inplace=True)
#Let's plot and see how the ratings change over the years
df_yr = df.groupby([df['Date'].dt.year]).agg({'No_Ratings':'sum'}).reset_index()
# df_yr.tail()

fig = go.Figure(data=[go.Scatter(
    x=df_yr['Date'], 
    y=df_yr['No_Ratings'],
    line=dict(width=4),
    mode='lines+markers+text',
    text=df_yr['No_Ratings'],
    textposition="top center",
    marker=dict(color='#934057', size=8)
)])



fig.update_layout(
    plot_bgcolor="#31334e",
    paper_bgcolor='#31334e',
    title={'text':"<b>Number of Ratings per Year</b>", 'x':0.5},
    xaxis_title='Years',
    yaxis_title='Number of Ratings',
    font=dict(color='#9fa6af'),
    margin=dict(t=70,l=80,b=60,r=40),
    xaxis_showgrid=False,
    yaxis_showgrid=True,
    separators=".,",
)

fig.update_xaxes(tickfont=dict(color='#9fa6af'))
fig.update_yaxes(tickfont=dict(color='#9fa6af'))

py.offline.iplot(fig)
#To simplify the graph below, let's add another column Year
df_cat_yr = df.copy()
df_cat_yr['Year'] = [t.year for t in df_cat_yr['Date']]
df_cat_yr.head()
#Let's see how the categories progress in number of apps over the years
df_cat_yr = df_cat_yr.groupby(['Year', 'Category']).agg({'Name':'count'}).rename(columns={'Name':'No_Apps'}).reset_index()

fig = px.bar(df_cat_yr, x="Year", y="No_Apps", color="Category")

fig.update_layout(
    plot_bgcolor="#31334e",
    paper_bgcolor='#31334e',
    title={'text':"<b>Number of Apps per Category 2010-2020</b>", 'x':0.5},
    yaxis_title='Number of Apps',
    xaxis_title='',
    font=dict(color='#9fa6af'),
    margin=dict(t=70,l=80,b=60,r=40),
    xaxis_showgrid=False,
    yaxis_showgrid=False,
)

fig.update_xaxes(tickfont=dict(color='#9fa6af'))
fig.update_yaxes(tickfont=dict(color='#9fa6af'))

py.offline.iplot(fig)
#Let's create new dataframe to discover which category is most popular and how many apps by categories
df_app_cat = df.groupby('Category').agg({'Name':'count','No_Ratings':'sum'}).reset_index()
df_app_cat.rename(columns={'Name':'No_Apps'}, inplace=True)

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Bar(
             x=df_app_cat['Category'],
             y=df_app_cat['No_Ratings'],
             name="Number of Ratings",
             marker_color='#a3a6b5',
),
             secondary_y=False,)

fig.add_trace(go.Scatter(
             x=df_app_cat['Category'],
             y=df_app_cat['No_Apps'],
             name="Number of Apps",
             mode='markers+lines',
             marker_color='#df4f80',
),
             secondary_y=True,)

fig.update_layout(
    plot_bgcolor="#31334e",
    paper_bgcolor='#31334e',
    title={'text':"<b>Most popular Categories</b>", 'x':0.5},
#     xaxis_title='Category',
    yaxis_title='Number of Ratings',
    font=dict(color='#9fa6af'),
    margin=dict(t=70,l=80,b=60,r=40),
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    xaxis_tickangle=-45,
    legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="center",
                x=0.2)
)

fig.update_xaxes(tickfont=dict(color='#9fa6af'))
fig.update_yaxes(tickfont=dict(color='#9fa6af'))
fig.update_yaxes(title_text="Number of Apps",tickfont=dict(color='#9fa6af'), secondary_y=True)

py.offline.iplot(fig)
df_top_app = df.groupby('Name').agg({'No_Ratings':'sum'}).reset_index().nlargest(10,'No_Ratings')
df_top_app = df_top_app.sort_values('No_Ratings', ascending=False)
df_top_app.head()
#Let's create new dataframe to discover which category is most popular and how many apps by categories
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Bar(
             x=df_top_app['No_Ratings'],
             y=df_top_app['Name'],
             name="Applications",
             marker_color='#a3a6b5',
             orientation='h'
),
             secondary_y=False,)

# fig.add_trace(go.Scatter(
#              x=df_app_cat['Category'],
#              y=df_app_cat['No_Apps'],
#              name="Number of Apps",
#              mode='markers+lines',
#              marker_color='#df4f80',
# ),
#              secondary_y=True,)

fig.update_layout(
    plot_bgcolor="#31334e",
    paper_bgcolor='#31334e',
    title={'text':"<b>Top 10 Most Popular Applications</b>", 'x':0.5},
    yaxis_title='Number of Ratings',
    font=dict(color='#9fa6af'),
    margin=dict(t=70,l=80,b=60,r=40),
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    xaxis_tickangle=-45,
    legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="center",
                x=0.2)
)

fig.update_xaxes(tickfont=dict(color='#9fa6af'))
fig.update_yaxes(tickfont=dict(color='#9fa6af'))
fig.update_yaxes(title_text="Number of Apps",tickfont=dict(color='#9fa6af'), secondary_y=True)

py.offline.iplot(fig)
#Let's create new dataframe with calculation of % Free & Paid apps.
data = {'Free':[((df["Price"] == 'Free').sum() / len(df.index) * 100).round(1)],
        'Paid':[((df["Price"] != "Free").sum() / len(df.index) * 100).round(1)]}
df_prc = pd.DataFrame(data, columns=['Free', 'Paid'])

df_prc.head()
labels = df_prc.columns
values = [97,3]

fig = go.Figure(data=[go.Pie(
    labels=labels, 
    values=values, 
    hole=.3,
    rotation=90,
    textinfo='label+percent',
    marker=dict(colors=['#a3a6b5','#df4f80'])
#     color_discrete_map={'Free':'#a3a6b5','Paid':'df4f80'}
)])

fig.update_layout(
    plot_bgcolor="#31334e",
    paper_bgcolor='#31334e',
    title={'text':"<b>Ration between Free and Paid apps</b>", 'x':0.5},
    font=dict(color='#9fa6af'),
    margin=dict(t=70,l=80,b=60,r=40),
    showlegend=False
)


py.offline.iplot(fig)
