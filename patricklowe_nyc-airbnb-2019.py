import numpy as np

import pandas as pd

# Graphing

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go

# Machine Learning



# Opening Files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df.nunique()
# Let's look at room types

print( df['room_type'].unique() )



# Replace 'Entire home/apt' with 'Entire Home' , not necessary but I feel it looks cleaner!

df['room_type'] = df['room_type'].replace('Entire home/apt','Entire Home')

print( df['room_type'].unique() )
# Remove unnecessary columns

df = df.drop(['id','name'],1)

#pd.value_counts(df['room_type'].isna(), normalize = True)
# Take a look at columns which may contain a lot of nulls

for col in df.columns:

    print( col, '{nulls:.2f}%'.format(nulls=(df[col].isna().sum() / df.shape[0]) * 100))

    

# Last Reviews and Reviews per month ar at 20%, 

# this seems okay as there may be new property entering the website
fig = px.box(df, 

             y="price",

            x = 'neighbourhood_group',

            points="all",

            color="room_type")



fig.update_layout(

    template="plotly_dark",

)



fig.show()
Q1 = df['price'].quantile(0.25)

Q3 = df['price'].quantile(0.75)

IQR = Q3 - Q1



filter = (df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 *IQR)

df = df.loc[filter]  
fig = px.box(df, 

             y="price",

            x = 'neighbourhood_group',

            color="room_type")

fig.show()
df['price'].replace('0', np.nan, inplace=True)

df.dropna(subset=['price'], inplace=True)
corr = df.corr()

fig = px.imshow(corr,

                color_continuous_scale='Oranges')



fig.update_layout(

    template="plotly_dark",

)

fig.show()

fig = px.density_mapbox(df, 

                        lat ='latitude', 

                        lon ='longitude', 

                        z = 'price', 

                        color_continuous_scale  = 'solar',

                        radius = 1,

                        center = dict(lat=40.75, lon=-73.9), 

                        zoom = 10,

                        mapbox_style = "carto-darkmatter",

                        )

fig.update_layout(

    title='NYC AirBnB - 2019',

    height=800,

    template="plotly_dark",

)



fig.show()
df_grouped = df.groupby(by=['neighbourhood_group','room_type'])['price'].mean().reset_index()

df_grouped
fig = make_subplots(rows=1, cols=3, shared_yaxes=True, )



fig.add_trace(

    go.Bar(

        x=df_grouped[ df_grouped["room_type"] =="Entire Home"]["neighbourhood_group"], 

        y=df_grouped[ df_grouped["room_type"] =="Entire Home"]["price"],

        name="Entire Home"),

    row=1, col=1

)



fig.add_trace(

    go.Bar(

        x=df_grouped[ df_grouped["room_type"] =="Shared room"]["neighbourhood_group"], 

        y=df_grouped[ df_grouped["room_type"] =="Shared room"]["price"],

        name="Shared room"

    ),

    row=1, col=2

)



fig.add_trace(

    go.Bar(

        x=df_grouped[ df_grouped["room_type"] =="Private room"]["neighbourhood_group"], 

        y=df_grouped[ df_grouped["room_type"] =="Private room"]["price"],

        name="Private room",),

    row=1, col=3

)



fig.update_layout(

    template="plotly_dark",

    title_text="Avg Price for Room Type, by Neighbourhood")

fig.show()
# Count of Owners and Properties

df_owners = df.groupby(by=['host_id'])['neighbourhood_group'].count().reset_index().sort_values(by=['neighbourhood_group'], ascending = False)

df_owners['bin'] = pd.cut(df_owners['neighbourhood_group'], [0, 1, 2, 3, 4, 5, 10, 15, 20,100], labels=['=1','=2', '=3', '=4','=5','5-10','10-15','15-20','20-100'])

df_owners = df_owners.groupby(by=['bin']).count().reset_index()

df_owners = df_owners.drop(['host_id'],1)

df_owners = df_owners.rename(columns={"neighbourhood_group": "count"})

df_owners
# Average Price by Binned Property Count

owners_avg_price = df.groupby(by=['host_id'])['price'].mean().reset_index().sort_values(by=['host_id'], ascending = False)

owners_count = df.groupby(by=['host_id'])['neighbourhood_group'].count().reset_index().sort_values(by=['host_id'], ascending = False)

owners_avg_price['count'] = owners_count['neighbourhood_group']

price_count = owners_avg_price.groupby(by=['count'])['price'].mean().reset_index().sort_values(by=['price'], ascending = False)
price_count['bin'] = pd.cut(price_count['count'], [0,1,2,3,4,5,10,15,20,50,100,150,200,300], labels=['=1','=2','=3','=4','=5','5-10','10-15','15-20','20-50','50-100','100-150','150-200','200-300'])

price_count = price_count.groupby(by=['bin'])['price'].mean().reset_index()

price_count
fig = make_subplots(rows=1, 

                    cols=2, 

                    subplot_titles=("Individuals Owning Property", "Avg Price ($) of Property, by amount of properties owned"),

                    #shared_yaxes=True, 

                   )



fig.add_trace(

    go.Bar(

        x=df_owners['bin'],

        y=df_owners['count'],

        name="Count"),

    row=1, col=1

)



fig.add_trace(

    go.Bar(

        x=price_count['bin'],

        y=price_count['price'],

        name="Price"),

    row=1, col=2

)





fig.update_layout(

    template="plotly_dark",

    title_text = 'Property Count / Avg Price',

)



fig.show()
busy_hosts = df

busy_hosts['days_booked'] = 365 - busy_hosts['availability_365']

most_booking = busy_hosts.groupby(by=['host_id','host_name'])['days_booked'].sum().reset_index()

most_booking['days_booked'] = np.ceil(most_booking['days_booked'])

most_booking = most_booking.sort_values(by=['days_booked'],ascending=False).head(5)

most_booking['host_id'] = most_booking['host_id'].astype(str)



avg_booking  = busy_hosts.groupby(by=['host_id','host_name'])['days_booked'].mean().reset_index()

avg_booking['days_booked'] = np.ceil(avg_booking['days_booked'])

avg_booking = avg_booking.sort_values(by=['days_booked'],ascending=False).head(5)

avg_booking['host_id'] = avg_booking['host_id'].astype(str)



busy_hosts['yearly_income'] = busy_hosts['price'] * busy_hosts['days_booked']

host_income = busy_hosts.groupby(by=['host_id','host_name'])['yearly_income'].sum().reset_index()

host_income = host_income.sort_values(by=['yearly_income'],ascending=False).head(5)

most_booking['host_id'] = most_booking['host_id'].astype(str)
fig = make_subplots(rows=1, 

                    cols=3, 

                    subplot_titles=("Most Bookings", "Top Avg Bookings","Most Income Generated"),

                   )



fig.add_trace(

    go.Bar(

        x=most_booking['host_name'],

        y=most_booking['days_booked'],

        name="Most Bookings"),

    row=1, col=1

)



fig.add_trace(

    go.Bar(

        x=avg_booking['host_name'],

        y=avg_booking['days_booked'],

        name="Avg Booking"),

    row=1, col=2

)



fig.add_trace(

    go.Bar(

        x=host_income['host_name'],

        y=host_income['yearly_income'],

        name="Most Income"),

    row=1, col=3

)



fig.update_layout(

    template="plotly_dark",

    title_text = 'Property Count / Avg Price',

)



fig.show()
# Property Count by Owner Name

named_prop = df.groupby(by=['host_name'])['host_id'].count().reset_index().sort_values(by=['host_id'],ascending=False)

named_prop = named_prop.sort_values(by=['host_id'],ascending=False).head(10)



fig = make_subplots(rows=1, 

                    cols=1, 

                   )



fig.add_trace(

    go.Bar(

        x=named_prop['host_name'],

        y=named_prop['host_id'],

        name="Amount of Properties owned"),

    row=1, col=1

)



fig.update_layout(

    template="plotly_dark",

    title_text = 'Property Count by Owner',

)



fig.show()
# Traffic by Areas (bookings)

traffic_areas = busy_hosts.groupby(by=['neighbourhood_group','neighbourhood'])['days_booked'].mean().reset_index().sort_values(by=['neighbourhood'])

traffic_areas2 = busy_hosts.groupby(by=['neighbourhood_group','neighbourhood'])['price'].mean().reset_index().sort_values(by=['neighbourhood'])



fig = make_subplots(rows=2, 

                    cols=1,

                    subplot_titles=("Bookings by Area", "Avg Prices by Area"),

                   )



fig.add_trace(

    go.Bar(

        x=traffic_areas['neighbourhood'],

        y=traffic_areas['days_booked'],

        name="Bookings by Area"),

    row=1, col=1

)



fig.add_trace(

    go.Bar(

        x=traffic_areas2['neighbourhood'],

        y=traffic_areas2['price'],

        name="Avg Price by Area"),

    row=2, col=1

)







fig.update_layout(

    template="plotly_dark",

    margin=dict(l=50, r=50, t=80, b=80),

    title_text = 'Area Info',

    height=800,

)



fig.show()