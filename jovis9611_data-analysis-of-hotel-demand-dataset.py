import os
import pandas as pd
import plotly.express as px

pd.set_option("display.precision", 2)

# Important code block for plotly graph display at Kaggle
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True) 


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
ds=pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')



print('ds.info')
print('='*30)
ds.info()
print('ds.describe')
print('='*30)
ds.describe() 
 

# Missing Value Check : Country, Children, Agent and Company columns are contain NULL value
# print('Missing value')
# print('='*30)
# df_missing_value=ds.isnull()
# for i in df_missing_value.columns:
#     d=df_missing_value[i].value_counts()
 
#     print(d)
#     print()
#     print('-'*30)


# Total guest by hotel. 
df_city=ds.loc[ds['hotel'] == 'City Hotel']
df_resort=ds.loc[ds['hotel'] == 'Resort Hotel']

# Get reservation status count for city hotel and resort hotel,
# Method: Group by 2 column and get the count
df_whole=ds.groupby(['reservation_status']).size().reset_index().rename(columns={0:'guest_count'})
df_guest_count=ds.groupby(['hotel', 'reservation_status']).size().reset_index().rename(columns={0:'guest_count'})

fig_w = px.bar(df_whole, 
             x="reservation_status", 
             y="guest_count", 
             title='Total Number of Guests by Reservation Status',
             text='guest_count',
            )
fig_w.update_traces(textposition='outside')

fig_w.show()

fig_h= px.bar(df_guest_count, 
             x="reservation_status", 
             y="guest_count", 
             color='hotel',
             title='Total Number of Guests by Reservation Status and Hotels',
             text='guest_count',
            )
fig_h.update_traces(textposition='outside')

fig_h.show()





import pycountry

df_country=ds.groupby([ 'reservation_status','country']).size().reset_index().rename(columns={0:'guest_count'})

list_alpha_2 = [i.alpha_2 for i in list(pycountry.countries)]
list_alpha_3 = [i.alpha_3 for i in list(pycountry.countries)]    

def country_flag(df):
    if (len(df['country'])==2 and df['country'] in list_alpha_2):
        return pycountry.countries.get(alpha_2=df['country']).name
    elif (len(df['country'])==3 and df['country'] in list_alpha_3):
        return pycountry.countries.get(alpha_3=df['country']).name
    else:
        return 'Invalid Code'

df_country['country_name']=df_country.apply(country_flag, axis = 1)

fig = px.scatter_geo(df_country, locations="country", color="country",
                     hover_name="country_name", size="guest_count",
                   
                     animation_frame="reservation_status",
                     projection="natural earth")


fig.update_layout(
        title_text = 'Total Guest Count By Country',

    showlegend = True,
        margin = dict(t=0, l=0, r=0, b=0),
    
       
    )



fig.show()

# Sunburst chart for countries
df_country_hotel=ds.groupby([ 'hotel','reservation_status','country']).size().reset_index().rename(columns={0:'guest_count'})
df_country_hotel['country_name']=df_country_hotel.apply(country_flag, axis = 1)

fig =px.sunburst(
    df_country_hotel,
    path=['hotel','reservation_status', 'country_name'],
    values='guest_count',
    color_continuous_scale='RdBu',
    color='guest_count',

)
fig.update_layout(
    margin = dict(t=10, l=10, r=10, b=10)
)



fig.update_traces(go.Sunburst(hovertemplate='<b>%{label} </b> <br><br>%{value:,.0f}',textinfo='label+percent parent'))



fig.show()
adults = ds['adults'].sum()
children=ds['children'].sum()
babies=ds['babies'].sum()


age_group={'age_group':['adults','children','babies'],
           'counts':[adults,children,babies]}
df_age_group=pd.DataFrame(age_group,columns=['age_group','counts'])


fig = px.pie(df_age_group, 
             values='counts', 
             names='age_group',
             title='Guest by Age Groups',
             hover_data=['age_group'], labels={'age_group':'Age Group'}
            
            )
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
fig.show()


df_repeated_guest=ds.groupby([ 'hotel','country'])['is_repeated_guest'].size().reset_index()



df_repeated_guest['country_name']=df_repeated_guest.apply(country_flag, axis = 1)

fig=px.sunburst(
    df_repeated_guest,
    path=['hotel','country_name'],
    values='is_repeated_guest',
    color_continuous_scale='RdBu',
    color='is_repeated_guest',
    maxdepth=2
)
fig.update_traces(go.Sunburst(hovertemplate='<b>%{label} </b> <br><br>%{value:,.0f}',textinfo='label+percent parent'))
fig.update_layout(

    margin = dict(t=0, l=0, r=0, b=0)
)

fig.show()
df_market_segment=ds.groupby(['hotel','market_segment','reservation_status']).size().reset_index().rename(columns={0:'guest_count'})
df_distribution_channel=ds.groupby([ 'hotel','reservation_status','distribution_channel']).size().reset_index().rename(columns={0:'guest_count'})


fig_m=px.sunburst(
    df_market_segment,
    path=['reservation_status','hotel','market_segment'],
    values='guest_count',
    color_continuous_scale='RdBu',
    color='guest_count',
    maxdepth=3
)


fig_d=px.sunburst(
    df_distribution_channel,
    path=['reservation_status','hotel','distribution_channel'],
    values='guest_count',
    color_continuous_scale='RdBu',
    color='guest_count',
    maxdepth=3,
    
)

fig_m.update_traces(go.Sunburst(hovertemplate='<b>%{label} </b> <br><br>%{value:,.0f}',textinfo='label+percent parent'))
fig_d.update_traces(go.Sunburst(hovertemplate='<b>%{label} </b> <br><br>%{value:,.0f}',textinfo='label+percent parent'))


fig_m.update_layout(
  
    margin = dict(t=0, l=0, r=0, b=0)
    
)


fig_d.update_layout(

    margin = dict(t=0, l=0, r=0, b=0)
)

fig_m.show()
fig_d.show()
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Actual Guest by year, weekday and weekend 

# Weekend and weekday
df_actual_guest=ds[ds['is_canceled']==0]
weekend_count=df_actual_guest['stays_in_weekend_nights'].sum()
weekday_count=df_actual_guest['stays_in_week_nights'].sum()

df_stay_length_wk=df_actual_guest.groupby('stays_in_weekend_nights').size().reset_index().rename(columns={0:'guest_count'})
df_stay_length_wd=df_actual_guest.groupby('stays_in_week_nights').size().reset_index().rename(columns={0:'guest_count'})

df_actualguest_by_year=df_actual_guest.groupby(['is_canceled','arrival_date_year']).size().reset_index().rename(columns={0:'guest_count'})


fig = make_subplots(rows=1, cols=2, specs=[[{},{"type": "pie"}]])

fig.add_trace(go.Scatter(x=list(df_actualguest_by_year.arrival_date_year),
                         y=list(df_actualguest_by_year.guest_count)),
                        
              row=1, col=1)

fig.add_trace(go.Pie(
     values=[weekday_count,weekend_count],
     labels=['Stay in weekday','Stay in weekend'],
     ), 
     row=1, col=2)



fig.update_layout(
                  title_text="Total Actual Guest by Time Factor")
fig.show()


fig_length_stay = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"},{"type": "xy"}]])

fig_length_stay.add_trace(go.Bar(x=list(df_stay_length_wk.stays_in_weekend_nights), y=list(df_stay_length_wk.guest_count),name='Total Count of Stay in Weekend Nights'
                   ),
           
     row=1, col=1)


fig_length_stay.add_trace(go.Bar(x=list(df_stay_length_wd.stays_in_week_nights), y=list(df_stay_length_wd.guest_count),name='Total Count of Stay in Week Nights'
                   ),
           
     row=1, col=2)



fig_length_stay.update_layout(
                  title_text="Total Actual Guest by Length Stay in Weekend and Weekday")
fig_length_stay.show()


df_meal=ds.groupby(['hotel','meal']).size().reset_index().rename(columns={0:'guest_count'})


fig = px.bar(df_meal, x="hotel", y="guest_count", color='meal', barmode='group',
             height=400,text='guest_count'
         )
fig.update_layout(title_text='Popular Meal Package Selection By Hotel')
fig.show()
df_parking_lot=ds.groupby(['hotel','required_car_parking_spaces']).size().reset_index().rename(columns={0:'guest_count'})


fig = px.bar(df_parking_lot, x="required_car_parking_spaces", y="guest_count", color='hotel',
             height=500,text='guest_count'
         )
fig.update_layout(title_text='Number of Car Parking Spaces Requirement By Hotel',barmode='group')
fig.show()