import pandas as pd
import numpy as np
from datetime import datetime

import plotly.graph_objects as go
import plotly.express as px
from ipywidgets import interact, widgets

from scipy.interpolate import interp1d
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# load the dataset 

death_df = pd.read_csv('/kaggle/input/covid19jhu/death.csv')
confirmed_df = pd.read_csv('/kaggle/input/covid19jhu/confirmed.csv')
recovered_df = pd.read_csv('/kaggle/input/covid19jhu/recovered.csv')
country_df = pd.read_csv('/kaggle/input/covid19jhu/country.csv')
# saving the data in my local as a csv

death_df.to_csv('death.csv', index=False)
confirmed_df.to_csv('confirmed.csv', index=False)
recovered_df.to_csv('recovered.csv', index=False)
country_df.to_csv('country.csv', index=False)
# printing the shape of each dataframe

print("The Shape of death_df is: ", death_df.shape)
print("The Shape of confirmed_df is: ", confirmed_df.shape)
print("The Shape of recovered_df is: ", recovered_df.shape)
print("The Shape of country_df is: ", country_df.shape)
# checking for null values

print(death_df.isnull().sum().head(2))
print(country_df.isnull().sum())

# column Province/State   contaim 185 null values, hence we will drop this columns
# droping the 'Province/State' columns as it containd null values

death_df.drop('Province/State', axis=1, inplace=True)
confirmed_df.drop('Province/State', axis=1, inplace=True)
recovered_df.drop('Province/State', axis=1, inplace=True)
country_df.drop(['People_Tested', 'People_Hospitalized'], axis=1, inplace=True)
# renaming column

death_df.rename(columns={'Country/Region': 'Country'}, inplace=True)
confirmed_df.rename(columns={'Country/Region': 'Country'}, inplace=True)
recovered_df.rename(columns={'Country/Region': 'Country'}, inplace=True)
country_df.rename(columns={'Country_Region': 'Country', 'Long_': 'Long'}, inplace=True)

death_df.head(3)
# un-pivot the dataframe

death_df2 = pd.melt(death_df, id_vars=['Country', 'Lat', 'Long', ], var_name='Date', value_name='Death_no')
confirmed_df2 = pd.melt(confirmed_df, id_vars=['Country', 'Lat', 'Long', ], var_name='Date', value_name='Confirmed_no')
recovered_df2 = pd.melt(recovered_df, id_vars=['Country', 'Lat', 'Long', ], var_name='Date', value_name='Recovered_no')
confirmed_df2.head()
# change the date to datetype

death_df2['Date'] = pd.to_datetime(death_df2['Date'])
confirmed_df2['Date'] = pd.to_datetime(confirmed_df2['Date'])
recovered_df2['Date'] = pd.to_datetime(confirmed_df2['Date'])
# sorting country_df with highest confirm rate

country_df.sort_values('Confirmed', ascending=False, inplace=True)
# checking for missing values

death_df2.isna().sum()
# checking the first five rows of the columns

death_df2.head()
# group by country and show data

def country_wise(country_name,df_type, number):
    # on select of category copy the dataframe to group by country
    if df_type == 'Confirmed cases':
        df_type = confirmed_df.copy(deep=True)
        category = 'COVID-19 confirmed cases'
        
    elif df_type == 'Death rate':
        df_type = death_df.copy(deep=True)
        category = 'COVID-19 Death rate'
        
    else:
        df_type = recovered_df.copy(deep=True)
        category = 'COVID-19 recovered cases'
        
    
    # group by country name
    country = df_type.groupby('Country')
    
    # select the given country
    country = country.get_group(country_name)
    
    # store daily death rate along with the date
    daily_cases = []
    case_date = []
    
    # iterate over each row
    for i, cols in enumerate(country):
        if i > 3:
            # take the sum of each column if there are multiple columns
            daily_cases.append(country[cols].sum())
            case_date.append(cols)
            zip_all_list = zip(case_date, daily_cases)
            
            # creata a data frame
            new_df = pd.DataFrame(data = zip_all_list, columns=['Date','coronavirus'])

    # append the country to the data frame
    new_df['Country'] = country['Country'].values[0]
    
    # get the daily death rate
    new_df = get_daily_date(new_df)
      
    # ploting the graph
    fig = px.line(new_df.iloc[-number:] ,
                  x='Date', y='coronavirus',
                  title='Daily ' + category +'  in ' + new_df['Country'].values[0])
    fig.update_layout(title_font_size=26,height=450)
    fig.show()

    return
# give the daily recovered case, death no, comfired cases

def get_daily_date(new_df):
    # calculation to get the no of death,recoverd, comfirmed cases for each day,
    # because each day new cases are added with the previous day cases in the csv
    
    new_df2 = new_df.copy(deep=True)
    for i in range(len(new_df) -1):
        new_df.iloc[i+1, 1] = new_df.iloc[1+i, 1] - new_df2.iloc[i, 1]
        if new_df.iloc[i+1, 1] < 0:
            new_df.iloc[i+1, 1] = 0
            
    return new_df
# select the country from the dropdown 

my_df_type = ['Confirmed cases', 'Death rate', 'Recovered cases']
drop_down = widgets.Dropdown(options=confirmed_df['Country'].unique().tolist(),
                                value='India',
                                description='Country',
                                disabled=False)


# slider to choose the number of days data to show

slider = widgets.IntSlider(value=40,
                              min=10,
                              max=len(death_df.columns[3:]),
                              step=1,
                              description='Select Days:',
                              disabled=False,
                              continuous_update=False,
                              orientation='horizontal',
                              readout=True,
                              readout_format='d')

# select the category 

category_drop_down = widgets.Dropdown(options=my_df_type,
                                value='Confirmed cases',
                                description='Category',
                                disabled=False)

# call the method on select or slide occour

interact(country_wise, country_name=drop_down, df_type=category_drop_down, number=slider);
# method to get the top 10 countries

def top_ten(number, sort_by):
    # sorting the columns with top death rate
    
    country_df.sort_values(by=sort_by, ascending=False, inplace=True)

    # sort country with highest number of cases
    
    fig = px.bar(country_df.head(number),
        x = "Country",
        y = sort_by,
        title= 'Top ' + str(number) +' Country - ' + sort_by + ' case',
        color="Country",
        height=500
    )
    
    fig.update_layout(title_font_size=26, height=550)
    
    return fig
# dropdown to select no of country and category

drop_down = widgets.Dropdown(options=list(range(1,31)),
                                value=10,
                                description='No Country',
                                disabled=False)

desc = widgets.Dropdown(options=country_df.loc[0:, ['Confirmed', 'Active', 'Deaths', 'Recovered', 'Mortality_Rate']].columns.tolist(),
                        value='Confirmed',
                        description='Category',
                        disabled=False)


interact(top_ten, number=drop_down, sort_by=desc);
# fixing the size of circle

margin = country_df['Confirmed'].values.tolist()
circel_range = interp1d([1, max(margin)], [0.2,12])
circle_radius = circel_range(margin)

# ploting the map

fig = px.scatter_mapbox(country_df, lat="Lat", lon="Long", hover_name="Country", hover_data=["Confirmed", "Deaths", 'Recovered'],
                        color_discrete_sequence=["#e60039"], zoom=1.5, height=500, size_max=50, size=circle_radius)
fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0}, height=450)
fig.show()
# confirmed and recovred cases

top_country = country_df.head(10)
top_country_name = list(top_country['Country'].values)

fig = go.Figure(data=[
    go.Bar(name='Confirmed',marker_color='rgb(100,20,205)', x=top_country_name, y=list(top_country['Confirmed'])),
    go.Bar(name='Recovered', marker_color='red',x=top_country_name, y=list(top_country['Recovered'])),
])

# Change the bar mode

fig.update_layout(barmode='group', height=600, title_text="Top 10 countires with Confirmed and Recovered case")
fig.show()