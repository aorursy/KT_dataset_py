import numpy as np

import pandas as pd

import requests

from datetime import datetime

from datetime import timedelta
url = {}

url['confirmed'] = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

url['deaths'] = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

url['recovered'] = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
def getData(url):

    html = requests.get(url)



    if html.status_code != 200:

        return None

    

    return html
for k, v in url.items():

    with open(k+'.csv', 'w+', encoding='utf8') as file:

        file.write(getData(v).text)
today = datetime.today()

base_report_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/'



html = None



while html is None:

    report_url = base_report_url + today.strftime('%m-%d-%Y') + '.csv'

    print('Try downloading data for', today.strftime('%m-%d-%Y'))

    html = getData(report_url)

    

    if html is None:

        print('No data for', today.strftime('%m-%d-%Y'), 'yet!')

        today = today - timedelta(days=1)

    else:

        print('Data downloaded!')

        with open('daily_report.csv', 'w+', encoding='utf8') as file:

            file.write(html.text)

        break
import pandas as pd



df_confirmed = pd.read_csv('confirmed.csv')

df_deaths = pd.read_csv('deaths.csv')

df_recovered = pd.read_csv('recovered.csv')

# df_daily_report = pd.read_csv('daily_report.csv')
print(f'Total row in df_confirmed: {df_confirmed.shape[0]}')

print(f'Number of unique countries: {df_confirmed.groupby("Country/Region").count().shape[0]}')

df_confirmed.head()



print(f'Total row in df_deaths: {df_deaths.shape[0]}')

print(f'Number of unique countries: {df_deaths.groupby("Country/Region").count().shape[0]}')

df_deaths.head()
print(f'Total row in df_recovered: {df_recovered.shape[0]}')

print(f'Number of unique countries: {df_recovered.groupby("Country/Region").count().shape[0]}')

df_recovered.head()
df_confirmed = df_confirmed.groupby('Country/Region').sum()

df_deaths = df_deaths.groupby('Country/Region').sum()

df_recovered = df_recovered.groupby('Country/Region').sum()
df_confirmed.drop(['Lat', 'Long'], axis=1, inplace=True)

df_deaths.drop(['Lat', 'Long'], axis=1, inplace=True)

df_recovered.drop(['Lat', 'Long'], axis=1, inplace=True)



print(f'Confirmed DF {df_confirmed.shape}\nDeath DF {df_deaths.shape}\nRecovered DF {df_recovered.shape}')
# Generate country list from scraped data



country_list = df_confirmed.index.values

print(f'Total: {len(country_list)} countries')
try:

    df_countries = pd.read_csv('../input/countries-geocode/countries_geocode.csv')

except:

    df_countries = pd.DataFrame(columns=['Country/Region', 'Lat', 'Long'])

    df_countries['Country/Region'] = country_list



print(df_countries.shape)



# Find out new country that not already in the coord list



country_coord_list = df_countries['Country/Region'].tolist()



missing_country = list(set(country_list)-set(country_coord_list))



print(f'Missing countries list: {missing_country}')
from geopy import Nominatim

from geopy.extra.rate_limiter import RateLimiter

import numpy as np



if missing_country:

    new_row = [{'Country/Region': i, 'Lat': np.nan, 'Long': np.nan} for i in missing_country]

    df_countries = df_countries.append(new_row, ignore_index=True)

else:

    print('No countries with missing coordinates info!')



locator = Nominatim(user_agent='Kaggle_covid')

geocode = RateLimiter(locator.geocode, min_delay_seconds=1)

    

for i in df_countries.index:

    if (np.isnan(df_countries.loc[i, 'Lat'])) or (np.isnan(df_countries.loc[i, 'Long'])):

        print(f'Get coordinates for {df_countries.loc[i, "Country/Region"]}...')

        loc = geocode(df_countries.loc[i, 'Country/Region'])



        if loc is None:

            print(f'Get coordinates for {df_countries.loc[i, "Country/Region"]} failed!')

            continue

        else:

            df_countries.loc[i, 'Lat'] = loc.latitude

            df_countries.loc[i, 'Long'] = loc.longitude



print('Done!')



# Save to file

df_countries.to_csv('countries_geocode.csv', index=False)
df_countries.set_index('Country/Region', drop=True, inplace=True)

df_countries.tail()
df_confirmed.sample()
date_list = df_confirmed.columns.tolist()



final = {'Country/Region':[], 'Lat': [], 'Long': [], 'Date':[], 'Confirmed':[], 'Deaths':[], 'Recovered':[]}



for c in country_list:

    coord = df_countries.loc[c].tolist()

    lat = coord[0]

    long = coord[1]



    for d in date_list:

        final['Country/Region'].append(c)

        final['Lat'].append(lat)

        final['Long'].append(long)

        final['Date'].append(d)

        final['Confirmed'].append(df_confirmed.loc[c, d])

        final['Deaths'].append(df_deaths.loc[c, d])

        final['Recovered'].append(df_recovered.loc[c, d])



df_final = pd.DataFrame(final)
print(df_final.dtypes)

print(df_final.shape)

df_final.head()
# Save the dataframe to file

# df_final.to_csv('covid-19_time_series_combined_by_country.csv')
import plotly.graph_objects as go

from plotly.subplots import make_subplots
date_range = 30

num_country = 10
import numpy as np

import matplotlib.cm as cm

import matplotlib.colors as colors



def make_rainbow(num_color):

    colors_array = cm.rainbow(np.linspace(0, 1, num_color))

    return [colors.rgb2hex(i) for i in colors_array]
# Import data, in case you don't want to run the data scraping and combining section above

# df = pd.read_csv('../input/covid19-time-series-combined-by-country/covid-19_time_series_combined_by_country.csv')



# Use if you run this notebook from the beginning

df = df_final



# Convert Date column into datetime64

df['Date'] = df['Date'].astype('datetime64')
df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered']
# Get last date of the data

last_date = df['Date'].max()



df_latest = df[df['Date'] == last_date].sort_values(by=['Confirmed'], ascending = True).reset_index(drop=True)



df_top = df_latest.tail(num_country)

top_countries = df_top['Country/Region'].tolist()



df_top.tail()
plot = {}



cat_color = make_rainbow(4)

cat = ['Confirmed', 'Active', 'Recovered', 'Deaths']



color_list = {}

for k, v in zip(cat, cat_color):

    color_list[k] = v



plot['top'] = go.Figure()





for c in cat[1:]:

    plot['top'].add_trace(go.Bar(x=df_top['Country/Region'], y=df_top[c], marker_color=color_list[c], name=c, 

               text=df_top['Confirmed'].apply('{:,.0f}'.format).astype(str) + ' confirmed cases', 

               hovertemplate='<b>%{label}</b><br>%{fullData.name}: <b>%{y:,.0f}</b> / %{text}<extra></extra>'))



plot['top'].update_layout(title=go.layout.Title(text=f"Top {num_country} Countries with COVID-19 on {last_date.strftime('%m/%d/%Y')}",

                          font=go.layout.title.Font(size=30)), height=600, barmode='stack');



plot['top']
not_top_10 = df_latest.head(df_latest.shape[0]-num_country).groupby('Country/Region').sum().index



df_temp = df_latest.replace(not_top_10, 'Other Countries').groupby('Country/Region').sum()



pull_out = []



for c in df_temp.index:

    if c == 'Other Countries': pull_out.append(0.2)

    else: pull_out.append(0)



plot['pie'] = go.Figure(data=[go.Pie(labels = df_temp.index, values = df_temp['Confirmed'], pull = pull_out,

                                     textinfo='label+percent', insidetextorientation='radial')])



plot['pie'].update_layout(title=go.layout.Title(text=f"Top {num_country} vs other countries", font=dict(size=30)),

                          legend = go.layout.Legend(bordercolor = 'black', borderwidth = 1),

                          height = 600);



plot['pie'].show()
from ipywidgets import widgets



# Get the first date in dataframe

min_date = df['Date'].min()



# Define widgets

country = widgets.Dropdown(

    description='Country/Region: ',

    value='Vietnam',

    options=df['Country/Region'].unique().tolist()

)



yaxis_log = widgets.Checkbox(

    description='Overview - Logarithmic y-axis ',

    value=False,

)



start_date = min_date



start_date_w = widgets.DatePicker(

    description='From:',

    value = start_date,

    disabled=False

)



end_date_w = widgets.DatePicker(

    description='To:',

    value = last_date,

    disabled=False

)



full_range_btn = widgets.Button(

    description='View all days',

    disabled=False,

    button_style='',

    tooltip='View from 1/22/2020'

)



last_7_btn = widgets.Button(description = 'Last 7 days',

                           disabled = False)



# Create df to use for plot

df_temp = df[df['Country/Region'] == 'Vietnam'].reset_index(drop=True)



# Insert daily columns

def insert_daily(inp_df, col_list, prefix = 'd_'):

    for c in col_list:

        inp_df[prefix+c] = inp_df[c]



        for i in range (1, len(inp_df)):

            inp_df.loc[i, prefix+c] = inp_df.loc[i, c] - inp_df.loc[i-1, c]

    

    return inp_df



df_temp = insert_daily(df_temp, cat)



df_int = df_temp[(df_temp['Date'] >= start_date) & (df_temp['Date'] <= last_date)].reset_index(drop=True)



df_pie = df_int[df_int['Date'] == end_date_w.value][['Active', 'Recovered', 'Deaths']].reset_index(drop=True).transpose()



# Draw initial plot

plot['Interactive'] = make_subplots(rows=2, cols=2, column_widths=[0.7, 0.3],

                                    shared_xaxes=True, vertical_spacing=0.05,

                                    specs=[[{"type": "xy"}, {"type": "domain", "rowspan": 2}], [{"type": "xy"}, None]],

                                    subplot_titles=("Overview","Distribution", "Daily change"))



pie_colors =[color_list[i] for i in cat[1:]]



# Overview chart

for k in cat:

    plot['Interactive'].add_trace(go.Scatter(x=df_int['Date'], y=df_int[k], mode='lines+markers', line_shape='spline',

                                             marker=dict(size=4, color=color_list[k]), name=k, text=df_int[k].astype(str)+' '+k),

                                  row=1, col=1)



# Pie chart

plot['Interactive'].add_trace(go.Pie(labels = df_pie.index, values = df_pie[0], pull = pull_out,

                                     textinfo='label+value+percent', insidetextorientation='radial',

                                    marker=dict(colors=pie_colors),

                                    name='Distribution',

                                    showlegend=False),

                              row=1, col=2)



# Daily chart

for k in cat:

    plot['Interactive'].add_trace(go.Scatter(x=df_int['Date'], y=df_int['d_'+k],

                                             name=f'Daily {k}', marker=dict(size=4, color=color_list[k]),

                                             mode='lines+markers', line_shape='spline',

                                            showlegend=False),

                                 row=2, col=1)

    

plot['Interactive'].update_layout(title=dict(text='COVID-19 in Vietnam', font=dict(size=30)), height=700, yaxis_type = '-')



g = go.FigureWidget(data=plot['Interactive'],

                   layout=go.Layout(height = 700))



def validate():

    if country.value in df['Country/Region'].unique():

        

        if start_date_w.value < min_date:

            start_date_w.value = min_date

        

        if end_date_w.value > last_date:

            end_date_w.value = last_date

        

        if start_date_w.value > end_date_w.value:

            start_date_w.value, end_date_w.value = end_date_w.value, start_date_w.value

        

        return True

    else:

        return False





def response(change):

    if validate():

        global df_temp, df_int, df_pie

        df_temp = df[df['Country/Region'] == country.value].reset_index(drop=True)

        df_temp = insert_daily(df_temp, cat)



        df_int = df_temp[(df_temp['Date'] >= pd.Timestamp(start_date_w.value)) & (df_temp['Date'] <= pd.Timestamp(end_date_w.value))].reset_index(drop=True)

        df_pie = df_int[df_int['Date'] == end_date_w.value][['Active', 'Recovered', 'Deaths']].reset_index(drop=True).transpose()



        with g.batch_update():

            idx = 0

            

            # Update Overview chart

            for k in cat:

                g.data[idx].x = df_int['Date']

                g.data[idx].y = df_int[k]

                g.data[idx].name = k

                g.data[idx].text = df_int[k].apply('{:,.0f}'.format).astype(str)+' '+k

                idx += 1

            

            # Update Distribution chart

            g.data[idx].labels = df_pie.index

            g.data[idx].values = df_pie[0]

            idx += 1

            

            # Update Daily change chart

            for k in cat:

                g.data[idx].x = df_int['Date']

                g.data[idx].y = df_int['d_'+k]

                g.data[idx].name = f'Daily {k}'

                g.data[idx].text = df_int['d_'+k].apply('{:,.0f}'.format).astype(str)+' '+k

                idx += 1

            

            g.layout.title.text = f'COVID-19 in {country.value}'

                

    else: print('Error')



def response_log(change):

    g.layout.yaxis.type = 'log' if yaxis_log.value else '-'

#     g.layout.yaxis2.type = 'log' if yaxis_log.value else '-'



def response_fullrange(change):

    start_date_w.value = min_date

    end_date_w.value = last_date



def response_7(change):

    start_date_w.value = last_date - pd.to_timedelta(7, unit='day')

    end_date_w.value = last_date

        

country.observe(response, names="value")

start_date_w.observe(response, names='value')

end_date_w.observe(response, names='value')

yaxis_log.observe(response_log, names='value')

full_range_btn.on_click(response_fullrange)

last_7_btn.on_click(response_7)



# Define interactive widget layout

row1 = widgets.HBox([country])

row2 = widgets.HBox([start_date_w, end_date_w, full_range_btn, last_7_btn])

row3 = widgets.HBox([yaxis_log])

widgets.VBox([row1, row2, g, row3])