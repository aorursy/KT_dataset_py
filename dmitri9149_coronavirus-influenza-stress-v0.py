import warnings

warnings.simplefilter(action='ignore', category=Warning)

import pandas as pd

pd.reset_option('all')
import numpy as np

import pandas as pd

import datetime



import matplotlib.pyplot as plt

import matplotlib.ticker as ticker



import seaborn as sns

import plotly.graph_objects as go 

import seaborn as sns

import plotly

import plotly.express as px

from fbprophet.plot import plot_plotly

from fbprophet import Prophet

from fbprophet.plot import add_changepoints_to_plot

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot, plot_mpl

import plotly.offline as py

init_notebook_mode(connected=True)



plt.rcParams.update({'font.size': 14})

data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates = ['ObservationDate','Last Update'])

data.shape
# Import data up to the 'Last Update'

# Our dataset looks like this:



dataset = (pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates = ['Last Update', 'ObservationDate'])

                       .sort_values(by='Last Update', ascending = False))

last_date = dataset['Last Update'].values.max()

print(f'The date of last update is {last_date}')

dataset.head(10)
#The empty values

dataset.isna().sum()
#Filling empty provinces with "NA"

#Rename Mainland China to China

dataset['Country/Region'].replace('Mainland China', 'China', inplace = True)

dataset['Province/State'].fillna('NA', inplace = True)
print('List of countries infected by Corona Virus')

list(dataset['Country/Region'].sort_values().unique())[0:50]
list_infected_countries = pd.DataFrame(data = dataset['Country/Region'].sort_values().unique(), columns = {'Country/Region'})



num_infected_countries = len(list(dataset['Country/Region'].sort_values().unique()))



print("There are %d countries infected by Corona Virus in the World \n \n" %

      len(list(dataset['Country/Region'].sort_values().unique())))



list_infected_countries
# Last observation date in dataset



last_data_day = dataset['ObservationDate'].values.max()

last_date = dataset['Last Update'].values.max()

#day = datetime.date(2020,3,1)

day = last_data_day





# Filtering the dataset with the selected date

df = dataset[dataset['ObservationDate'].eq(day)]





# Creating a dataset grouped by countries and sorted by confirmed cases

df_group_all = pd.DataFrame(data = (df.groupby(['Country/Region'], as_index = False)

      .sum()

      .sort_values(by='Confirmed', ascending=False)

      .reset_index(drop=True)))



# Removing 'SNo' column

df_group_all.drop(columns = ['SNo'], inplace = True)





df_group_all.head(40)
countries_south_hem= ['Australia', 'Bolivia', 'Brazil', 'Chile', 'Congo', 'Equador', 'Equatorial Guinea', 'Eswatini', 

                     'Gabon', 'Kenya', 'Namibia', 'Indonesia', 'Others', 'Paraguay', 'Reunion', 'Rwanda',

                     'Seychelles', 'South Africa', 'Uruguay', 'Argentina', 'Zambia', 'Somalia', 'Mayotte',

                      'Zimbabve', 'Papua New Guinea', 'Uganda']
df_south = df_group_all[df_group_all['Country/Region'].isin(countries_south_hem)]

south_confirm = df_south['Confirmed'].sum()

print(f'Confirmed cases in the South hemisphere: {south_confirm}')

total_confirm =  df_group_all.Confirmed.sum() 

print(f'Total confirmed cases for the world for now:  {total_confirm}')

print(f'Confirmed cases in the South hemisphere: {south_confirm/total_confirm:.2%} from total cases')

print(f'In the South hemisphere there are living about 10 % of the total human population')
def draw_barchart(day, most_infected):

    

# Creating Dataset of most_infected countries

    df = dataset[dataset['ObservationDate'].eq(day)]

    

    df_group = (df.groupby(['Country/Region'], as_index = False)

          .sum()

          .sort_values(by='Confirmed', ascending=False)

          .reset_index(drop=True))



    df_group.drop(columns = ['SNo'], inplace = True)

    

    #Creating Bar Chart

    ax.clear()

    df_group = df_group[most_infected::-1]

    ax.barh(df_group['Country/Region'], df_group['Confirmed'])

    

    dx = df_group['Confirmed'].max() / 1000

    

    #Format Bar Chart

    for i, (value, name) in enumerate(zip(df_group['Confirmed'], df_group['Country/Region'])):

        

        ax.text(value-dx, i,     name,           size=14, weight=600, ha='right', va='center')

        ax.text(value+dx, i,     f'{value:,.0f}',  size=14, ha='left',  va='center')

    

    ax.text(1, 0.4, day.strftime("%d/%m/%Y"), transform=ax.transAxes, color='#444444', size=30, ha='right', weight=600)

    ax.text(0, 1.06, 'Confirmed Cases', transform=ax.transAxes, size=12, color='#444444')

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    ax.xaxis.set_ticks_position('top')

    ax.tick_params(axis='x', colors='#444444', labelsize=12)

    ax.set_yticks([])

    ax.margins(0, 0.01)

    ax.grid(which='major', axis='x', linestyle='-')

    ax.set_axisbelow(True)

    ax.text(0, 1.12, 'Confirmed Corona Virus cases in the world',

            transform=ax.transAxes, size=24, weight=600, ha='left')

    ax.text(1, 0, '', transform=ax.transAxes, ha='right',

            color='#444444', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))

    plt.box(False)
fig, ax = plt.subplots(figsize=(12, 20))

draw_barchart(dataset['ObservationDate'].max(), 40)
# Drop some columns

dataset = dataset.drop(['SNo', 'Last Update', 'Province/State'], axis=1)

# Check null values

dataset.isnull().sum()
# we may check the dataset at diffrent time periods 

day_time = last_data_day

#day_time = datetime.date(2020, 3, 15)

dataset = dataset[dataset['ObservationDate'] <= day_time]
#This creates a table that sums up every element in the Confirmed, Deaths, and recovered columns.

temp = dataset.groupby('ObservationDate')['Confirmed', 'Deaths', 'Recovered'].sum()

temp = temp.reset_index()

temp = temp.sort_values('ObservationDate', ascending=False)

temp['Mortality Rate %']=temp['Deaths']/temp['Confirmed']*100

temp.head().style.background_gradient(cmap='Reds')
fig = go.Figure()

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=temp['ObservationDate'], 

                         y=np.log2(temp['Confirmed']),

                         mode='lines+markers',

                         name='Confirmed',

                         line=dict(color='Yellow', width=2)))

# Update yaxis properties

fig.update_yaxes(title_text="Log 2 Confirmed Cases")

fig.show()
delta = np.log2(temp[temp['ObservationDate']== datetime.date(2020,3,20)]['Confirmed'].values) - np.log2(temp[temp['ObservationDate']== datetime.date(2020,3,1)]['Confirmed'].values)

slope = float(delta/19.)

print(f'Confirmed cases double every {1./slope} days for the period 1 March ... 20 March')



delta = np.log2(temp[temp['ObservationDate']== datetime.date(2020,2,5)]['Confirmed'].values) - np.log2(temp[temp['ObservationDate']== datetime.date(2020,1,22)]['Confirmed'].values)

slope = float(delta/14.)

print(f'Confirmed cases double every {1./slope} days at the beginning of the infection')
def slope(country_list,from_date, last_date):

# dates to be in form : datetime.date(2020,3,16)

    Affected = dataset.copy()



    Affected = Affected[Affected['Country/Region'].isin(country_list)]

    Affected = Affected.groupby(['ObservationDate', 'Country/Region']).agg({'Confirmed': ['sum']})

    Affected.columns = ['Confirmed Cases']

    Affected = Affected.reset_index()



    Affected['Confirmed Cases']= np.log2(Affected['Confirmed Cases'])

#    Cases = Affected.copy()

    Cases_from_date = Affected[Affected['ObservationDate'] == from_date]

    Cases_last_date = Affected[Affected['ObservationDate']== last_date]



    Slope = Cases_last_date.copy()

    Confirmed_last_date = Cases_last_date['Confirmed Cases'].values

    Confirmed_from_date = Cases_from_date['Confirmed Cases'].values

#    Confirmed_03_01 = Cases_03_01['Confirmed Cases'].values

    Delta = Confirmed_last_date - Confirmed_from_date

    Slope['Delta']= Delta

    Slope['Delta'].mean()

    time_delta = last_date-from_date

    days = time_delta.days

#    print(days)

#    print(Slope.values)

    mean_slope = Slope['Delta'].mean()/float(days)

    #print(mean_slope)

    return 1./mean_slope
Affected = dataset.copy()

first_selection=['Germany','Finland', 'France',

         'Spain', 'UK', 'Greece','Czech Republic', 'Switzerland', 'Iceland', 'Romania']



Affected = Affected[Affected['Country/Region'].isin(first_selection)]

Affected = Affected.groupby(['ObservationDate', 'Country/Region']).agg({'Confirmed': ['sum']})

Affected.columns = ['Confirmed Cases']

Affected = Affected.reset_index()

Affected['Confirmed Cases']= np.log2(Affected['Confirmed Cases'])

fig = px.line(Affected, x="ObservationDate", y="Confirmed Cases", color="Country/Region",

              line_group="Country/Region", hover_name="Country/Region")

fig.update_layout(template='plotly_dark')

fig.update_yaxes(title_text="Log 2 Confirmed Cases")

fig.show()
first_selection=['Germany','Finland', 'France',

         'Spain', 'UK', 'Greece','Czech Republic', 'Switzerland', 'Iceland', 'Romania']

from_date = datetime.date(2020,3,1)

last_date = datetime.date(2020,3,21)

EU_slope = slope(first_selection,from_date, last_date)

print(f'For the period from {from_date} to {last_date}')

print(f'Confirmed cases double every {EU_slope} days for the {first_selection}')

print(f'This is the  mean value calculated for the set of countries.')
Affected_2 = dataset.copy()

second_list=['Portugal',

          'Slovenia', 'Australia', 'Italy', 'Japan','China',

         'Germany', 'Poland', 'Ireland', 'Estonia', 'Luxembourg', 'South Korea', 'US']



Affected_2 = Affected_2[Affected_2['Country/Region'].isin(second_list)]

Affected_2 = Affected_2.groupby(['ObservationDate', 'Country/Region']).agg({'Confirmed': ['sum']})

Affected_2.columns = ['Confirmed Cases']

Affected_2 = Affected_2.reset_index()



Affected_2['Confirmed Cases']= np.log2(Affected_2['Confirmed Cases'] + 0.1)







fig = px.line(Affected_2, x="ObservationDate", y="Confirmed Cases", color="Country/Region",

              line_group="Country/Region", hover_name="Country/Region")

fig.update_layout(template='plotly_dark')

fig.update_yaxes(title_text="Log 2 Confirmed Cases")

fig.show()
from_date = datetime.date(2020,3,1)

last_date = datetime.date(2020,3,21)

US_slope = slope(['US'],from_date, last_date)

Australia_slope = slope(['Australia'],from_date, last_date)

Japan_slope = slope(['Japan'],from_date, last_date)

Italy_slope = slope(['Italy'], from_date, last_date)

Finland_slope = slope(['Finland'], from_date, last_date)

Germany_slope = slope(['Germany'], from_date, last_date)

print(f'For the period from {from_date} to {last_date}')

print('')

print(f'Confirmed cases double every {US_slope} days for US')

print(f'Confirmed cases double every {Australia_slope} days for Australia')

print(f'Confirmed cases double every {Japan_slope} days for Japan')

print('')

first_selection=['Germany','Finland', 'France',

         'Spain', 'UK', 'Greece','Czech Republic', 'Switzerland', 'Iceland', 'Romania']

EU_slope = slope(first_selection,from_date, last_date)

print(f'Confirmed cases double every {Italy_slope} days for Italy')

print(f'Confirmed cases double every {Finland_slope} days for Finland')

print(f'Confirmed cases double every {Germany_slope} days for Germany')

print('')

print(f'Confirmed cases double every {EU_slope} days for the {first_selection}')

print(f'This is the  mean value calculated for the set of countries.\n')

from_date = datetime.date(2020,3,6)

last_date = datetime.date(2020,3,21)

second_selection = ['Slovenia', 'Poland', 'Portugal', 'Luxembourg', 'Ireland', 'Estonia']

EU_slope_second = slope(second_selection,from_date, last_date)

print(f'For the period from {from_date} to {last_date}')

print('')

print(f'Confirmed cases double every {EU_slope_second} days for the {second_selection}')

print(f'This is the  mean value calculated for the set of countries.')
Affected_3 = dataset.copy()

third_list=['China',

         'Finland', 'Germany', 'South Korea', 'Norway', 'Sweden', 'Italy', 'US']



Affected_3 = Affected_3[Affected_3['Country/Region'].isin(third_list)]

Affected_3 = Affected_3.groupby(['ObservationDate', 'Country/Region']).agg({'Confirmed': ['sum']})

Affected_3.columns = ['Confirmed Cases']

Affected_3 = Affected_3.reset_index()





Affected_3['Confirmed Cases']= np.log2(Affected_3['Confirmed Cases'] + 0.2)







fig = px.line(Affected_3, x="ObservationDate", y="Confirmed Cases", color="Country/Region",

              line_group="Country/Region", hover_name="Country/Region")

fig.update_layout(template='plotly_dark')

fig.update_yaxes(title_text="Log 2 Confirmed Cases")

fig.show()