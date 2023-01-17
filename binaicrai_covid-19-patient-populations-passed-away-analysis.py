# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import warnings

warnings.simplefilter('ignore')

warnings.filterwarnings('ignore')



import seaborn as sns

import matplotlib as p

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.graph_objs as gobj

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)



import plotly.express as px       

import plotly.offline as py       

import plotly.graph_objects as go 

from plotly.subplots import make_subplots
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df





def import_data(file):

    """create a dataframe and optimize its memory usage"""

    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)

    df = reduce_mem_usage(df)

    return df
# To use for Mem Reduction: Limit issue possible when trying to groupby values

#pd.set_option('display.max_columns', 200)

#situation_report = import_data('/kaggle/input/uncover/UNCOVER/WHO/who-situation-reports-covid-19.csv')

#situation_report.head()
situation_report = pd.read_csv('/kaggle/input/uncover/UNCOVER/WHO/who-situation-reports-covid-19.csv')

situation_report.dtypes
situation_report['date'] = situation_report['reported_date'].astype('datetime64[ns]')

situation_report['month'] = situation_report['date'].dt.month

situation_report.head()
situation_report['confirmed_cases'] = situation_report['confirmed_cases'].str.replace(' ', '')

situation_report.confirmed_cases = situation_report.confirmed_cases.astype(float)

situation_report.dtypes
confirmed_by_country = situation_report.groupby('reporting_country_territory')['confirmed_cases'].sum().sort_values(ascending=True).to_frame(name = 'sum').reset_index()

grouped_by_country_nonzero = confirmed_by_country[confirmed_by_country['sum'] != 0]

plt.figure(figsize=(20,40))

plt.barh('reporting_country_territory', 'sum', data=grouped_by_country_nonzero)

plt.xlabel("confirmed_cases", size=15)

plt.ylabel("reporting_country_territory", size=15)

plt.tick_params(axis='x', rotation = 90, labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title("Total Confirmed Cases by Country", size=18)
jan = situation_report[situation_report['month'] == 1]

jan_grouped = jan.groupby('reported_date')['confirmed_cases'].sum().to_frame(name = 'sum')

jan_grouped = jan_grouped[jan_grouped['sum'] != 0].reset_index()

jan_grouped
feb = situation_report[(situation_report['month'] == 2)]

feb_grouped = feb.groupby('reported_date')['confirmed_cases'].sum().to_frame(name = 'sum')

feb_grouped = feb_grouped[feb_grouped['sum'] != 0].reset_index()

feb_grouped
mar = situation_report[(situation_report['month'] == 3)]

mar_grouped = mar.groupby('reported_date')['confirmed_cases'].sum().to_frame(name = 'sum')

mar_grouped = mar_grouped[mar_grouped['sum'] != 0].reset_index()

mar_grouped
apr = situation_report[(situation_report['month'] == 4)]

apr_grouped = apr.groupby('reported_date')['confirmed_cases'].sum().to_frame(name = 'sum')

apr_grouped = apr_grouped[apr_grouped['sum'] != 0].reset_index()

apr_grouped['log'] = np.log(apr_grouped['sum'])

apr_grouped
plt.figure(figsize=(15,8))

ax = sns.lineplot(x="reported_date", y="sum", data=jan_grouped, linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum", size=14)

plt.tick_params(axis='x', labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('January - confirmed cases', size = 14)

plt.show()
plt.figure(figsize=(15,8))

ax = sns.lineplot(x="reported_date", y="sum", data=feb_grouped, c = 'yellow', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum", size=14)

plt.tick_params(axis='x', rotation = 45,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('February - confirmed cases', size = 14)

plt.show()
plt.figure(figsize=(15,8))

ax = sns.lineplot(x="reported_date", y="sum", data=mar_grouped, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum", size=14)

plt.tick_params(axis='x', rotation = 45,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('March - confirmed cases', size = 14)

plt.show()
plt.figure(figsize=(15,8))

# datelist = ['2020-04-01', '2020-04-02', '2020-04-03', '2020-04-04', '2020-04-05']

ax = sns.lineplot(x= 'reported_date', y= 'log', data=apr_grouped.sort_values('reported_date'), c = 'green', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum", size=14)

plt.tick_params(axis='x', rotation = 45,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('April - confirmed cases', size = 14)

plt.show()
situation_report['new_confirmed_cases'] = situation_report['new_confirmed_cases'].fillna(0)

df = situation_report[(situation_report['confirmed_cases'] ==0) & (situation_report['new_confirmed_cases'] != 0)]

df
situation_report['new_confirmed_cases'] = situation_report['new_confirmed_cases'].fillna(0)

new_confirmed_by_country = situation_report.groupby('reporting_country_territory')['new_confirmed_cases'].sum().sort_values(ascending=True).to_frame(name = 'sum').reset_index()

grouped_by_country_nonzero = new_confirmed_by_country[new_confirmed_by_country['sum'] != 0]

plt.figure(figsize=(20,40))

plt.barh('reporting_country_territory', 'sum', data=grouped_by_country_nonzero)

plt.xlabel("new_confirmed_cases", size=15)

plt.ylabel("reporting_country_territory", size=15)

plt.tick_params(axis='x', rotation = 90, labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title("Total New Confirmed Cases by Country", size=18)
new_confirmed_by_date = situation_report.groupby('reported_date')['new_confirmed_cases'].sum().sort_values(ascending=True).to_frame(name = 'sum').reset_index()

grouped_by_date_nonzero = new_confirmed_by_date[new_confirmed_by_date['sum'] != 0]

plt.figure(figsize=(20,40))

plt.barh('reported_date', 'sum', data=grouped_by_date_nonzero)

plt.xlabel("new_confirmed_cases", size=15)

plt.ylabel("reported_date", size=15)

plt.tick_params(axis='x', rotation = 90, labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title("Total New Confirmed Cases by Reported Date", size=18)
situation_report[['total_deaths', 'new_total_deaths']] = situation_report[['total_deaths', 'new_total_deaths']].fillna(0)

total_deaths_by_country = situation_report.groupby('reporting_country_territory')['total_deaths'].sum().sort_values(ascending=True).to_frame(name = 'sum').reset_index()

grouped_by_country_nonzero = total_deaths_by_country[total_deaths_by_country['sum'] != 0]

plt.figure(figsize=(20,40))

plt.barh('reporting_country_territory', 'sum', data=grouped_by_country_nonzero)

plt.xlabel("total_deaths", size=15)

plt.ylabel("reporting_country_territory", size=15)

plt.tick_params(axis='x', rotation = 90, labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title("Total Deaths by Country", size=18)
data = dict(type = 'choropleth',

            locations = grouped_by_country_nonzero['reporting_country_territory'],

            locationmode = 'country names',

            autocolorscale = False,

            colorscale = 'Rainbow',

            text= grouped_by_country_nonzero['reporting_country_territory'],

            z= grouped_by_country_nonzero['sum'],

            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 1)),

            colorbar = {'title':'Total Deaths by Country','len':0.25,'lenmode':'fraction'})

layout = dict(geo = dict(scope='world'), width = 1500, height = 1000)



worldmap = gobj.Figure(data = [data],layout = layout)

iplot(worldmap)
py.init_notebook_mode(connected=True)



#GroupingBy the dataset for the map

formated_gdf = situation_report.groupby(['reported_date', 'reporting_country_territory']).sum()['total_deaths'].to_frame(name = 'sum').reset_index()

formated_gdf['sum'] = formated_gdf['sum'].fillna(0)

formated_gdf['reported_date'] = pd.to_datetime(formated_gdf['reported_date'])

formated_gdf['reported_date'] = formated_gdf['reported_date'].dt.strftime('%m/%d/%Y')



formated_gdf['log_confirmedCases'] = np.log(formated_gdf['sum'] + 2)



#Plotting the figure

fig = px.choropleth(formated_gdf, locations="reporting_country_territory", locationmode='country names', 

                     color="log_confirmedCases", hover_name="reporting_country_territory",projection="mercator",

                     animation_frame="reported_date",width=1000, height=800,

                     color_continuous_scale=px.colors.sequential.Viridis,

                     title='Total Deaths by Country')



#Showing the figure

fig.update(layout_coloraxis_showscale=True)

py.offline.iplot(fig)
next_list = ['United Kingdom', 'Netherlands', 'Germany', 'Belgium', 'Republic of Korea', 'Switzerland']

next_in_line = situation_report[situation_report['reporting_country_territory'].isin(next_list)]

next_in_line = next_in_line[next_in_line['total_deaths'] != 0]
uk = next_in_line[next_in_line['reporting_country_territory'] == next_list[0]].sort_values('reported_date')

ned = next_in_line[next_in_line['reporting_country_territory'] == next_list[1]].sort_values('reported_date')

ger = next_in_line[next_in_line['reporting_country_territory'] == next_list[2]].sort_values('reported_date')

bel = next_in_line[next_in_line['reporting_country_territory'] == next_list[3]].sort_values('reported_date')

korea = next_in_line[next_in_line['reporting_country_territory'] == next_list[4]].sort_values('reported_date')

swit = next_in_line[next_in_line['reporting_country_territory'] == next_list[5]].sort_values('reported_date')
plt.figure(figsize=(25,8))

plt.plot('reported_date','total_deaths', data=uk, c='blue')

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.legend(('United Kingdom', 'Netherlands', 'Germany', 'Belgium', 'Republic of Korea', 'Switzerland'))

plt.show()
plt.figure(figsize=(25,8))

plt.plot('reported_date','total_deaths', data=ned, c='red')

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

#plt.title('Netherlands')

plt.legend(('Netherlands', 'United Kingdom','Germany', 'Belgium', 'Republic of Korea', 'Switzerland'))

plt.show()
plt.figure(figsize=(25,8))

plt.plot('reported_date','total_deaths', data=ger, c='green')

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

#plt.title('Germany')

plt.legend(('Germany','Netherlands', 'United Kingdom', 'Belgium', 'Republic of Korea', 'Switzerland'))

plt.show()
plt.figure(figsize=(25,8))

plt.plot('reported_date','total_deaths', data=bel, c='purple')

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

#plt.title('Belgium')

plt.legend(('Belgium','Netherlands', 'United Kingdom','Germany',  'Republic of Korea', 'Switzerland'))

plt.show()
plt.figure(figsize=(25,8))

plt.plot('reported_date','total_deaths', data=korea, c='orange')

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

#plt.title('Republic of Korea')

plt.legend(('Republic of Korea', 'Netherlands', 'United Kingdom','Germany', 'Belgium', 'Switzerland'))

plt.show()
plt.figure(figsize=(25,8))

plt.plot('reported_date','total_deaths', data=swit, c='brown')

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

#plt.title('Switzerland')

plt.legend(('Switzerland','Belgium','Netherlands', 'United Kingdom','Germany',  'Republic of Korea'))

plt.show()
next_list = ['Turkey', 'Brazil', 'Sweden', 'Indonesia', 'Portugal', 'Japan']

next_in_line = situation_report[situation_report['reporting_country_territory'].isin(next_list)]

next_in_line = next_in_line[next_in_line['total_deaths'] != 0]
turk = next_in_line[next_in_line['reporting_country_territory'] == next_list[0]].sort_values('reported_date')

braz = next_in_line[next_in_line['reporting_country_territory'] == next_list[1]].sort_values('reported_date')

swed = next_in_line[next_in_line['reporting_country_territory'] == next_list[2]].sort_values('reported_date')

indo = next_in_line[next_in_line['reporting_country_territory'] == next_list[3]].sort_values('reported_date')

port = next_in_line[next_in_line['reporting_country_territory'] == next_list[4]].sort_values('reported_date')

japn = next_in_line[next_in_line['reporting_country_territory'] == next_list[5]].sort_values('reported_date')
plt.figure(figsize=(25,8))

plt.plot('reported_date','total_deaths', data=turk, c='blue')

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.legend(('Turkey', 'Brazil', 'Sweden', 'Indonesia', 'Portugal', 'Japan'))

plt.show()
plt.figure(figsize=(25,8))

plt.plot('reported_date','total_deaths', data=braz, c='red')

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.legend(('Brazil', 'Turkey', 'Sweden', 'Indonesia', 'Portugal', 'Japan'))

plt.show()
plt.figure(figsize=(25,8))

plt.plot('reported_date','total_deaths', data=swed, c='green')

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.legend(('Sweden','Turkey', 'Brazil',  'Indonesia', 'Portugal', 'Japan'))

plt.show()
plt.figure(figsize=(25,8))

plt.plot('reported_date','total_deaths', data=indo, c='purple')

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.legend(('Indonesia', 'Turkey', 'Brazil', 'Sweden', 'Portugal', 'Japan'))

plt.show()
plt.figure(figsize=(25,8))

plt.plot('reported_date','total_deaths', data=port, c='orange')

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.legend(('Portugal', 'Turkey', 'Brazil', 'Sweden', 'Indonesia', 'Japan'))

plt.show()
plt.figure(figsize=(25,8))

plt.plot('reported_date','total_deaths', data=japn, c='brown')

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.legend(('Japan', 'Turkey', 'Brazil', 'Sweden', 'Indonesia', 'Portugal'))

plt.show()
situation_report[['total_deaths', 'new_total_deaths']] = situation_report[['total_deaths', 'new_total_deaths']].fillna(0)

new_total_deaths_by_country = situation_report.groupby('reporting_country_territory')['new_total_deaths'].sum().sort_values(ascending=True).to_frame(name = 'sum').reset_index()

new_grouped_by_country_nonzero = new_total_deaths_by_country[new_total_deaths_by_country['sum'] != 0]

plt.figure(figsize=(20,40))

plt.barh('reporting_country_territory', 'sum', data=new_grouped_by_country_nonzero)

plt.xlabel("new_total_deaths", size=15)

plt.ylabel("reporting_country_territory", size=15)

plt.tick_params(axis='x', rotation = 90, labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title("New Total Deaths by Country", size=18)
data = dict(type = 'choropleth',

            locations = new_grouped_by_country_nonzero['reporting_country_territory'],

            locationmode = 'country names',

            autocolorscale = False,

            colorscale = 'Rainbow',

            text= new_grouped_by_country_nonzero['reporting_country_territory'],

            z= new_grouped_by_country_nonzero['sum'],

            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 1)),

            colorbar = {'title':'New Total Deaths by Country','len':0.25,'lenmode':'fraction'})

layout = dict(geo = dict(scope='world'), width = 1500, height = 1000)



worldmap = gobj.Figure(data = [data],layout = layout)

iplot(worldmap)
py.init_notebook_mode(connected=True)



#GroupingBy the dataset for the map

formated_gdf = situation_report.groupby(['reported_date', 'reporting_country_territory']).sum()['new_total_deaths'].to_frame(name = 'sum').reset_index()

formated_gdf['sum'] = formated_gdf['sum'].fillna(0)

formated_gdf['reported_date'] = pd.to_datetime(formated_gdf['reported_date'])

formated_gdf['reported_date'] = formated_gdf['reported_date'].dt.strftime('%m/%d/%Y')



formated_gdf['log_confirmedCases'] = np.log(formated_gdf['sum'] + 2)



#Plotting the figure

fig = px.choropleth(formated_gdf, locations="reporting_country_territory", locationmode='country names', 

                     color="log_confirmedCases", hover_name="reporting_country_territory",projection="mercator",

                     animation_frame="reported_date",width=1000, height=800,

                     color_continuous_scale=px.colors.sequential.Viridis,

                     title='New Total Deaths by Country')



#Showing the figure

fig.update(layout_coloraxis_showscale=True)

py.offline.iplot(fig)
new_grouped_by_country_nonzero = new_grouped_by_country_nonzero.sort_values('sum', ascending = False)

top7_newdeaths_list = new_grouped_by_country_nonzero.nlargest(7, ['sum']).reporting_country_territory.to_list()

top7_newdeaths = situation_report[situation_report['reporting_country_territory'].isin(top7_newdeaths_list)]

top7_newdeaths_list
italy = top7_newdeaths[top7_newdeaths['reporting_country_territory'] == top7_newdeaths_list[0]].sort_values('reported_date')

spain = top7_newdeaths[top7_newdeaths['reporting_country_territory'] == top7_newdeaths_list[1]].sort_values('reported_date')

usa = top7_newdeaths[top7_newdeaths['reporting_country_territory'] == top7_newdeaths_list[2]].sort_values('reported_date')

france = top7_newdeaths[top7_newdeaths['reporting_country_territory'] == top7_newdeaths_list[3]].sort_values('reported_date')

iran = top7_newdeaths[top7_newdeaths['reporting_country_territory'] == top7_newdeaths_list[4]].sort_values('reported_date')

china = top7_newdeaths[top7_newdeaths['reporting_country_territory'] == top7_newdeaths_list[5]].sort_values('reported_date')

uk = top7_newdeaths[top7_newdeaths['reporting_country_territory'] == top7_newdeaths_list[6]].sort_values('reported_date')
# Grouping if multiple entries occuring for single date

italy = italy.groupby('reported_date').sum()['new_total_deaths'].to_frame(name = 'sum')

italy = italy.sort_values('reported_date').reset_index()

plt.figure(figsize=(25,8))

plt.plot('reported_date','sum', data=italy, c='blue')

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.legend(('Italy', ''))

plt.show()
spain = spain.groupby('reported_date').sum()['new_total_deaths'].to_frame(name = 'sum')

spain = spain.sort_values('reported_date').reset_index()

plt.figure(figsize=(25,8))

plt.plot('reported_date','sum', data=spain, c='red')

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.legend(('Spain', ''))

plt.show()
usa = usa.groupby('reported_date').sum()['new_total_deaths'].to_frame(name = 'sum')

usa = usa.sort_values('reported_date').reset_index()

plt.figure(figsize=(25,8))

plt.plot('reported_date','sum', data=usa, c='green')

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.legend(('USA', ''))

plt.show()
iran = iran.groupby('reported_date').sum()['new_total_deaths'].to_frame(name = 'sum')

iran = iran.sort_values('reported_date').reset_index()

plt.figure(figsize=(25,8))

plt.plot('reported_date','sum', data=iran, c='purple')

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.legend(('Iran', ''))

plt.show()
china = china.groupby('reported_date').sum()['new_total_deaths'].to_frame(name = 'sum')

china = china.sort_values('reported_date').reset_index()

plt.figure(figsize=(25,8))

plt.plot('reported_date','sum', data=china, c='orange')

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.legend(('China', ''))

plt.show()
uk = uk.groupby('reported_date').sum()['new_total_deaths'].to_frame(name = 'sum')

uk = uk.sort_values('reported_date').reset_index()

plt.figure(figsize=(25,8))

plt.plot('reported_date','sum', data=uk, c='brown')

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.legend(('UK', ''))

plt.show()
situation_report.transmission_classification = situation_report.transmission_classification.replace({'Local transmission': 'Local Transmission'})

situation_report.transmission_classification = situation_report.transmission_classification.fillna('Unknown')
sns.catplot('transmission_classification', data= situation_report, kind='count', alpha=0.7, height=4, aspect= 3)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = situation_report.transmission_classification.value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of Transmission Types', fontsize = 20, color = 'black')

plt.show()
deathgrouped_by_transmission = situation_report.groupby('transmission_classification').sum()['total_deaths'].to_frame(name = 'sum')

deathgrouped_by_transmission = deathgrouped_by_transmission.sort_values('sum', ascending = False).reset_index()

deathgrouped_by_transmission
# Places where the 11 deaths occurred are Under investigation

df = situation_report[(situation_report['transmission_classification'] == 'Under investigation') & (situation_report['total_deaths'] != 0) ]

df
df_ico = situation_report[(situation_report['transmission_classification'] == 'Imported cases only') & (situation_report['total_deaths'] != 0) ]

df_ico.head()
df_ico_by_country = df_ico.groupby('reporting_country_territory')['total_deaths'].sum().sort_values(ascending=True).to_frame(name = 'sum').reset_index()

data = dict(type = 'choropleth',

            locations = df_ico_by_country['reporting_country_territory'],

            locationmode = 'country names',

            autocolorscale = False,

            colorscale = 'Rainbow',

            text= df_ico_by_country['reporting_country_territory'],

            z= df_ico_by_country['sum'],

            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 1)),

            colorbar = {'title':'Total Deaths by Country/Territory for Imported Cases only','len':0.25,'lenmode':'fraction'})

layout = dict(geo = dict(scope='world'), width = 1500, height = 1000)



worldmap = gobj.Figure(data = [data],layout = layout)

iplot(worldmap)
df_unknown = situation_report[(situation_report['transmission_classification'] == 'Unknown') & (situation_report['total_deaths'] != 0) ]

df_unknown.head()
df_unknown_by_country = df_unknown.groupby('reporting_country_territory')['total_deaths'].sum().sort_values(ascending=True).to_frame(name = 'sum').reset_index()

data = dict(type = 'choropleth',

            locations = df_unknown_by_country['reporting_country_territory'],

            locationmode = 'country names',

            autocolorscale = False,

            colorscale = 'Rainbow',

            text= df_unknown_by_country['reporting_country_territory'],

            z= df_unknown_by_country['sum'],

            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 1)),

            colorbar = {'title':'Total Deaths by Country/Territory for Unknown','len':0.25,'lenmode':'fraction'})

layout = dict(geo = dict(scope='world'), width = 1500, height = 1000)



worldmap = gobj.Figure(data = [data],layout = layout)

iplot(worldmap)
df_lt = situation_report[(situation_report['transmission_classification'] == 'Local Transmission') & (situation_report['total_deaths'] != 0) ]

df_lt.head()
df_lt_by_country = df_lt.groupby('reporting_country_territory')['total_deaths'].sum().sort_values(ascending=True).to_frame(name = 'sum').reset_index()

data = dict(type = 'choropleth',

            locations = df_lt_by_country['reporting_country_territory'],

            locationmode = 'country names',

            autocolorscale = False,

            colorscale = 'Rainbow',

            text= df_lt_by_country['reporting_country_territory'],

            z= df_lt_by_country['sum'],

            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 1)),

            colorbar = {'title':'Total Deaths by Country/Territory for Unknown','len':0.25,'lenmode':'fraction'})

layout = dict(geo = dict(scope='world'), width = 1500, height = 1000)



worldmap = gobj.Figure(data = [data],layout = layout)

iplot(worldmap)