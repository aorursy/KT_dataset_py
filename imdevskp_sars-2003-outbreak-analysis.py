# numerical analysis

import numpy as np

# processing and storing in dataframe

import pandas as pd



# basic plotting

import matplotlib.pyplot as plt

# advanced plotting

import seaborn as sns

# interactive plotting

import plotly.express as px



# dealing with geographic data

import geopandas as gpd

# to get geolocation 

from geopandas.tools import geocode



# register the converters:

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
# plotly offline

import plotly

plotly.offline.init_notebook_mode(connected=True)
def plot_cal(title, start, end):

    '''Plot incubation period'''

    inc_day = []

    for i in range(1, 29):

        if(i>=start and i<=end):

            inc_day.append(1)

        else:

            inc_day.append(0)

            

    inc_day = np.array(inc_day)

    inc_day = inc_day.reshape(4, 7)

    

    fig, ax = plt.subplots(figsize=(6, 3))

    ax=sns.heatmap(inc_day, linewidths=2, square=True, 

                   xticklabels='', yticklabels='', ax=ax,

                   cbar=False, cmap=['whitesmoke','royalblue'])

    ax.set_title(title, fontdict={'fontsize':16}, loc='left', pad=15)

    

    plt.show()
def plot_barh(col):

    

    temp_df = latest.sort_values(col, ascending=False).head(10)

    temp_df = temp_df[temp_df[col]!=0]

    

    hover_data = ['Cumulative total cases', 'No. of deaths', 'Case fatalities ratio (%)']

    

    fig =  px.bar(temp_df, x=col, y='Country/Region', orientation='h', color='Country/Region', 

                  text=col, title=col, width=700, hover_data = hover_data,

                  color_discrete_sequence = px.colors.qualitative.Dark2)

    fig.update_traces(textposition='auto')

    fig.update_layout(xaxis_title="", yaxis_title="", showlegend=False,

                      uniformtext_minsize=8, uniformtext_mode='hide')

    fig.show()
def plot_pie(col1, col2, title, pal):



    temp = latest[[col1, col2]].sum()

    temp = pd.DataFrame(temp).reset_index()

    temp.columns = ['Column', 'Value']

    

    fig = px.sunburst(temp, path=['Column'], values='Value',

                      color_discrete_sequence=pal, title=title)

    fig.data[0].textinfo = 'label+text+value+percent root'

    fig.show()
# list of files

!ls -lt ../input/sars-outbreak-2003-complete-dataset
# importing daywise dataset

df = pd.read_csv('../input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv', parse_dates=['Date'])

# rename columns

df.columns = ['Date', 'Country', 'Cases', 'Deaths', 'Recovered']

# active no. of cases

df['Active'] = df['Cases'] - df['Deaths'] - df['Recovered']

# first few rows

df.head()
# day wise data

day_wise = df.groupby(['Date'])['Cases', 'Deaths', 'Recovered', 'Active'].sum()

# reset index

day_wise = day_wise.reset_index()

# first few rows

day_wise.head()
# importing summary dataset

latest = pd.read_csv('../input/sars-outbreak-2003-complete-dataset/summary_data_clean.csv')

# fix datetime columns

cols = ['Date onset first probable case', 'Date onset last probable case']

for col in cols:

    latest[col] = pd.to_datetime(latest[col])

cols = ['Number of Imported cases', 'Percentage of Imported cases']

for col in cols:

    latest[col] = latest[col].fillna(0)

# new columns

latest['Number of locally transmitted cases'] = latest['Cumulative total cases'] - latest['Number of Imported cases']

latest['Percentage of locally transmitted cases'] = 100 - latest['Percentage of Imported cases']

latest['Number of non HCW affected'] = latest['Cumulative total cases'] - latest['Number of HCW affected']

latest['Percentage of non HCW affected'] = 100 - latest['Percentage of HCW affected']

latest['Non fatality case (%)'] = 100 - latest['Case fatalities ratio (%)']

latest['No. of recovered'] = latest['Cumulative total cases'] - latest['No. of deaths']



# first few rows

latest.head()
# load countries map

world_map = gpd.read_file('../input/human-development-index-hdi/countries.geojson')

# avoid Antartica

world_map = world_map[world_map['name']!='Antarctica']

# select only important columns

world_map = world_map[['name', 'continent', 'geometry']]

# first few rows

world_map.head()
fig, ax = plt.subplots(figsize=(18, 14))

sns.set_style('whitegrid')



world_map.plot(ax=ax, color='white', edgecolor='black', alpha=0.4)

ax.set_title('First infected humans', 

             loc='left', fontdict={'fontsize': 24, 

                                   'fontfamily': 'monospace', 

                                   'fontweight': 'bold',

                                   'color': 'black'})



ax.scatter(113.7633, 23.3790, color='orangered', s=200, alpha=0.8)

ax.text(120.7633, 23.3790, 'Guandong, China \nin 2002', 

        fontfamily='monospace', fontsize=12, fontweight='bold',

        color='white', backgroundcolor='black')



ax.set_axis_off()
plot_cal('Incubation period of Ebola is from 2 to 7 days', 2, 7)
fig = px.choropleth(latest, locations="Country/Region", locationmode='country names',

                    color="Cumulative total cases", hover_name="Country/Region", 

                    color_continuous_scale="Sunset", 

                    title='Choropleth map cumulative no. of cases')

fig.update(layout_coloraxis_showscale=True)

fig.show()
plot_barh('Cumulative total cases')
plot_barh('No. of deaths')
plot_barh('Case fatalities ratio (%)')
plot_barh('Number of Imported cases')
plot_barh('Number of HCW affected')
plot_pie('No. of deaths', 'No. of recovered', 'CFR', ['lightseagreen', 'orangered'])
plot_pie('Cumulative male cases', 'Cumulative female cases', 'Gender wise', ['royalblue', 'crimson'])
plot_pie('Number of HCW affected', 'Number of non HCW affected', 'HCW', ['whitesmoke', 'dodgerblue'])
plot_pie('Number of Imported cases', 'Number of locally transmitted cases', 'Imported cases', ['slateblue', 'gold'])
latest = latest.sort_values('Date onset first probable case', ascending=False)



country = latest['Country/Region']

start = latest['Date onset first probable case'].to_numpy()

end = latest['Date onset last probable case'].to_numpy()



sns.set_style('whitegrid')

plt.figure(figsize=(15, 12))



plt.hlines(y=country, xmin=start, xmax=end, color='black', alpha=0.8)

plt.scatter(start, country, color='tomato', alpha=1, s=200, label='Date onset first probable case')

plt.scatter(end, country, color='black', alpha=1 , s=200, label='Date onset last probable case')



sns.despine(left=False, bottom=True)

plt.title('Onset Date', loc='left', fontsize=24)

plt.legend(ncol=2)

plt.show()
age = latest[['Country/Region', 'Median age', 'Age range']]

age = age.sort_values('Median age', ascending=False)

age = age.dropna(subset=['Median age'])        



age['Min age'] = age['Age range'].str.extract('(\d+)')

age['Max age'] = age['Age range'].str.extract('\-(\d+)')



for col in ['Median age', 'Min age', 'Max age']:

    age[col] = pd.to_numeric(age[col])



country = age['Country/Region']

median_age = pd.to_numeric(age['Median age'])

min_age = pd.to_numeric(age['Min age'])

max_age = pd.to_numeric(age['Max age'])



sns.set_style('whitegrid')

plt.figure(figsize=(15, 12))



plt.hlines(y=country, xmin=min_age, xmax=max_age, color='black', alpha=0.8)

plt.scatter(min_age, country, color='dimgray', alpha=1, s=200, label='Minimum Age')

plt.scatter(median_age, country, color='turquoise', alpha=1, s=200, label='Median age')

plt.scatter(max_age, country, color='black', alpha=1 , s=200, label='Maximum Age')



sns.despine(left=False, bottom=True)

plt.title('Age Range', loc='left', fontsize=24)

plt.legend(ncol=3)

plt.show()
def plot_daywise(col, hue):

    temp = day_wise[day_wise['Date'] > '2003-04-12']

    fig = px.area(temp, x="Date", y=col, width=700, 

                  color_discrete_sequence=[hue])

    fig.update_layout(title=col, xaxis_title="", yaxis_title="")

    fig.show()
def plot_stacked(col):

    temp = df[df['Date'] > '2003-04-12']

    fig = px.area(temp, x="Date", y=col, color='Country', 

                 height=600, title=col, 

                 color_discrete_sequence = px.colors.cyclical.mygbm)

    fig.update_layout(showlegend=True)

    fig.show()
plot_daywise('Cases', 'black')
plot_daywise('Deaths', 'orangered')
plot_daywise('Recovered', 'limegreen')
plot_daywise('Active', 'crimson')
plot_stacked('Cases')
plot_stacked('Deaths')
plot_stacked('Recovered')
plot_stacked('Active')
# temp = df.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()

# temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],

#                  var_name='Case', value_name='Count')

# temp.head()



# # fig = px.area(temp, x="Date", y="Count", color='Case', 

#               height=600, width=700, title='Cases over time', 

#               color_discrete_sequence = ['limegreen', 'crimson', 'black'])

# fig.update_layout(xaxis_rangeslider_visible=True)

# fig.show()
temp = latest[['Country/Region', 'Cumulative male cases', 'Cumulative female cases']]

temp['% Male'] = temp['Cumulative male cases']*100/(temp['Cumulative male cases']+temp['Cumulative female cases'])

temp['% Female'] = temp['Cumulative female cases']*100/(temp['Cumulative male cases']+temp['Cumulative female cases'])

temp.head()
temp1 = temp.melt(id_vars='Country/Region', 

                 value_vars=['Cumulative male cases', 'Cumulative female cases'], 

                 var_name='Case', value_name='Count')

temp1 = temp1.sort_values(['Count'], ascending=True)





fig = px.bar(temp1, x='Count', y='Country/Region', color='Case', 

             opacity=1, orientation='h', height=600,

             barmode='stack',

             color_discrete_sequence=['indigo', 'deeppink'])

fig.update_layout(title='No. of case based on Gender', xaxis_title="", yaxis_title="")

fig.show()
temp2 = temp.melt(id_vars='Country/Region', 

                 value_vars=['% Male', '% Female'], 

                 var_name='Case', value_name='Percentage')

temp2 = temp2.sort_values(['Percentage'], ascending=True)



fig = px.bar(temp2, x='Percentage', y='Country/Region', color='Case', 

             opacity=1, orientation='h', height=600,

             barmode='stack',

             color_discrete_sequence=['indigo', 'deeppink'])

fig.update_layout(title='No. of case based on Gender', xaxis_title="", yaxis_title="")

fig.show()
def plot_overlay_bar(col, hue):

    temp = latest[['Country/Region', 'Cumulative total cases', col]]

    temp = temp.melt(id_vars='Country/Region', 

                     value_vars=['Cumulative total cases', col], 

                     var_name='Case', value_name='Count')

    temp = temp.sort_values(['Case', 'Count'], ascending=True)

    

    fig = px.bar(temp, x='Count', y='Country/Region', color='Case', 

                 opacity=1, orientation='h', height=600,

                 barmode='overlay',

                 color_discrete_sequence=['black', hue])

    fig.update_layout(title=col, xaxis_title="", yaxis_title="")

    fig.show()
def plot_overlay_percent_bar(col1, col2, hue):

    temp = latest[['Country/Region', col1, col2]]

    temp = temp.melt(id_vars='Country/Region', 

                     value_vars=[col1, col2], 

                     var_name='Case', value_name='Percentage')

    temp = temp.sort_values(['Case', 'Percentage'], ascending=True)

    

    fig = px.bar(temp, x='Percentage', y='Country/Region', color='Case', 

                 opacity=1, orientation='h', height=600,

                 barmode='stack', range_x=[0,100],

                 color_discrete_sequence=[hue, 'black'])

    fig.update_layout(title=col1, xaxis_title="", yaxis_title="")

    fig.show()
plot_overlay_bar('No. of deaths', 'orangered')
plot_overlay_percent_bar('Case fatalities ratio (%)', 'Non fatality case (%)', 'orangered')
plot_overlay_bar('Number of Imported cases', 'gold')
plot_overlay_percent_bar('Percentage of Imported cases', 'Percentage of locally transmitted cases', 'gold')
plot_overlay_bar('Number of HCW affected', 'cornflowerblue')
plot_overlay_percent_bar('Percentage of HCW affected', 'Percentage of non HCW affected', 'cornflowerblue')