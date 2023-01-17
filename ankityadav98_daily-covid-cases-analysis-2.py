import numpy as np 

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import random

import matplotlib.colors as mcolors

%matplotlib inline

import plotly.express as px

import plotly.graph_objs as go

from plotly.offline import plot, iplot, init_notebook_mode

init_notebook_mode(connected=True)

import warnings

warnings.filterwarnings("ignore")
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
usa_data= pd.read_csv('/kaggle/input/covid19-assignment/07-12-2020.csv')

csse= pd.read_csv('/kaggle/input/covid19-assignment/csse_daily.csv')
csse.info()
csse.head()
csse.isna().sum()
plt.figure(figsize=(8,6))

sns.heatmap(csse.isna(),yticklabels=False,cbar=False,cmap='viridis');
#Null value for Case-Fatality_Ratio is filled as No. of Deaths/No. of Confirmed cases.

csse['Case-Fatality_Ratio']=csse.apply(lambda x: x['Deaths']/(x['Confirmed']+1) 

                                       if np.isnan(x['Case-Fatality_Ratio']) else x['Case-Fatality_Ratio'],axis=1)
#Null value for Active cases is filled as Active= Confirmed-Deaths-Recovered

csse['Active']= csse.apply(lambda x: np.abs(x['Confirmed']-x['Deaths']-x['Recovered']) 

                                       if np.isnan(x['Active']) else x['Active'],axis=1)
csse['Incidence_Rate'].fillna(0,inplace=True)
csse.loc[csse['Province_State']=='Greenland','Country_Region']='Greenland'
csse.isna().sum()
country_wise=csse.groupby(['Country_Region'])['Confirmed','Deaths','Recovered','Active'].sum().reset_index()

country_wise.head()
confirm_color= 'deepskyblue'

death_color='red'

recovered_color='limegreen'

active_color='grey'
def plot_hbar(df, col,n,color,hover_data=[]):

    fig = px.bar(df.sort_values(col).tail(n), x=col, y="Country_Region",

                 text=col, orientation='h', width=700, hover_data=hover_data,color_discrete_sequence=[color])

    fig.update_layout(title='Total '+col+' Cases in top {} countries'.format(n), xaxis_title="No. of cases", yaxis_title="Country Name",

                      yaxis_categoryorder = 'total ascending')

    fig.show()
plot_hbar(country_wise,'Confirmed',15,confirm_color)
plot_hbar(country_wise,'Deaths',15,death_color)
plot_hbar(country_wise,'Recovered',15,recovered_color)
plot_hbar(country_wise,'Active',15,active_color)
def plot_map(df, col):

    df = df[df[col]>0]

    fig = px.choropleth(df, locations="Country_Region", locationmode='country names', 

                        color=col, hover_name="Country_Region",

                        title=col+' Cases in the World', hover_data=[col], color_continuous_scale='Inferno_r')

    fig.show()
plot_map(country_wise, 'Confirmed')
plot_map(country_wise, 'Deaths')
plot_map(country_wise,'Recovered')
plot_map(country_wise,'Active')
india_cases= csse[csse['Country_Region']=='India'].reset_index(drop=True)

india_cases.drop(columns=['FIPS','Admin2','Last_Update','Lat','Long_','Combined_Key'],inplace=True)

india_cases
india_cases['Province_State'].unique()
india_cases['Province_State'].replace('Jammu and Kashmir','Jammu & Kashmir',inplace=True)
def india_map_plot(df,col,color):

    df_curr= df.reset_index(drop=True)

    fig = px.choropleth(df_curr,

                        geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",

                        featureidkey='properties.ST_NM',

                        locations=df['Province_State'],

                        color=col,

                        color_continuous_scale=color,

                        title= col+' Cases in India'

                       )

    fig.update_geos(fitbounds="locations", visible=False)

    fig.show()
india_map_plot(india_cases,'Confirmed','viridis_r')
india_map_plot(india_cases,'Deaths','Inferno_r')
india_map_plot(india_cases,'Recovered','magma_r')
def plot_hbar_India(df, col,n,color,hover_data=[]):

    fig = px.bar(df.sort_values(col).tail(n), x=col, y="Province_State",

                 text=col, orientation='h', width=700, hover_data=hover_data,color_discrete_sequence=[color])

    fig.update_layout(title='Top {} states- '.format(n)+col, xaxis_title=""+str(col), yaxis_title="State Name",

                      yaxis_categoryorder = 'total ascending')

    fig.show()
plot_hbar_India(india_cases,'Incidence_Rate',10,'skyblue')
plot_hbar_India(india_cases,'Case-Fatality_Ratio',10,'mediumvioletred')
plt.figure(figsize=(6,10))

sns.barplot(x='Incidence_Rate',y='Province_State',data=india_cases.sort_values('Incidence_Rate'));
usa_state_codes = {

    'District of Columbia' : 'dc','Mississippi': 'MS', 'Oklahoma': 'OK', 

    'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 

    'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 

    'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 

    'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 

    'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 

    'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 

    'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI',

    'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 

    'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 

    'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 

    'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 

    'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 

    'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 

    'Nevada': 'NV', 'Maine': 'ME'}
list_usa_states=list(usa_state_codes.keys())
usa_data.head()
temp_df = pd.DataFrame(data=list_usa_states, columns=['State Names in list_usa_states'])

data_df = pd.DataFrame(usa_data['Province_State'].unique(), columns=['State Names in DATA'])

temp_df.merge(data_df,how='outer',left_on='State Names in list_usa_states',

             right_on='State Names in DATA')
#Creating a column with state codes of USA which will be used for plotting Geographical plot of USA.



usa_data['State_Code'] = usa_data['Province_State'].apply(lambda s: usa_state_codes[s] 

                                                              if s in usa_state_codes.keys() else 'UNKNOWN')
usa_data.drop(usa_data[usa_data['State_Code']=='UNKNOWN'].index,axis=0,inplace=True)
#Null values heatmap (yellow color shows null value in particular column).

plt.figure(figsize=(8,6))

sns.heatmap(usa_data.isna(),cmap='viridis',cbar=False,yticklabels=False);
#filling null value with 0.

usa_data['Recovered'].fillna(0,inplace=True)
def usa_map_plot(df,col,color):

    data = [dict(type="choropleth",

                 colorscale= color,

                 locations = df['State_Code'],

                 z = df[col],

                 locationmode = 'USA-states',

                 text = df['Province_State'],

                 marker = dict(line = dict (color = 'black',width = 1)),

                 colorbar = dict(title = 'No. of cases')

                )]



    lyt = dict(title = str(col)+' Cases in USA',

    geo=dict(scope="usa",showlakes = True,lakecolor = 'rgb(255, 255, 255)'))



    iplot(go.Figure(data=data,layout=lyt),validate=False)
usa_map_plot(usa_data,'Confirmed','magma_r')
usa_map_plot(usa_data,'Deaths','RdBu_r')
usa_map_plot(usa_data,'Recovered','Cividis_r')
usa_map_plot(usa_data,'Active','Inferno_r')
#filling null values of People_Hospitalized with 0.

#filling null value of Hospitalization_Rate with formula of People_Hospitalized/Confirmed Cases.

usa_data['People_Hospitalized'].fillna(0,inplace=True)

usa_data['Hospitalization_Rate']= usa_data.apply(lambda x: (x['People_Hospitalized']/x['Confirmed']) 

                                                 if np.isnan(x['Hospitalization_Rate']) else x['Hospitalization_Rate'],axis=1)
def plot_hbar_usa(df, col,n,color,hover_data=[]):

    fig = px.bar(df.sort_values(col).tail(n), x=col, y="Province_State",

                 text=col, orientation='h', width=700, hover_data=hover_data,color_discrete_sequence=[color])

    fig.update_layout(title="Top {} States- ".format(n)+str(col), xaxis_title=""+str(col), yaxis_title="State Name",

                      yaxis_categoryorder = 'total ascending')

    fig.show()
plot_hbar_usa(usa_data,'Incident_Rate',10,'skyblue')
plot_hbar_usa(usa_data,'People_Tested',10,'yellow')
plot_hbar_usa(usa_data,'People_Hospitalized',10,'grey')
plot_hbar_usa(usa_data,'Mortality_Rate',10,'brown')
plot_hbar_usa(usa_data,'Testing_Rate',10,'green')
plt.figure(figsize=(6,15))

sns.barplot(x='People_Tested',y='Province_State',data=usa_data.sort_values('People_Tested'));
# Only show 10 countries with the most cases, the rest are grouped into the other category

def top10_country(col):

    sorted_country=country_wise.sort_values(col,ascending=False).reset_index(drop=True)

    unique_countries = [] 

    cases = []

    others = np.sum(sorted_country[col][10:])

    values=[]



    for i in range(len(sorted_country[col][:10])):

        unique_countries.append(sorted_country['Country_Region'].unique()[i])

        cases.append(sorted_country[col][i])



    unique_countries.append('Others')

    cases.append(others)

    values.append(unique_countries)

    values.append(cases)

    return values
def plot_pie_charts(x, y, title):

    c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(country_wise['Country_Region'].unique()))

    plt.figure(figsize=(8,8))

    plt.title(title, size=20)

    plt.pie(y, colors=c,autopct='%.2f%%',pctdistance=0.9)

    plt.legend(x, loc='best', bbox_to_anchor=(1.5,1),fontsize=15)

    plt.show()
countries_cases= top10_country('Confirmed')

plot_pie_charts(countries_cases[0],countries_cases[1],'Percentage of Confirmed Cases in the World')
countries_cases= top10_country('Deaths')

plot_pie_charts(countries_cases[0],countries_cases[1],'Percentage of Death Cases in tthe World')
countries_cases= top10_country('Recovered')

plot_pie_charts(countries_cases[0],countries_cases[1],'Percentage of Recovered Cases in the world')