import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import plotly.express as px
import plotly.graph_objs as go
import statsmodels.api as sm
import math
import pycountry
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot#, init_notebook_mode(connected=True) 
#import dash            
#import dash_core_components as dcc
#import dash_html_components as html
#from dash.dependencies import Input, Output, State
#from dash.exceptions import PreventUpdate
%matplotlib inline
df_2015 = pd.read_csv("../input/world-happiness/2015.csv")
df_2016 = pd.read_csv("../input/world-happiness/2016.csv")
df_2017 = pd.read_csv("../input/world-happiness/2017.csv")
df_2018 = pd.read_csv("../input/world-happiness/2018.csv")
df_2019 = pd.read_csv("../input/world-happiness/2019.csv")
df_2015.head()
df_2015.columns
# Use Year 2015 columns in our analysis and insert another column for year. 

df_2015_temp = df_2015.filter(['Happiness Rank', 'Country', 'Region','Economy (GDP per Capita)','Family',
                        'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)','Generosity', 
                        'Happiness Score' ])
df_2015_temp.insert(0, "Year", 2015)
df_2016.head()
df_2016.columns
# Use Year 2016 columns in our analysis and insert another column for year. 

df_2016_temp = df_2016.filter(['Happiness Rank','Country', 'Region','Economy (GDP per Capita)','Family',
                        'Health (Life Expectancy)','Freedom', 'Trust (Government Corruption)', 'Generosity',
                        'Happiness Score'])
df_2016_temp.insert(0, "Year", 2016)
df_2017.head()
df_2017.columns
df_2017.rename(columns = {'Happiness.Rank':'Happiness Rank',
                          'Happiness.Score':'Happiness Score',
                          'Economy..GDP.per.Capita.' : 'Economy (GDP per Capita)',
                          'Health..Life.Expectancy.' : 'Health (Life Expectancy)',
                          'Trust..Government.Corruption.' : 'Trust (Government Corruption)',
                           }, inplace = True)
# Use Year 2017 columns in our analysis and insert another column for year. 

df_2017_temp = df_2017.filter(['Happiness Rank','Country','Economy (GDP per Capita)','Family',
                        'Health (Life Expectancy)','Freedom', 'Trust (Government Corruption)', 'Generosity',
                        'Happiness Score'])
df_2017_temp.insert(0, "Year", 2017)
df_2018.head()
df_2018.columns
df_2018.rename(columns = {'Overall rank' : 'Happiness Rank',
                          'Country or region' : 'Country',
                          'Score' : 'Happiness Score',
                          'Social support' : 'Family',
                          'GDP per capita' : 'Economy (GDP per Capita)',
                          'Healthy life expectancy' : 'Health (Life Expectancy)',
                          'Freedom to make life choices' :'Freedom',
                          'Perceptions of corruption' : 'Trust (Government Corruption)'
                          }, inplace = True)
# Use Year 2018 columns in our analysis and insert another column for year. 

df_2018_temp = df_2018.filter(['Happiness Rank','Country','Economy (GDP per Capita)','Family',
                        'Health (Life Expectancy)','Freedom', 'Trust (Government Corruption)', 'Generosity',
                        'Happiness Score'])
df_2018_temp.insert(0, "Year", 2018)
df_2019.head()
df_2019.rename(columns = {'Overall rank' : 'Happiness Rank',
                          'Country or region' : 'Country',
                          'Score' : 'Happiness Score',
                          'Social support' : 'Family',
                          'GDP per capita' : 'Economy (GDP per Capita)',
                          'Healthy life expectancy' : 'Health (Life Expectancy)',
                          'Freedom to make life choices' :'Freedom',
                          'Perceptions of corruption' : 'Trust (Government Corruption)'
                          }, inplace = True)
# Use Year 2019 columns in our analysis and insert another column for year. 

df_2019_temp = df_2019.filter(['Happiness Rank','Country','Economy (GDP per Capita)','Family',
                        'Health (Life Expectancy)','Freedom', 'Trust (Government Corruption)', 'Generosity',
                        'Happiness Score'])
df_2019_temp.insert(0, "Year", 2019)
df_2016_temp
print ('Numbers of rows and columns in year 2015 :', df_2015_temp.shape)
print ('Numbers of rows and columns in year 2016 :', df_2016_temp.shape)
print ('Numbers of rows and columns in year 2017 :', df_2017_temp.shape)
print ('Numbers of rows and columns in year 2018 :', df_2018_temp.shape)
print ('Numbers of rows and columns in year 2019 :', df_2019_temp.shape)
# Merge Data so that we can get Region in each missing database. This can be done using comparing two columns from 
# to different databases and merging it together and dropping all the columns. Run this once!

df_2017_temp = df_2017_temp.merge(df_2016_temp,left_on = 'Country', right_on = 'Country', how = 'inner')
df_2018_temp = df_2018_temp.merge(df_2016_temp,left_on = 'Country', right_on = 'Country', how = 'inner')
df_2019_temp = df_2019_temp.merge(df_2015_temp,left_on = 'Country', right_on = 'Country', how = 'inner')
print (df_2017_temp.columns)
print (df_2018_temp.columns)
print (df_2019_temp.columns)
df_2017_temp.drop(columns = ['Year_y', 'Happiness Rank_y','Economy (GDP per Capita)_y',
       'Family_y', 'Health (Life Expectancy)_y', 'Freedom_y',
       'Trust (Government Corruption)_y', 'Generosity_y', 'Happiness Score_y'], inplace = True)
df_2018_temp.drop(columns = ['Year_y','Happiness Rank_y','Economy (GDP per Capita)_y',
       'Family_y', 'Health (Life Expectancy)_y', 'Freedom_y','Year_y',
       'Trust (Government Corruption)_y', 'Generosity_y', 'Happiness Score_y'], inplace = True)
df_2019_temp.drop(columns = ['Year_y', 'Happiness Rank_y','Economy (GDP per Capita)_y',
       'Family_y', 'Health (Life Expectancy)_y', 'Freedom_y',
       'Trust (Government Corruption)_y', 'Generosity_y', 'Happiness Score_y'], inplace = True)
df_2017_temp.rename(columns = {'Year_x' : 'Year',
                          'Happiness Rank_x' : 'Happiness Rank',
                          'Happiness Score_x':'Happiness Score',
                          'Family_x':'Family',
                          'Economy (GDP per Capita)_x':'Economy (GDP per Capita)',
                          'Health (Life Expectancy)_x': 'Health (Life Expectancy)',
                          'Freedom_x' : 'Freedom',
                          'Trust (Government Corruption)_x': 'Trust (Government Corruption)',
                          'Generosity_x':'Generosity'
                          }, inplace = True)
df_2018_temp.rename(columns = {'Year_x' : 'Year',
                          'Happiness Rank_x' : 'Happiness Rank',
                          'Happiness Score_x':'Happiness Score',
                          'Family_x':'Family',
                          'Economy (GDP per Capita)_x':'Economy (GDP per Capita)',
                          'Health (Life Expectancy)_x': 'Health (Life Expectancy)',
                          'Freedom_x' : 'Freedom',
                          'Trust (Government Corruption)_x': 'Trust (Government Corruption)',
                          'Generosity_x':'Generosity'
                          }, inplace = True)
df_2019_temp.rename(columns = {'Year_x' : 'Year',
                          'Happiness Rank_x' : 'Happiness Rank',
                          'Happiness Score_x':'Happiness Score',
                          'Family_x':'Family',
                          'Economy (GDP per Capita)_x':'Economy (GDP per Capita)',
                          'Health (Life Expectancy)_x': 'Health (Life Expectancy)',
                          'Freedom_x' : 'Freedom',
                          'Trust (Government Corruption)_x': 'Trust (Government Corruption)',
                          'Generosity_x':'Generosity'
                          }, inplace = True)
df_2015_temp = df_2015.filter(['Year','Country','Region', 'Economy (GDP per Capita)','Family',
                        'Health (Life Expectancy)','Freedom', 'Trust (Government Corruption)', 'Generosity', 
                        'Happiness Score', 'Happiness Rank'])
df_2016_temp = df_2016.filter(['Year','Country','Region', 'Economy (GDP per Capita)','Family',
                        'Health (Life Expectancy)','Freedom', 'Trust (Government Corruption)', 'Generosity', 
                        'Happiness Score', 'Happiness Rank'])
df_2017_temp = df_2017_temp.filter(['Year','Country','Region', 'Economy (GDP per Capita)','Family',
                        'Health (Life Expectancy)','Freedom', 'Trust (Government Corruption)', 'Generosity', 
                        'Happiness Score', 'Happiness Rank'])
df_2018_temp = df_2018_temp.filter(['Year','Country','Region', 'Economy (GDP per Capita)','Family',
                        'Health (Life Expectancy)','Freedom', 'Trust (Government Corruption)', 'Generosity', 
                        'Happiness Score', 'Happiness Rank'])
df_2019_temp = df_2019_temp.filter(['Year','Country','Region', 'Economy (GDP per Capita)','Family',
                        'Health (Life Expectancy)','Freedom', 'Trust (Government Corruption)', 'Generosity', 
                        'Happiness Score', 'Happiness Rank'])
df_2016_temp
# Create dataset including all the data from all the years together.

df_final = pd.concat([df_2015_temp,df_2016_temp,df_2017_temp,df_2018_temp,df_2019_temp], 
                     sort = False, ignore_index=True)
df_final
df_final.columns
df_final.isna().sum()
#identify the NA value

df_final[df_final['Trust (Government Corruption)'].isna()]
df_final.info()
#Replace the NaN value with the mean of all values from each year for United Arab Emirates.

df_UAE = df_final[df_final['Country'] == 'United Arab Emirates']
df_UAE
df_UAE['Trust (Government Corruption)'].mean()  #find mean 
# repace the value to mean

df_final.fillna(0.311982, inplace=True)
corr=df_final.corr()
plt.figure(figsize=(30, 30))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.5)], vmax=3, linewidths=0.01,
            square=True,annot=True,cmap='GnBu',linecolor="green")
plt.title('Correlation between features')
sns.pairplot(df_final)
df_final
df_final.columns
df_final['Region'].value_counts()
#only for Southern Asia

df_region_southasia = df_final[df_final['Region'] == 'Southern Asia']

#df_region_southasia['Year'].value_counts()
fig = px.bar(data_frame=df_region_southasia,
            x = 'Year',
            y = 'Economy (GDP per Capita)',
            color = 'Country',
            opacity=0.5,
             
            title='Economy per Capita For Souther Asian Countries',
             
            barmode='group')
            #template='plotly_dark')
            #color_= ['grey', 'yellow','red', ''])
fig.show()
#Combining Southeastern and Eastern Regions from Asia

df_region_asia = df_final[df_final['Region'].isin(['Southeastern Asia', 'Eastern Asia'])]

df_region_asia.head()
fig = px.bar(data_frame=df_region_asia,
            x = 'Country',
            y = 'Economy (GDP per Capita)',
            color = 'Country',
            barmode='group',
            orientation= 'v',
            
            title='Economy per Capita For Eastern and Southern Eastern Asia Countries',
             
            animation_frame='Year', 
            range_y=[0,2],
             
            template='plotly_dark',
            text='Happiness Score'
            )
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside',
                 width = [0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]) # thickness of the bar
fig.show()
df_final
df_region_africa1 = df_final.groupby('Year')['Year','Country','Region', 'Economy (GDP per Capita)' ]
df_region_africa1 = df_final[df_final['Region'] == 'Sub-Saharan Africa']

df_region_africa1 = df_region_africa1.sort_values(['Happiness Score'], ascending=False)

#df_region_africa1 = df_region_africa1.groupby(['Year'])['Year', 'Country','Region', 'Economy (GDP per Capita)']
fig = px.bar(data_frame=df_region_africa1,
            x = 'Year',
            y = 'Economy (GDP per Capita)',
            color = 'Country',
            barmode='group',
            orientation= 'v',
             
            text='Happiness Score',            
           
            labels={'Economy (GDP per Capita)':'Economy',
                    'Year':'Year'},           
            title='Economy per Capita For Sub - African Countries', 
            )

fig.update_traces(texttemplate='%{text:.2s}')#, textposition='outside')

fig.show()
df_region_europe = df_final[df_final['Region'].isin(['Western Europe', 'Central and Eastern Europe'])]

#df_region_europe
fig = px.sunburst(
    data_frame = df_region_europe,
    path = ['Region', "Country",'Year',"Happiness Score"], values = ("Economy (GDP per Capita)"),
    color = "Country",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    maxdepth = 3,
    branchvalues='total',
    #title='Economy per Capita For European Countries'
)

fig.update_traces(textinfo='label+percent entry')
fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))


fig.show()
fig = px.sunburst(
    data_frame = df_final,
    path = ['Region', "Country",'Year',"Happiness Score"], 
    values = ("Economy (GDP per Capita)"),
    color = "Country",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    maxdepth = 3,
    branchvalues='remainder',
    labels = {"Economy (GDP per Capita)":"Economy",#},
                    "Year":"Year"}
)

fig.update_traces(textinfo='label')
fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))


fig.show()
df_final['Region'].value_counts()
df_region_southasia.shape
fig = px.scatter(df_region_southasia,
                x = 'Health (Life Expectancy)',
                y = 'Happiness Score',
                color = 'Country',
                size = 'Economy (GDP per Capita)', 
                facet_row = 'Year',
                labels ={"Happiness Score":"H Score"},
                #trendline = 'ols',
                title = 'Health vs Happiness Score for South Asia, with Bubble size indication of GDP',
                )

fig.show()

df_region_europe1 = df_region_europe[df_region_europe['Country'].isin(['Switzerland', 
                                                                       'Norway','Finland','Netherlands',
                                                                      'Sweden', 'Austria'])]
#df_region_europe1
#'Greece', 'Ukraine','Georgia', 'Bulgaria', 'Tajikistan','Armenia'])] 
fig = px.scatter(df_region_europe,
                x = 'Health (Life Expectancy)',
                y = 'Happiness Score',
                color = 'Country',
                size = 'Economy (GDP per Capita)', 
                #facet_row = 'Year',
                facet_col = 'Year',
                facet_col_wrap = 5,
                labels ={"Health (Life Expectancy)":"Health"},
                title = 'Health vs Happiness Score for Europe, with Bubble size indication of GDP',
                )
fig.show()
fig = px.scatter(df_region_europe1,
                x = 'Health (Life Expectancy)',
                y = 'Happiness Score',
                color = 'Country',
                size = 'Economy (GDP per Capita)', 
                #facet_row = 'Year',
                facet_col = 'Year',
                facet_col_wrap = 5,
                labels ={"Health (Life Expectancy)":"Health"},
                title = 'Health vs Happiness Score for Western European Countries, with Bubble size indicating of GDP',
                )
fig.show()
df_region_europe2 = df_final[df_final['Region']=='Central and Eastern Europe']

#df_region_europe2
fig = px.scatter(df_region_europe2,
                x = 'Health (Life Expectancy)',
                y = 'Happiness Score',
                color = 'Country',
                size = 'Economy (GDP per Capita)', 
                #facet_row = 'Year',
                facet_col = 'Year',
                facet_col_wrap = 5,
                labels ={"Health (Life Expectancy)":"Health"},
                title = 'Health vs Happiness Score for Central and Eastern Europe, with Bubble size indicating of GDP',
                )
fig.show()
df_final['Happiness Score'].median()
df_region_america = df_final[df_final['Region']=='North America']
#df_region_america
fig = px.scatter(df_region_america,
                x = 'Health (Life Expectancy)',
                y = 'Happiness Score',
                color = 'Country',
                #labels ={"Happiness Score":"H Score"},
                #facet_col = 'Year',
                #facet_col_wrap = 5,
                trendline = 'ols',
                title = 'Health vs Happiness Score for America, with Bubble size indication of GDP',
                marginal_x = 'rug',
                marginal_y = 'box'
                )

fig.show()
#custom selecting countries from parent database.

df_custom_countries = df_final[df_final['Country'].isin(['Switzerland', 'Norway',
                               'Austria', 'New Zealand',  'Israel', 'Bhutan',  'India'
                                'Bangladesh', 'Mauritius', 'Nigeria', 'Zambia', 'Czech Republic', 
                                'Uzbekistan', 'Slovakia','Canada', 'United States', 
                                'Poland', 'Turkmenistan', 'Costa Rica', 'Mexico', 'Brazil',
                                'Israel', 'United Arab Emirates', 
                                'Qatar', 'Saudi Arabia', 'Singapore', 'Thailand', 
                                'Philippines', 'Malaysia', 
                                'South Korea', 'Japan',  'China'
                                 ])]

#'Finland','Netherlands', 'Sweden','Australia','Pakistan','Oman','Taiwan','Hong Kong',
fig = px.scatter(df_custom_countries,
                x = 'Health (Life Expectancy)',
                y = 'Happiness Score',
                color = 'Country',
                size = 'Happiness Score', 
                #facet_row = 'Year',
                #labels ={"Happiness Score":"H Score"},
                #trendline = 'ols',
                title = 'Health vs Happiness Score, with Bubble size indication of GDP',
                animation_frame = 'Year',
                range_x = [-0.05,1.3], # define the x and y limit so that graph is not outbound.
                range_y = [2,9],
                category_orders={'Year': [2015,2016,2017,2018,2019]},
                )

fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000 #speed of change from one from to next frames.
fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500 #speed of change in graph.

fig.show()
df_region_southasia_2019 =  df_region_southasia[df_region_southasia['Year'] == 2019]
df_region_southasia_2019 
fig = px.pie(data_frame=df_region_southasia_2019,
             values='Family',
             names='Country',
             color='Country',                      
             title='2019 Family Support Index for Southasia Country',     
             template='seaborn',           
             width=800,                          
             height=600,                         
             hole=0.5,                          
            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
df_region_africa1_2019 =  df_region_africa1[df_region_africa1['Year'] == 2019]
fig = px.pie(data_frame=df_region_africa1_2019,
             values='Family',
             names='Country',
             color='Country',                    
             title='2019 Family Support Index for Sub-Sahara African Countries',     
             template='seaborn',            
             width=800,                       
             height=600,                         
             #hole=0.5,                           
            )

fig.show()
df_region_comb = df_final[df_final['Region'].isin(['North America', 'Australia and New Zealand'])]
#df_region_comb
fig = px.pie(data_frame=df_region_comb,
             values='Family',
             names='Country',
             color='Country',                      
             title='Family Index for America, Australia and New Zealand Countries for Years 2015 - 2019',     
             template='seaborn',            
             width=800,                          
             height=600,                         
             hole=0.5,                           
            )
fig.update_traces(textposition='inside', textinfo='percent+label'),
#                         marker=dict(line=dict(color='#000000', width=4)),
#                         pull=[0, 0, 0.2, 0], opacity=0.7, rotation=180)

fig.show()
df_region_eastasia = df_final[df_final['Region'] == 'Eastern Asia']
df_region_eastasia_2019 = df_region_eastasia[df_region_eastasia['Year'] == 2019]
df_region_eastasia_2019
fig = px.pie(data_frame=df_region_eastasia_2019,
             values='Family',
             names='Country',
             color='Country',                      
             hover_name='Happiness Score',             
             #hover_data=['Happiness Score'],    
             title='2019 Family Support Index for East Asian Countries wiht Happiness Score.',     
             template='seaborn',                                         
             width=800,                          
             height=600,                         
             hole=0.5,                           
            )

fig.update_traces(textposition='outside', textinfo='percent+label',
                         marker=dict(line=dict(color='#000000', width=4)),
                         pull=[0.2, 0, 0, 0, 0, 0.2], opacity=0.7, rotation=15)



fig.show()
!pip install dash
import dash            
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([

    html.Div([
        dcc.Graph(id='the_graph')
    ]),

    html.Div([
        dcc.Input(id='input_state', type='number', inputMode='numeric', value=2015,
                  max=2015, min=2019, step=1, required=True),
        html.Button(id='submit_button', n_clicks=0, children='Submit'),
        html.Div(id='output_state'),
    ],style={'text-align': 'center'}),

])
@app.callback(
    [Output('output_state', 'children'),
    Output(component_id='the_graph', component_property='figure')],
    [Input(component_id='submit_button', component_property='n_clicks')],
    [State(component_id='input_state', component_property='value')]
)

def update_output(num_clicks, val_selected):
    if val_selected is None:
        raise PreventUpdate
    else:
        df_map = df_final1.query("Year=={}".format(val_selected))
        # print(df[:3])

        fig = px.choropleth(df_map, locations="Alpha-3 code",
                            color="Generosity",
                            hover_name="Country",
                            projection='natural earth',
                            title='Generosity of Countries by Year',
                            color_continuous_scale=px.colors.sequential.Plasma)

        fig.update_layout(title=dict(font=dict(size=28),x=0.5,xanchor='center'),
                          margin=dict(l=60, r=60, t=50, b=50))

        return ('The input value was "{}" and the button has been \
                clicked {} times'.format(val_selected, num_clicks), fig)

if __name__ == '__main__':
    app.run_server(debug=True, ssl_context='adhoc')