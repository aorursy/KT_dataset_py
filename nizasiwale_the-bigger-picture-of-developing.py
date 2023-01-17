#Firstly, lets import the libraries we will be using
import math
import networkx as nx
import matplotlib.pyplot as plt
from plotly.graph_objs import *
import pyproj as proj
import numpy as np
import pandas as pd
import plotly as py
import sys
import pip
from plotly.graph_objs import Scatter, Layout
import plotly.graph_objs as go
from IPython.display import display
import plotly.figure_factory as ff
survey_results_public = pd.read_csv("../input/survey_results_public.csv")
survey_results_schema = pd.read_csv("../input/survey_results_schema.csv")
survey_results_public.head()
#Create new dataframe by copying the Respondent and Country Columns
usersfrom = survey_results_public[['Respondent','Country']].copy()
#Add a new column to hold Number of respondents from each country and delete duplicates
usersfrom['Number Of Users from country']  = usersfrom['Country'].map(usersfrom["Country"].value_counts())
usersfrom = usersfrom.drop_duplicates(subset='Country', keep="last")
usersfrom.tail()
limits = [(0,499),(500,999),(1000,4999),(5000,9999),(10000,25309)]
colors = ["#002F32","#42826C","#A5C77F","#FFC861","#C84663"]
countries = []
scale = 10

for i in range(len(limits)):
    lim = limits[i]
    country_sub = usersfrom.loc[((usersfrom['Number Of Users from country']>lim[0]) &(usersfrom['Number Of Users from country']<lim[1]))]
    country = dict(
        type = 'scattergeo',
        locationmode = 'country names',
        locations = country_sub['Country'],
        text = None,
        marker = dict(
            size = country_sub['Number Of Users from country']/scale,
            color = colors[i],
            line = dict(width=0.5, color='#fafafa'),
            sizemode = 'area'
        ),
        name = '{0} - {1}'.format(lim[0],lim[1])
    )
    countries.append(country)

layout = dict(
        title = 'Repondents Location Map',
        showlegend = True,
        geo = dict(
            projection=dict( type='Mercator' ),
            showland = True,
            showocean = True,
            oceancolor = 'rgb(107,201,213)',
            landcolor = '#FFFDE7',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(238, 236, 221)",
            countrycolor="rgb(238, 236, 221)"
        ),
    )

fig = dict( data=countries, layout=layout )
py.offline.init_notebook_mode(connected=True)
py.offline.iplot( fig, validate=False, filename='repondentsmap' )
#Create largestCountries dataframe
usersfrom = usersfrom.sort_values(by='Number Of Users from country', ascending=True)
largestCountries = usersfrom.tail(20)
data  = go.Data([
            go.Bar(
              y = largestCountries['Country'],
              x = largestCountries['Number Of Users from country'],
              orientation='h'
        )])
layout = go.Layout(
        height = '1000',
        margin=go.Margin(l=300),
        title = "Country Respondents"
)
fig  = go.Figure(data=data, layout=layout)

py.offline.iplot( fig, validate=False, filename='Country Bar Chart' )

def splitDataFrameList(df,target_column,separator):
    ''' df = dataframe to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split
    returns: a dataframe with each entry for the target column separated, with each element moved into a new row. 
    The values in the other columns are duplicated across the newly divided rows.
    '''
    def splitListToRows(row,row_accumulator,target_column,separator):
        try:
            split_row = row[target_column].split(separator)
            for s in split_row:
                new_row = row.to_dict()
                new_row[target_column] = s
                row_accumulator.append(new_row)
        except:
            pass
            
    new_rows = []
    df.apply(splitListToRows,axis=1,args = (new_rows,target_column,separator))
    new_df = pd.DataFrame(new_rows)
    return new_df
#Create new DataFrame
LanguageWorkedWith = survey_results_public[['Respondent','LanguageWorkedWith']].copy()
LanguageWorkedWithTrimmed = LanguageWorkedWith.sample(500)
#Split LanguageWorkedWith columns for users who work with multiple languages and add those to new rows
formtedLanguageWorkedWithFrame = splitDataFrameList(LanguageWorkedWithTrimmed,'LanguageWorkedWith',";") 

formtedLanguageWorkedWithFrame.head()
plt.figure(figsize=(30, 23))

g = nx.from_pandas_dataframe(formtedLanguageWorkedWithFrame, source='Respondent', target='LanguageWorkedWith') 

layout = nx.spring_layout(g,k=.5,iterations=5)


allLanguagesWorkedWith = pd.unique(formtedLanguageWorkedWithFrame[['LanguageWorkedWith']].values.ravel('K')) 

nx.draw_networkx_edges(g, layout,width=0.05, edge_color='#AAAAAA')

Respondent = [node for node in g.nodes() if node in pd.unique(formtedLanguageWorkedWithFrame[['Respondent']].values.ravel('K'))]
nx.draw_networkx_nodes(g, layout, nodelist=Respondent, node_size=30, node_color='#AAAAAA',with_labels=True,label='Devloper Who Worked With One Languagea')

high_degree_people = [node for node in g.nodes() if node in pd.unique(formtedLanguageWorkedWithFrame[['Respondent']].values.ravel('K'))  and g.degree(node) > 1]
nx.draw_networkx_nodes(g, layout, nodelist=high_degree_people, node_size=30, node_color='#fc8d62',with_labels=True,label='Devloper Who Worked With Multiple Languagea')

AllLanguages = [node for node in g.nodes() if node in pd.unique(formtedLanguageWorkedWithFrame[['LanguageWorkedWith']].values.ravel('K')) ]
size = [g.degree(node) * 20 for node in g.nodes() if node in pd.unique(formtedLanguageWorkedWithFrame[['LanguageWorkedWith']].values.ravel('K'))]
nx.draw_networkx_nodes(g, layout, nodelist=AllLanguages, node_size=size, node_color='lightblue',with_labels=True,label='Computer Language')


allLanguagesWorkedWith_dict = dict(zip(allLanguagesWorkedWith, allLanguagesWorkedWith))
nx.draw_networkx_labels(g, layout, labels=allLanguagesWorkedWith_dict)

plt.axis('off')
lgnd = plt.legend( numpoints=1, fontsize=10)
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
lgnd.legendHandles[2]._sizes = [30]

plt.title("Language Worked With")

plt.show()
usersUsedLang =splitDataFrameList(LanguageWorkedWith,'LanguageWorkedWith',";") 
usersUsedLang['Percentage Of Developers Who Used Language'] = usersUsedLang.sort_values('Respondent').groupby(['LanguageWorkedWith'], sort=False).cumcount()+1
usersUsedLang['Percentage Of Developers Who Used Language']  = usersUsedLang['LanguageWorkedWith'].map(usersUsedLang["LanguageWorkedWith"].value_counts())
usersUsedLang = usersUsedLang.drop_duplicates(subset='LanguageWorkedWith', keep="last")
usersUsedLang = usersUsedLang[['LanguageWorkedWith','Percentage Of Developers Who Used Language']].copy()
usersUsedLang = usersUsedLang.sort_values(by='Percentage Of Developers Who Used Language', ascending=False)

TopCountries = pd.unique(usersUsedLang[['LanguageWorkedWith']].values.ravel('K')) 
Total = survey_results_public.shape[0]

for i in range(len(TopCountries)):
    ChartValue = (usersUsedLang.loc[((usersUsedLang['LanguageWorkedWith']==TopCountries[i]))].iloc[0]['Percentage Of Developers Who Used Language']/
                Total)*100
    usersUsedLang.iloc[i,1]=(math.ceil(ChartValue*100)/100)
    
table = ff.create_table(usersUsedLang)
py.offline.init_notebook_mode(connected=True)
py.offline.iplot( table, validate=False, filename='Year Coding Bar Chart' )
#Create new DataFrame
LanguageDesireNextYear = survey_results_public[['Respondent','LanguageDesireNextYear']].copy()
LanguageDesireNextYearTrimmed = LanguageDesireNextYear.sample(1000)
formtedLanguageDesireNextYear =splitDataFrameList(LanguageDesireNextYearTrimmed,'LanguageDesireNextYear',";") 
formtedLanguageDesireNextYear.head()
plt.figure(figsize=(40, 30))

# 1. Create the graph
g = nx.from_pandas_dataframe(formtedLanguageDesireNextYear, source='Respondent', target='LanguageDesireNextYear') 

# 2. Create a layout for our nodes 
layout = nx.spring_layout(g,k=0.8,iterations=5)


allLanguageDesireNextYear = pd.unique(formtedLanguageDesireNextYear[['LanguageDesireNextYear']].values.ravel('K')) 

nx.draw_networkx_edges(g, layout,width=0.03, edge_color='#AAAAAA')



Respondent = [node for node in g.nodes() if node in pd.unique(formtedLanguageDesireNextYear[['Respondent']].values.ravel('K'))]
nx.draw_networkx_nodes(g, layout, nodelist=Respondent, node_size=30, node_color='#AAAAAA',with_labels=True,label='Developers Who Only Desire One Language')

high_degree_people = [node for node in g.nodes() if node in pd.unique(formtedLanguageDesireNextYear[['Respondent']].values.ravel('K'))  and g.degree(node) > 1]
nx.draw_networkx_nodes(g, layout, nodelist=high_degree_people, node_size=30, node_color='#fc8d62',with_labels=True,label='Developers Who Only Desire Multiple Languages')

AllLanguages = [node for node in g.nodes() if node in pd.unique(formtedLanguageDesireNextYear[['LanguageDesireNextYear']].values.ravel('K')) ]
size = [g.degree(node) * 20 for node in g.nodes() if node in pd.unique(formtedLanguageDesireNextYear[['LanguageDesireNextYear']].values.ravel('K'))]
nx.draw_networkx_nodes(g, layout, nodelist=AllLanguages, node_size=size, node_color='lightblue',with_labels=True,label='Computer Language')


allLanguageDesireNextYear_dict = dict(zip(allLanguageDesireNextYear, allLanguageDesireNextYear))
nx.draw_networkx_labels(g, layout, labels=allLanguageDesireNextYear_dict)

# 4. Turn off the axis because I know you don't want it
plt.axis('off')

lgnd = plt.legend( numpoints=1, fontsize=10)
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
lgnd.legendHandles[2]._sizes = [30]

plt.title("Language Desired Next Year")

# 5. Tell matplotlib to show it
plt.show()
usersUsedLang =splitDataFrameList(LanguageDesireNextYear,'LanguageDesireNextYear',";") 
usersUsedLang['Percentage Of Developers Who Desire To Learn Language'] = usersUsedLang.sort_values('Respondent').groupby(['LanguageDesireNextYear'], sort=False).cumcount()+1
usersUsedLang['Percentage Of Developers Who Desire To Learn Language']  = usersUsedLang['LanguageDesireNextYear'].map(usersUsedLang["LanguageDesireNextYear"].value_counts())
usersUsedLang = usersUsedLang.drop_duplicates(subset='LanguageDesireNextYear', keep="last")
usersUsedLang = usersUsedLang[['LanguageDesireNextYear','Percentage Of Developers Who Desire To Learn Language']].copy()
usersUsedLang = usersUsedLang.sort_values(by='Percentage Of Developers Who Desire To Learn Language', ascending=False)

TopCountries = pd.unique(usersUsedLang[['LanguageDesireNextYear']].values.ravel('K')) 
Total = survey_results_public.shape[0]

for i in range(len(TopCountries)):
    ChartValue = (usersUsedLang.loc[((usersUsedLang['LanguageDesireNextYear']==TopCountries[i]))].iloc[0]['Percentage Of Developers Who Desire To Learn Language']/
                Total)*100
    usersUsedLang.iloc[i,1]=(math.ceil(ChartValue*100)/100)
    
table = ff.create_table(usersUsedLang)
py.offline.init_notebook_mode(connected=True)
py.offline.iplot( table, validate=False, filename='Year Coding Bar Chart' )
#Create new data frame
SalaryYearsCoding = survey_results_public[['Respondent','ConvertedSalary','YearsCoding','Country']].copy()
SalaryYearsCodingSub = SalaryYearsCoding.copy()
SalaryYearsCodingSub.head()
data = [go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='0-2 years','ConvertedSalary'], boxpoints=False,name='0-2 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='3-5 years','ConvertedSalary'], boxpoints=False,name='3-5 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='6-8 years','ConvertedSalary'], boxpoints=False,name='6-8 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='9-11 years','ConvertedSalary'], boxpoints=False,name='9-11 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='12-14 years','ConvertedSalary'], boxpoints=False,name='12-14 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='15-17 years','ConvertedSalary'], boxpoints=False,name='15-17 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='18-20 years','ConvertedSalary'], boxpoints=False,name='18-20 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='24-26 years','ConvertedSalary'], boxpoints=False,name='24-26 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='27-29 years','ConvertedSalary'], boxpoints=False,name='27-29 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='30 or more years','ConvertedSalary'], boxpoints=False,name='30 or more years')]

layout = go.Layout(title='Salaries By Experience',
yaxis=dict(title='Salaries USD'))

fig = go.Figure(data=data, layout=layout)

py.offline.init_notebook_mode(connected=True)
py.offline.iplot( fig, validate=False, filename='Year Coding Bar Chart' )


#Create new Dataframe for developing countries
TopDevelopingCountries = ['India','Russian Federation','Brazil','Ukraine','Pakistan','Romania','Iran, Islamic Republic of Iran','Mexico','Bangladesh','South Africa']
TopDevelopingCountriesData = survey_results_public.loc[(survey_results_public['Country'].isin(TopDevelopingCountries))]
SalaryYearsCoding = TopDevelopingCountriesData[['Respondent','ConvertedSalary','YearsCoding','Country']].copy()
SalaryYearsCodingSub = SalaryYearsCoding.copy()
SalaryYearsCodingSub.head()
data = [go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='0-2 years','ConvertedSalary'], boxpoints=False,name='0-2 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='3-5 years','ConvertedSalary'], boxpoints=False,name='3-5 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='6-8 years','ConvertedSalary'], boxpoints=False,name='6-8 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='9-11 years','ConvertedSalary'], boxpoints=False,name='9-11 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='12-14 years','ConvertedSalary'], boxpoints=False,name='12-14 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='15-17 years','ConvertedSalary'], boxpoints=False,name='15-17 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='18-20 years','ConvertedSalary'], boxpoints=False,name='18-20 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='24-26 years','ConvertedSalary'], boxpoints=False,name='24-26 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='27-29 years','ConvertedSalary'], boxpoints=False,name='27-29 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='30 or more years','ConvertedSalary'], boxpoints=False,name='30 or more years')]

layout = go.Layout(title='Salaries By Experience (Top Developing Countries)',
yaxis=dict(title='Salaries USD'))

fig = go.Figure(data=data, layout=layout)

py.offline.init_notebook_mode(connected=True)
py.offline.iplot( fig, validate=False, filename='Year Coding Bar Chart' )
TopDevelopedCountries = ['United States','United Kingdom','Canada','France','Poland','Australia','Netherlands','Spain','Italy','Sweden']
TopDevelopedCountriesData = survey_results_public.loc[(survey_results_public['Country'].isin(TopDevelopedCountries))]
SalaryYearsCoding = TopDevelopedCountriesData[['Respondent','ConvertedSalary','YearsCoding','Country']].copy()
SalaryYearsCodingSub = SalaryYearsCoding.copy()
SalaryYearsCodingSub.head()
data = [go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='0-2 years','ConvertedSalary'], boxpoints=False,name='0-2 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='3-5 years','ConvertedSalary'], boxpoints=False,name='3-5 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='6-8 years','ConvertedSalary'], boxpoints=False,name='6-8 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='9-11 years','ConvertedSalary'], boxpoints=False,name='9-11 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='12-14 years','ConvertedSalary'], boxpoints=False,name='12-14 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='15-17 years','ConvertedSalary'], boxpoints=False,name='15-17 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='18-20 years','ConvertedSalary'], boxpoints=False,name='18-20 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='24-26 years','ConvertedSalary'], boxpoints=False,name='24-26 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='27-29 years','ConvertedSalary'], boxpoints=False,name='27-29 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='30 or more years','ConvertedSalary'], boxpoints=False,name='30 or more years')]

layout = go.Layout(title='Salaries By Experience (Top Developed Countries)',
yaxis=dict(title='Salaries USD'))

fig = go.Figure(data=data, layout=layout)

py.offline.init_notebook_mode(connected=True)
py.offline.iplot( fig, validate=False, filename='Year Coding Bar Chart' )
WomenCountriesData = survey_results_public.loc[((survey_results_public['Gender']=='Female')&(survey_results_public['Country'].isin(TopDevelopedCountries)))]
SalaryYearsCoding = WomenCountriesData[['Respondent','ConvertedSalary','YearsCoding','Country']].copy()
SalaryYearsCodingSub = SalaryYearsCoding.copy()
SalaryYearsCodingSub.head()
data = [go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='0-2 years','ConvertedSalary'], boxpoints=False,name='0-2 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='3-5 years','ConvertedSalary'], boxpoints=False,name='3-5 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='6-8 years','ConvertedSalary'], boxpoints=False,name='6-8 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='9-11 years','ConvertedSalary'], boxpoints=False,name='9-11 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='12-14 years','ConvertedSalary'], boxpoints=False,name='12-14 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='15-17 years','ConvertedSalary'], boxpoints=False,name='15-17 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='18-20 years','ConvertedSalary'], boxpoints=False,name='18-20 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='24-26 years','ConvertedSalary'], boxpoints=False,name='24-26 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='27-29 years','ConvertedSalary'], boxpoints=False,name='27-29 years'),
go.Box(y=SalaryYearsCodingSub.loc[SalaryYearsCodingSub['YearsCoding']=='30 or more years','ConvertedSalary'], boxpoints=False,name='30 or more years')]

layout = go.Layout(title='Salaries By Experience (Women Developed Countries)',
yaxis=dict(title='Salaries USD'))

fig = go.Figure(data=data, layout=layout)

py.offline.init_notebook_mode(connected=True)

py.offline.iplot( fig, validate=False, filename='Year Coding Bar Chart' )
SanKeyChartValues = []
largestCountries = usersfrom.sort_values(by='Number Of Users from country', ascending=False)
TopCountries = pd.unique(largestCountries[['Country']].head(5).values.ravel('K')) 
YearsCoding = ["0-2 years", "3-5 years", "6-8 years"
             , "9-11 years", "12-14 years", "15-17 years", "18-20 years", "24-26 years", "27-29 years", "30 or more years"]
for i in range(len(TopCountries)):
    for k in range(len(YearsCoding)):
        ChartValue = (survey_results_public.loc[((survey_results_public['Country']==TopCountries[i])
                                         &(survey_results_public['YearsCoding']==YearsCoding[k]))].shape[0]/
                     survey_results_public.loc[((survey_results_public['Country']==TopCountries[i]))].shape[0])*100
        SanKeyChartValues.append(ChartValue)

trace1 = {
  "domain": {
    "x": [0, 1], 
    "y": [0, 1]
  }, 
  "link": {
    "color": [
              "#8A59888c"
              , "#449E9E8c"
              , "#D3D3D38c"
              , "#E91E638c"
              , "#9C27B08c"
              , "#3F51B58c"
              , "#00BCD48c"
              , "#7955488c"
              , "#FF98008c"
              , "#607D8B8c"
              
              ,"#8A59888c"
              , "#449E9E8c"
              , "#D3D3D38c"
              , "#E91E638c"
              , "#9C27B08c"
              , "#3F51B58c"
              , "#00BCD48c"
              , "#7955488c"
              , "#FF98008c"
              , "#607D8B8c"
              
              
              ,"#8A59888c"
              , "#449E9E8c"
              , "#D3D3D38c"
              , "#E91E638c"
              , "#9C27B08c"
              , "#3F51B58c"
              , "#00BCD48c"
              , "#7955488c"
              , "#FF98008c"
              , "#607D8B8c"
              
              
              ,"#8A59888c"
              , "#449E9E8c"
              , "#D3D3D38c"
              , "#E91E638c"
              , "#9C27B08c"
              , "#3F51B58c"
              , "#00BCD48c"
              , "#7955488c"
              , "#FF98008c"
              , "#607D8B8c"
              
              ,"#8A59888c"
              , "#449E9E8c"
              , "#D3D3D38c"
              , "#E91E638c"
              , "#9C27B08c"
              , "#3F51B58c"
              , "#00BCD48c"
              , "#7955488c"
              , "#FF98008c"
              , "#607D8B8c"], 
    "source": [0, 0, 0,0,0,0,0,0,0,0
               , 1, 1, 1,1,1,1,1,1,1,1
               , 2, 2, 2,2,2,2,2,2,2,2
               , 3, 3, 3,3,3,3,3,3,3,3
               , 4, 4, 4,4,4,4,4,4,4,4], 
    "target": [5, 6, 7,8,9,10,11,12,13,14
               ,5, 6, 7,8,9,10,11,12,13,14
               ,5, 6, 7,8,9,10,11,12,13,14
               , 5, 6, 7,8,9,10,11,12,13,14
               , 5, 6, 7,8,9,10,11,12,13,14], 
    "value": SanKeyChartValues
  }, 
  "node": {
    "color": ["#F27420", "#4994CE", "#FABC13", "#7FC241", "#D3D3D3", "#8A5988", "#449E9E", "#D3D3D3"
              , "#E91E63", "#9C27B0", "#3F51B5", "#00BCD4", "#795548", "#FF9800", "#607D8B"
              , None, None, None, None, None, None, None
             , None, None, None, None, None, None, None], 
    "label": ["United States", "India", "Germany", "United Kingdom", "Canada", "0-2 years", "3-5 years", "6-8 years"
             , "9-11 years", "12-14 years", "15-17 years", "18-20 years", "24-26 years", "27-29 years", "30 or more years"], 
    "line": {
      "color": "black", 
      "width": 0
    }, 
    "pad": 6, 
    "thickness": 18
  }, 
  "orientation": "h", 
  "type": "sankey", 
  "valueformat": ".0f"
}
data = Data([trace1])
layout = {
  "font": {"size": 10}, 
  "height": 744, 
  "title": "Percentage Exprience of The Top 5 Countries"
}
fig = Figure(data=data, layout=layout)
        
        


py.offline.iplot( fig, validate=False, filename='Sankey Salary' )
SanKeyChartValues = []

#TopCountries = pd.unique(TopDevelopingCountries[['Country']].head(5).values.ravel('K')) 
TopDevelopingCountriesMin = TopDevelopingCountries[:5]
TopCountries = TopDevelopingCountriesMin
#print(TopCountries)
YearsCoding = ["0-2 years", "3-5 years", "6-8 years"
             , "9-11 years", "12-14 years", "15-17 years", "18-20 years", "24-26 years", "27-29 years", "30 or more years"]
for i in range(len(TopCountries)):
    for k in range(len(YearsCoding)):
        ChartValue = (survey_results_public.loc[((survey_results_public['Country']==TopCountries[i])
                                         &(survey_results_public['YearsCoding']==YearsCoding[k]))].shape[0]/
                     survey_results_public.loc[((survey_results_public['Country']==TopCountries[i]))].shape[0])*100
        SanKeyChartValues.append(ChartValue)

trace1 = {
  "domain": {
    "x": [0, 1], 
    "y": [0, 1]
  }, 
  "link": {
    "color": [
              "#8A59888c"
              , "#449E9E8c"
              , "#D3D3D38c"
              , "#E91E638c"
              , "#9C27B08c"
              , "#3F51B58c"
              , "#00BCD48c"
              , "#7955488c"
              , "#FF98008c"
              , "#607D8B8c"
              
              ,"#8A59888c"
              , "#449E9E8c"
              , "#D3D3D38c"
              , "#E91E638c"
              , "#9C27B08c"
              , "#3F51B58c"
              , "#00BCD48c"
              , "#7955488c"
              , "#FF98008c"
              , "#607D8B8c"
              
              
              ,"#8A59888c"
              , "#449E9E8c"
              , "#D3D3D38c"
              , "#E91E638c"
              , "#9C27B08c"
              , "#3F51B58c"
              , "#00BCD48c"
              , "#7955488c"
              , "#FF98008c"
              , "#607D8B8c"
              
              
              ,"#8A59888c"
              , "#449E9E8c"
              , "#D3D3D38c"
              , "#E91E638c"
              , "#9C27B08c"
              , "#3F51B58c"
              , "#00BCD48c"
              , "#7955488c"
              , "#FF98008c"
              , "#607D8B8c"
              
              ,"#8A59888c"
              , "#449E9E8c"
              , "#D3D3D38c"
              , "#E91E638c"
              , "#9C27B08c"
              , "#3F51B58c"
              , "#00BCD48c"
              , "#7955488c"
              , "#FF98008c"
              , "#607D8B8c"], 
    "source": [0, 0, 0,0,0,0,0,0,0,0
               , 1, 1, 1,1,1,1,1,1,1,1
               , 2, 2, 2,2,2,2,2,2,2,2
               , 3, 3, 3,3,3,3,3,3,3,3
               , 4, 4, 4,4,4,4,4,4,4,4], 
    "target": [5, 6, 7,8,9,10,11,12,13,14
               ,5, 6, 7,8,9,10,11,12,13,14
               ,5, 6, 7,8,9,10,11,12,13,14
               , 5, 6, 7,8,9,10,11,12,13,14
               , 5, 6, 7,8,9,10,11,12,13,14], 
    "value": SanKeyChartValues
  }, 
  "node": {
    "color": ["#F27420", "#4994CE", "#FABC13", "#7FC241", "#D3D3D3", "#8A5988", "#449E9E", "#D3D3D3"
              , "#E91E63", "#9C27B0", "#3F51B5", "#00BCD4", "#795548", "#FF9800", "#607D8B"
              , None, None, None, None, None, None, None
             , None, None, None, None, None, None, None], 
    "label": [TopCountries[0], TopCountries[1], TopCountries[2], TopCountries[3], TopCountries[4], "0-2 years", "3-5 years", "6-8 years"
             , "9-11 years", "12-14 years", "15-17 years", "18-20 years", "24-26 years", "27-29 years", "30 or more years"], 
    "line": {
      "color": "black", 
      "width": 0
    }, 
    "pad": 6, 
    "thickness": 18
  }, 
  "orientation": "h", 
  "type": "sankey", 
  "valueformat": ".0f"
}
data = Data([trace1])
layout = {
  "font": {"size": 10}, 
  "height": 744, 
  "title": "Percentage Exprience of The Top 5 Developing Countries"
}
fig = Figure(data=data, layout=layout)
        
        


py.offline.iplot( fig, validate=False, filename='Sankey Salary' )
SanKeyChartValues = []
largestCountries = usersfrom.sort_values(by='Number Of Users from country', ascending=False)
TopCountries = pd.unique(largestCountries[['Country']].head(5).values.ravel('K')) 
YearsCoding = ["0-2 years", "3-5 years", "6-8 years"
             , "9-11 years", "12-14 years", "15-17 years", "18-20 years", "24-26 years", "27-29 years", "30 or more years"]
for i in range(len(TopCountries)):
    for k in range(len(YearsCoding)):
        ChartValue = (survey_results_public.loc[((survey_results_public['Country']==TopCountries[i])
                                         &(survey_results_public['YearsCoding']==YearsCoding[k])&(survey_results_public['Gender']=='Female'))].shape[0]/
                     survey_results_public.loc[((survey_results_public['Country']==TopCountries[i])&(survey_results_public['Gender']=='Female'))].shape[0])*100
        SanKeyChartValues.append(ChartValue)

trace1 = {
  "domain": {
    "x": [0, 1], 
    "y": [0, 1]
  }, 
  "link": {
    "color": [
              "#8A59888c"
              , "#449E9E8c"
              , "#D3D3D38c"
              , "#E91E638c"
              , "#9C27B08c"
              , "#3F51B58c"
              , "#00BCD48c"
              , "#7955488c"
              , "#FF98008c"
              , "#607D8B8c"
              
              ,"#8A59888c"
              , "#449E9E8c"
              , "#D3D3D38c"
              , "#E91E638c"
              , "#9C27B08c"
              , "#3F51B58c"
              , "#00BCD48c"
              , "#7955488c"
              , "#FF98008c"
              , "#607D8B8c"
              
              
              ,"#8A59888c"
              , "#449E9E8c"
              , "#D3D3D38c"
              , "#E91E638c"
              , "#9C27B08c"
              , "#3F51B58c"
              , "#00BCD48c"
              , "#7955488c"
              , "#FF98008c"
              , "#607D8B8c"
              
              
              ,"#8A59888c"
              , "#449E9E8c"
              , "#D3D3D38c"
              , "#E91E638c"
              , "#9C27B08c"
              , "#3F51B58c"
              , "#00BCD48c"
              , "#7955488c"
              , "#FF98008c"
              , "#607D8B8c"
              
              ,"#8A59888c"
              , "#449E9E8c"
              , "#D3D3D38c"
              , "#E91E638c"
              , "#9C27B08c"
              , "#3F51B58c"
              , "#00BCD48c"
              , "#7955488c"
              , "#FF98008c"
              , "#607D8B8c"], 
    "source": [0, 0, 0,0,0,0,0,0,0,0
               , 1, 1, 1,1,1,1,1,1,1,1
               , 2, 2, 2,2,2,2,2,2,2,2
               , 3, 3, 3,3,3,3,3,3,3,3
               , 4, 4, 4,4,4,4,4,4,4,4], 
    "target": [5, 6, 7,8,9,10,11,12,13,14
               ,5, 6, 7,8,9,10,11,12,13,14
               ,5, 6, 7,8,9,10,11,12,13,14
               , 5, 6, 7,8,9,10,11,12,13,14
               , 5, 6, 7,8,9,10,11,12,13,14], 
    "value": SanKeyChartValues
  }, 
  "node": {
    "color": ["#F27420", "#4994CE", "#FABC13", "#7FC241", "#D3D3D3", "#8A5988", "#449E9E", "#D3D3D3"
              , "#E91E63", "#9C27B0", "#3F51B5", "#00BCD4", "#795548", "#FF9800", "#607D8B"
              , None, None, None, None, None, None, None
             , None, None, None, None, None, None, None], 
    "label": ["United States", "India", "Germany", "United Kingdom", "Canada", "0-2 years", "3-5 years", "6-8 years"
             , "9-11 years", "12-14 years", "15-17 years", "18-20 years", "24-26 years", "27-29 years", "30 or more years"], 
    "line": {
      "color": "black", 
      "width": 0
    }, 
    "pad": 6, 
    "thickness": 18
  }, 
  "orientation": "h", 
  "type": "sankey", 
  "valueformat": ".0f"
}
data = Data([trace1])
layout = {
  "font": {"size": 10}, 
  "height": 744, 
  "title": "Percentage Exprience of The Top 5 Countries (Women)"
}
fig = Figure(data=data, layout=layout)
py.offline.iplot( fig, validate=False, filename='Sankey Salary' )
values = [
     survey_results_public.loc[ ((survey_results_public['FormalEducation']=='Bachelor’s degree (BA, BS, B.Eng., etc.)')
                                   | (survey_results_public['FormalEducation']=='Associate degree')
                                          | (survey_results_public['FormalEducation']=='Master’s degree (MA, MS, M.Eng., MBA, etc.)')
                                          | (survey_results_public['FormalEducation']=='Professional degree (JD, MD, etc.)'))].shape[0]
          
          
          ,survey_results_public.loc[(((survey_results_public['FormalEducation']=='Bachelor’s degree (BA, BS, B.Eng., etc.)')
                                   | (survey_results_public['FormalEducation']=='Associate degree')
                                          | (survey_results_public['FormalEducation']=='Master’s degree (MA, MS, M.Eng., MBA, etc.)')
                                          | (survey_results_public['FormalEducation']=='Professional degree (JD, MD, etc.)'))
                                     & ((survey_results_public['Employment']=='Employed part-time')
                                   | (survey_results_public['Employment']=='Employed full-time')))].shape[0]
          
          , survey_results_public.loc[(((survey_results_public['FormalEducation']=='Bachelor’s degree (BA, BS, B.Eng., etc.)')
                                   | (survey_results_public['FormalEducation']=='Associate degree')
                                          | (survey_results_public['FormalEducation']=='Master’s degree (MA, MS, M.Eng., MBA, etc.)')
                                          | (survey_results_public['FormalEducation']=='Professional degree (JD, MD, etc.)'))
                                     & ((survey_results_public['Employment']=='Employed part-time')
                                   | (survey_results_public['Employment']=='Employed full-time'))
                                      & ((survey_results_public['JobSatisfaction']=='Extremely satisfied')
                                   | (survey_results_public['JobSatisfaction']=='Moderately satisfied')
                                        | (survey_results_public['JobSatisfaction']=='Slightly satisfied')))].shape[0]
    
     , survey_results_public.loc[(((survey_results_public['FormalEducation']=='Bachelor’s degree (BA, BS, B.Eng., etc.)')
                                   | (survey_results_public['FormalEducation']=='Associate degree')
                                          | (survey_results_public['FormalEducation']=='Master’s degree (MA, MS, M.Eng., MBA, etc.)')
                                          | (survey_results_public['FormalEducation']=='Professional degree (JD, MD, etc.)'))
                                     & ((survey_results_public['Employment']=='Employed part-time')
                                   | (survey_results_public['Employment']=='Employed full-time'))
                                      & ((survey_results_public['JobSatisfaction']=='Extremely satisfied')
                                   | (survey_results_public['JobSatisfaction']=='Moderately satisfied')
                                        | (survey_results_public['JobSatisfaction']=='Slightly satisfied'))
                                 & ((survey_results_public['HopeFiveYears']=='Doing the same work')))].shape[0]
          ]
phases = [ 'Has a degree','Is Employed', 'Is Satisfied with Job','Wants to keep the same job']
colors = ['#4A849F', '#B4D6C6', '#F5F1D5', '#F2845C']
n_phase = len(phases)
plot_width = 400

# height of a section and difference between sections 
section_h = 100
section_d = 10

# multiplication factor to calculate the width of other sections
unit_width = plot_width / max(values)

# width of each funnel section relative to the plot width
phase_w = [int(value * unit_width) for value in values]

# plot height based on the number of sections and the gap in between them
height = section_h * n_phase + section_d * (n_phase - 1)
# list containing all the plot shapes
shapes = []

# list containing the Y-axis location for each section's name and value text
label_y = []

for i in range(n_phase):
        if (i == n_phase-1):
                points = [phase_w[i] / 2, height, phase_w[i] / 2, height - section_h]
        else:
                points = [phase_w[i] / 2, height, phase_w[i+1] / 2, height - section_h]

        path = 'M {0} {1} L {2} {3} L -{2} {3} L -{0} {1} Z'.format(*points)

        shape = {
                'type': 'path',
                'path': path,
                'fillcolor': colors[i],
                'line': {
                    'width': 1,
                    'color': colors[i]
                }
        }
        shapes.append(shape)
        
        # Y-axis location for this section's details (text)
        label_y.append(height - (section_h / 2))

        height = height - (section_h + section_d)
# For phase names
label_trace = go.Scatter(
    x=[-350]*n_phase,
    y=label_y,
    mode='text',
    text=phases,
    textposition='middle-right',
    textfont=dict(
        color='rgb(200,200,200)',
        size=12
    )
)
 
# For phase values
value_trace = go.Scatter(
    x=[350]*n_phase,
    y=label_y,
    mode='text',
    textposition='middle-left',
    text=values,
    textfont=dict(
        color='rgb(200,200,200)',
        size=12
    )
)

data = [label_trace, value_trace]
 
layout = go.Layout(
    title="Analysis of Developers Job Satisfaction",
    titlefont=dict(
        size=20,
        color='rgb(203,203,203)'
    ),
    shapes=shapes,
    height=560,
    width=800,
    showlegend=False,
    paper_bgcolor='#1B3440',
    plot_bgcolor='#1B3440',
    xaxis=dict(
        showticklabels=False,
        zeroline=False,
    ),
    yaxis=dict(
        showticklabels=False,
        zeroline=False
    )
)
 
fig = go.Figure(data=data, layout=layout)
py.offline.init_notebook_mode(connected=True)
py.offline.iplot( fig, validate=False, filename='map' )
