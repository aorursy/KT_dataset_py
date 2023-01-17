import plotly.io as pio

pio.renderers.default = "kaggle"

!pip install plotly==4.6.0
!pip install "notebook>=5.3" "ipywidgets>=7.2"
!pip install ipywidgets==7.5.1

!jupyter nbextension enable --py --sys-prefix widgetsnbextension
!pip install ipyaggrid

!jupyter nbextension enable --py --sys-prefix ipyaggrid
#General imports

import pandas as pd

from IPython.core.display import display, HTML

# ipyaggrid

from ipyaggrid import Grid

import datetime

import numpy as np
# visualizations

import plotly

import plotly.figure_factory as ff

plotly.offline.init_notebook_mode(connected=True)

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objects as go

import plotly.express as px

#import chart_studio.plotly as py

from plotly.subplots import make_subplots

%matplotlib inline
# imports ipywidgets

import ipywidgets as widgets

from ipywidgets import interact, interact_manual

from ipywidgets import Layout

from ipywidgets import TwoByTwoLayout
#import itables.interactive

#from itables import show

#import itables.options as opt
#data = pd.read_csv('../input/german_data.csv',header=0)

covid19_data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',index_col = 0)

covid19_line_list_data = pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv',index_col=0)

time_series_covid19_confirmed= pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv',index_col=0)

time_series_covid19_deaths= pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv',index_col=0)

time_series_covid19_recovered= pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv',index_col=0)
covid19_data.head()
covid19_data = covid19_data.reset_index()
covid19_data.rename(columns={'Last Update':'LastUpdate'},inplace=True)

covid19_data.rename(columns={'Province/State':'Province_State'},inplace=True)

covid19_data.rename(columns={'Country/Region':'Country_Region'},inplace=True)
covid19_data['ObservationDate'] = pd.to_datetime(covid19_data['ObservationDate'])

covid19_data['LastUpdate'] = pd.to_datetime(covid19_data['LastUpdate'])
# confirm

covid19_data.dtypes
print('Covid19_data number of rows:', covid19_data.shape[0])

print('Covid19_data number of columns:',covid19_data.shape[1])
# missing values

missing = covid19_data.isnull().sum()

missing_perc = (100*(missing/len(covid19_data))).round(1)

missing_values = pd.DataFrame({'missing_values':missing, 'missing_values_%':missing_perc})

missing_values
# Timespan

print('Start observation date:', covid19_data.ObservationDate.min())

print('End observation Date:', covid19_data.ObservationDate.max())
covid19_data['ActiveCases'] = covid19_data['Confirmed'] - covid19_data['Recovered'] - covid19_data['Deaths']
columns_defs = [{'field':c} for c in covid19_data.columns]



grid_options = {'columnDefs': columns_defs,

               'enableSorting': True,

               'enableFilter' :True,

               'enableColResize': True,

               'enableRangeSelection':True,'enableValue': True,

                'statusBar': {

        'statusPanels': [

            { 'statusPanel': 'agTotalAndFilteredRowCountComponent', 'align': 'left' },

            { 'statusPanel': 'agTotalRowCountComponent', 'align': 'center' },

            { 'statusPanel': 'agFilteredRowCountComponent' },

            { 'statusPanel': 'agSelectedRowCountComponent' },

            { 'statusPanel': 'agAggregationComponent' }

        ]

    }

               }



buttons = [{'name':'Table 1. Covid19_data'}]



g = Grid(grid_data = covid19_data,

        theme = 'ag-theme-fresh',

        quick_filter = True,

        show_toggle_delete = False,

        show_toggle_edit = False,

        grid_options = grid_options,

        index = True,

        width=900,

        height=500,

        center = False,

        menu = {'buttons':buttons},

        )



g
covid19_data_no_province= covid19_data.copy()

covid19_data_no_province = covid19_data_no_province.drop(['Province_State','LastUpdate'],axis=1)

covid19_data_no_province = covid19_data_no_province.groupby(['Country_Region','ObservationDate']).sum().reset_index()

covid19_data_no_province = covid19_data_no_province.sort_values(by=['Country_Region', 'ObservationDate', 'Confirmed'])

# create column with new cases per day

covid19_data_no_province['NewCasesPerDay'] = covid19_data_no_province.groupby(['Country_Region'])['Confirmed'].diff().fillna(covid19_data_no_province['Confirmed'])
columns_defs = [{'field':c} for c in covid19_data_no_province.columns]



grid_options = {'columnDefs': columns_defs,

               'enableSorting': True,

               'enableFilter' :True,

               'enableColResize': True,

               'enableRangeSelection':True,'enableValue': True,

                'statusBar': {

        'statusPanels': [

            { 'statusPanel': 'agTotalAndFilteredRowCountComponent', 'align': 'left' },

            { 'statusPanel': 'agTotalRowCountComponent', 'align': 'center' },

            { 'statusPanel': 'agFilteredRowCountComponent' },

            { 'statusPanel': 'agSelectedRowCountComponent' },

            { 'statusPanel': 'agAggregationComponent' }

        ]

    }

               }



buttons = [

{'name':'Table2. Covid19_data excluding Provinces_State'}]



g = Grid(grid_data = covid19_data_no_province,

        theme = 'ag-theme-fresh',

        quick_filter = True,

        show_toggle_delete = False,

        show_toggle_edit = False,

        grid_options = grid_options,

        index = True,

        width=1200,

        height=500,

        center = False, 

        menu = {'buttons':buttons}

        )



g

covid19_data_grouped = covid19_data_no_province.groupby(['Country_Region'],as_index=False)['Confirmed','Deaths','Recovered','ActiveCases','NewCasesPerDay'].agg(lambda x:x.max())

covid19_data_grouped.nlargest(15,'Confirmed') # for the second plot



countries = ['All'] + sorted(covid19_data_grouped.Country_Region.unique().tolist())

columns_drop = widgets.Dropdown(options=countries,description='Select a Country',value='All',layout={'width': 'max-content'},style={'description_width': 'initial'})





#figure 1

df1 = covid19_data_grouped.copy()



x = df1['Country_Region']

trace1 = go.Bar(name='Confirmed',x=x,y=df1.Confirmed,text=df1.Confirmed,textposition='auto',marker_color='blue')

trace2 = go.Bar(name='Deaths',x=x,y=df1.Deaths,text=df1.Deaths,textposition='auto',marker_color='red')

trace3 = go.Bar(name='Recovered',x=x,y=df1.Recovered,text=df1.Recovered,textposition='auto',marker_color='green')

trace4 = go.Bar(name='ActiveCases',x=x,y=df1.ActiveCases,text=df1.ActiveCases,textposition='auto',marker_color='orange')

g = go.FigureWidget(data=[trace1, trace2,trace3, trace4],layout=go.Layout(title=dict(text='Confirmed, Deaths and Recovered cases per country'),barmode='group'))

g.update_layout(barmode='group',xaxis_type='category',template='simple_white')





# figure 2

top = covid19_data_grouped.nlargest(15,'Confirmed')

x2 = top['Country_Region']

trace1 = go.Bar(name='Confirmed',x=x2,y=top.Confirmed,text=top.Confirmed,textposition='auto',showlegend=False,marker_color='blue')

trace2 = go.Bar(name='Deaths',x=x2,y=top.Deaths,text=top.Deaths,textposition='auto',showlegend=False,marker_color='red')

trace3 = go.Bar(name='Recovered',x=x2,y=top.Recovered,text=top.Recovered,textposition='auto',showlegend=False,marker_color='green')

trace4 = go.Bar(name='ActiveCases',x=x2,y=top.ActiveCases,text=top.ActiveCases,textposition='auto',showlegend=False,marker_color='orange')



g2 = go.FigureWidget(data=[trace1, trace2,trace3, trace4],layout=go.Layout(title=dict(text='Top 15 countries most affected by Covid19'),barmode='group'))

g2.update_layout(barmode='group',xaxis_type='category',template='simple_white')

#trace1.text = top.Confirmed



def plot_country(change):   

    if columns_drop.value == 'All':

        df2 = df1  

    else:

        df2 = df1[df1['Country_Region']==columns_drop.value]

        

    x1 = df2['Country_Region']

    y1 = df2['Confirmed']

    y2 = df2['Deaths']

    y3 = df2['Recovered']

    y4 = df2['ActiveCases']

    

    with g.batch_update():

        g.data[0].x = x1

        g.data[1].x = x1

        g.data[2].x = x1

        g.data[3].x = x1

        g.data[0].y = y1

        g.data[1].y = y2

        g.data[2].y = y3

        g.data[3].y = y4

            

        g.layout.barmode = 'group'

        g.layout.xaxis.title = 'Country'

        g.layout.yaxis.title = 'Number of cases'

        

columns_drop.observe(plot_country, names="value")



widgets.VBox([columns_drop,g,g2])
covid19_data_grouped_date = covid19_data_no_province.groupby(['Country_Region','ObservationDate'],as_index=False)['Confirmed','Deaths','Recovered','ActiveCases','NewCasesPerDay'].agg(lambda x:x.sum())



countries = ['All'] + sorted(covid19_data_grouped_date.Country_Region.unique().tolist())

columns_drop = widgets.Dropdown(options=countries,description='Select a Country',value='All',layout={'width': 'max-content'},style={'description_width': 'initial'})





df3 = covid19_data_grouped_date.copy()



subplots = make_subplots(rows=3,cols=2,subplot_titles=['Confirmed cases','Deaths','Recovered','Active cases','New cases per 24h'],vertical_spacing=0.2,column_widths=[0.3, 0.3])



g3 = go.FigureWidget(subplots)

x = df3['ObservationDate']

g3.add_trace(go.Scatter(x=x,y=df3['Confirmed'],showlegend=False,mode='markers + lines',marker_color='blue'),row=1,col=1)

g3.add_trace(go.Scatter(x=x,y=df3['Deaths'],showlegend=False,mode='markers + lines',marker_color = 'red'),row=1,col=2)

g3.add_trace(go.Scatter(x=x,y=df3['Recovered'],showlegend=False,mode='markers + lines',marker_color='green'),row=2,col=1)

g3.add_trace(go.Scatter(x=x,y=df3['ActiveCases'],showlegend=False,mode='markers + lines',marker_color='orange'),row=2,col=2) 

g3.add_trace(go.Scatter(x=x,y=df3['NewCasesPerDay'],showlegend=False,mode='markers + lines',marker_color='magenta'),row=3,col=1)  

g3.update_layout(barmode='group',template='simple_white',height=800)





def time_series(change):

    if columns_drop.value == 'All':

        df4 = df3.groupby('ObservationDate').sum().reset_index()

    else:

        df4 = df3[df3['Country_Region']==columns_drop.value]

        

    x1 = df4['ObservationDate']

    y1 = df4['Confirmed']

    y2 = df4['Deaths']

    y3 = df4['Recovered']

    y4 = df4['ActiveCases']

    y5 = df4['NewCasesPerDay']

    

    with g3.batch_update():

        g3.data[0].x = x1

        g3.data[1].x = x1

        g3.data[2].x = x1

        g3.data[3].x = x1

        g3.data[4].x = x1

        g3.data[0].y = y1

        g3.data[1].y = y2

        g3.data[2].y = y3

        g3.data[3].y = y4

        g3.data[4].y = y5            

        g3.layout.barmode = 'group'

        g3.layout.xaxis.title = 'Date'

        g3.layout.yaxis.title = 'Number of cases'

        

        

columns_drop.observe(time_series, names="value")



widgets.VBox([columns_drop,g3])
covid19_data_provinces = covid19_data.copy()
# not all countries have provinces or states

covid19_data_provinces = covid19_data_provinces[covid19_data_provinces['Province_State'].notna()]
# calculate again new cases per 24 h

covid19_data_provinces = covid19_data_provinces.groupby(['Country_Region','Province_State','ObservationDate']).sum().reset_index()

# create column with new cases per day

covid19_data_provinces['NewCasesPerDay'] = covid19_data_provinces.groupby(['Country_Region','Province_State'])['Confirmed'].diff().fillna(covid19_data_provinces['Confirmed'])
columns_defs = [{'field':c} for c in covid19_data_provinces.columns]



grid_options = {'columnDefs': columns_defs,

               'enableSorting': True,

               'enableFilter' :True,

               'enableColResize': True,

               'enableRangeSelection':True,'enableValue': True,

                'statusBar': {

        'statusPanels': [

            { 'statusPanel': 'agTotalAndFilteredRowCountComponent', 'align': 'left' },

            { 'statusPanel': 'agTotalRowCountComponent', 'align': 'center' },

            { 'statusPanel': 'agFilteredRowCountComponent' },

            { 'statusPanel': 'agSelectedRowCountComponent' },

            { 'statusPanel': 'agAggregationComponent' }

        ]

    }

               }



buttons = [

{'name':'Table3. Covid19_data including Provinces_State and excluding countries without Province_State'}]



g = Grid(grid_data = covid19_data_provinces,

        theme = 'ag-theme-fresh',

        quick_filter = True,

        show_toggle_delete = False,

        show_toggle_edit = False,

        grid_options = grid_options,

        index = True,

        width=1200,

        height=500,

        center = False, 

        menu = {'buttons':buttons}

        )



display(g)


covid19_data_grouped_provinces = covid19_data_provinces.groupby(['Country_Region','Province_State'],as_index=False)['Confirmed','Deaths','Recovered','ActiveCases','NewCasesPerDay'].agg(lambda x:x.max())

countries = sorted(covid19_data_grouped.Country_Region.unique().tolist())

columns_drop = widgets.Dropdown(options=countries,description='Select a Country',layout={'width': 'max-content'},style={'description_width': 'initial'})



df5 = covid19_data_grouped_provinces.copy()



g4 = go.FigureWidget()

x = df5['Province_State']

g4.add_trace(go.Bar(name='Confirmed',x=x,y=df5.Confirmed,text=df1.Confirmed,textposition='auto',marker_color='blue'))

g4.add_trace(go.Bar(name='Deaths',x=x,y=df5.Deaths,text=df1.Deaths,textposition='auto',marker_color='red'))

g4.add_trace(go.Bar(name='Recovered',x=x,y=df5.Recovered,text=df1.Recovered,textposition='auto',marker_color='green'))

g4.add_trace(go.Bar(name='ActiveCases',x=x,y=df5.ActiveCases,text=df1.ActiveCases,textposition='auto',marker_color='orange'))

g4.update_layout(barmode='group',template='simple_white',height=500)



def plot_country2(change):

    df6 = df5[df5['Country_Region']==columns_drop.value]

        

    x1 = df6['Province_State']

    y1 = df6['Confirmed']

    y2 = df6['Deaths']

    y3 = df6['Recovered']

    y4 = df6['ActiveCases']

    

    with g4.batch_update():

        g4.data[0].x = x1

        g4.data[1].x = x1

        g4.data[2].x = x1

        g4.data[3].x = x1

        g4.data[0].y = y1

        g4.data[1].y = y2

        g4.data[2].y = y3

        g4.data[3].y = y4           

        g4.layout.barmode = 'group'

        g4.layout.xaxis.title = 'Province/State'

        g4.layout.yaxis.title = 'Number of cases'

        g4.layout.title = 'Number of cases per Province/State'

        

columns_drop.observe(plot_country2, names="value")



widgets.VBox([columns_drop,g4])
#covid19_line_list_data = pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv',index_col=0)
covid19_line_list_data.head(3)
# missing values

missing = covid19_line_list_data.isnull().sum()

missing_perc = (100*(missing/len(covid19_line_list_data))).round(1)

missing_values = pd.DataFrame({'missing_values':missing, 'missing_values_%':missing_perc})

missing_values
# drop cols with 100% of missing values.

covid19_line_list_data = covid19_line_list_data.drop(['Unnamed: 3','link','Unnamed: 21','Unnamed: 22','Unnamed: 23',

                                                     'Unnamed: 24','Unnamed: 25','Unnamed: 26'],axis=1)
columns_defs = [{'field':c} for c in covid19_line_list_data.columns]



grid_options = {'columnDefs': columns_defs,

               'enableSorting': True,

               'enableFilter' :True,

               'enableColResize': True,

               'enableRangeSelection':True,'enableValue': True,

                'statusBar': {

        'statusPanels': [

            { 'statusPanel': 'agTotalAndFilteredRowCountComponent', 'align': 'left' },

            { 'statusPanel': 'agTotalRowCountComponent', 'align': 'center' },

            { 'statusPanel': 'agFilteredRowCountComponent' },

            { 'statusPanel': 'agSelectedRowCountComponent' },

            { 'statusPanel': 'agAggregationComponent' }

        ]

    }

               }



g = Grid(grid_data = covid19_line_list_data,

        theme = 'ag-theme-fresh',

        quick_filter = True,

        show_toggle_delete = False,

        show_toggle_edit = False,

        grid_options = grid_options,

        index = True,

        width=1000,

        height=500,

        center = False, columns_fit='auto',

         

        )



g
covid19_line_list_data.dtypes
# change name columns

covid19_line_list_data.rename(columns={'reporting date':'ReportingDate'},inplace=True)

covid19_line_list_data.rename(columns={'visiting Wuhan':'visiting_Wuhan'},inplace=True)

covid19_line_list_data.rename(columns={'from Wuhan':'from_Wuhan'},inplace=True)
# change to datetime

covid19_line_list_data['ReportingDate'] = pd.to_datetime(covid19_line_list_data['ReportingDate'])

covid19_line_list_data['symptom_onset'] = pd.to_datetime(covid19_line_list_data['symptom_onset'])

covid19_line_list_data['hosp_visit_date'] = pd.to_datetime(covid19_line_list_data['hosp_visit_date'])

covid19_line_list_data['exposure_start'] = pd.to_datetime(covid19_line_list_data['exposure_start'])

covid19_line_list_data['exposure_end'] = pd.to_datetime(covid19_line_list_data['exposure_end'])
covid19_line_list_data['recovered'].value_counts()
filtered = covid19_line_list_data[['recovered','death']]

columns_defs = [{'field':c} for c in filtered.columns]



grid_options = {'columnDefs': columns_defs,

               'enableSorting': True,

               'enableFilter' :True,

               'enableColResize': True,

               'enableRangeSelection':True,'enableValue': True,

                'statusBar': {

        'statusPanels': [

            { 'statusPanel': 'agTotalAndFilteredRowCountComponent', 'align': 'left' },

            { 'statusPanel': 'agTotalRowCountComponent', 'align': 'center' },

            { 'statusPanel': 'agFilteredRowCountComponent' },

            { 'statusPanel': 'agSelectedRowCountComponent' },

            { 'statusPanel': 'agAggregationComponent' }

        ]

    }

               }



g = Grid(grid_data = filtered,

        theme = 'ag-theme-fresh',

        quick_filter = True,

        show_toggle_delete = False,

        show_toggle_edit = False,

        grid_options = grid_options,

        index = True,

        width=1000,

        height=500,

        center = False, 

        )



g
filtered = covid19_line_list_data[['ReportingDate','country','gender', 'age', 'symptom_onset','hosp_visit_date','visiting_Wuhan',

       'from_Wuhan','symptom']].reset_index()
filtered
filtered_grouped = filtered.groupby(['symptom','gender']).size().reset_index(name='counts')

filtered_grouped_male = filtered_grouped[filtered_grouped['gender']=='male']

filtered_grouped_female = filtered_grouped[filtered_grouped['gender']=='female']
columns_defs = [{'field':c} for c in filtered_grouped.columns]



grid_options = {'columnDefs': columns_defs,

               'enableSorting': True,

               'enableFilter' :True,

               'enableColResize': True,

               'enableRangeSelection':True,'enableValue': True,

                'statusBar': {

        'statusPanels': [

            { 'statusPanel': 'agTotalAndFilteredRowCountComponent', 'align': 'left' },

            { 'statusPanel': 'agTotalRowCountComponent', 'align': 'center' },

            { 'statusPanel': 'agFilteredRowCountComponent' },

            { 'statusPanel': 'agSelectedRowCountComponent' },

            { 'statusPanel': 'agAggregationComponent' }

        ]

    }

               }



g = Grid(grid_data = filtered_grouped,

        theme = 'ag-theme-fresh',

        quick_filter = True,

        show_toggle_delete = False,

        show_toggle_edit = False,

        grid_options = grid_options,

        index = True,

        width=800,

        height=500,

        center = False, 

        

        )



g
fig=go.Figure()

fig.add_trace(go.Bar(name='Male',x=filtered_grouped_male.symptom,y=filtered_grouped_male.counts,marker_color='lime'))

fig.add_trace(go.Bar(name='Female',x=filtered_grouped_female.symptom,y=filtered_grouped_female.counts,marker_color='magenta'))

fig.update_layout(barmode='group',template='simple_white',width=1000,height=900)

fig.update_xaxes(title_text='Symptoms')

fig.update_yaxes(title_text='Count')

fig.show()
print('Minimum age:',filtered['age'].min())

print("maximum age:", filtered['age'].max())
bins = [0,10,20,30,40,50,60,70,80,90,100]

filtered['age_binned'] = pd.cut(filtered['age'],bins)

filtered
filtered_grouped_age = filtered.groupby(['symptom','age_binned']).size().reset_index(name='counts')

fig=go.Figure()

fig = px.bar(filtered_grouped_age, x='symptom', y='counts',color='age_binned',width=1000,height=900,template='simple_white')



fig.show()