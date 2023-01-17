# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import io

import requests



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly import tools

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')



from datetime import tzinfo, timedelta, datetime, date



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Defaults

customFrance = True

rollingWindow=7
owidurl = "https://covid.ourworldindata.org/data/owid-covid-data.csv"



url = owidurl

s = requests.get(url).content

owid_ds = pd.read_csv(io.StringIO(s.decode('utf-8')))
    

sp_capa_quot_fr_url = "https://www.data.gouv.fr/en/datasets/r/dd0de5d9-b5a5-4503-930a-7b08dc0adc7c"

#"https://www.data.gouv.fr/en/datasets/r/44b46964-8583-4f18-b93f-80fefcbf3b74"

s = requests.get(sp_capa_quot_fr_url).content

capa_fr_ds = pd.read_csv(io.StringIO(s.decode('utf-8')))
capa_fr_ds.cl_age90.unique()

capa_fr_ds.groupby(['cl_age90', 'jour']).p.sum()

capa_fr_ds[(capa_fr_ds.jour==capa_fr_ds.jour.max()) & (capa_fr_ds.cl_age90 != 0)].p.sum()
capa_frt_ds = capa_fr_ds[(capa_fr_ds.cl_age90 == 0)].copy()



capa_frt_ds = capa_frt_ds.reset_index()

capa_frt_ds["location"] = "France"

capa_frt_ds["date"] = capa_frt_ds["jour"]

#owidfr = owid_ds.loc[( (owid_ds['location'] == 'France') & (owid_ds['date'] > "2020-05-12"))].copy()



left = owid_ds.set_index(['location', 'date'])

right = capa_frt_ds.set_index(['location', 'date'])



newdf = left.join(right, lsuffix='', rsuffix='_fr')

newdf = newdf.reset_index()
owidie = owid_ds.loc[( (owid_ds['location'] == 'Ireland') & (owid_ds['date'] > "2020-05-10"))].copy()



owidie["new_testss"]=owidie['total_tests'].diff(7)

values = {'new_testss': 0}

owidie.fillna(value=values, inplace=True)

owidie["new_tests_smootheds"]=owidie["new_testss"].rolling(7).mean()
# France



fig = go.Figure()



fig = make_subplots(

    rows=2, cols=1,

    specs=[[{"secondary_y": True}],[{"secondary_y": True}]],

    subplot_titles=(" Tests", ))



frdf = newdf[(newdf['location'] == 'France') & (newdf.date > "2020-02-28")]

frdf['rolling_new_cases'] = frdf.loc[:,['new_cases']].rolling(window=rollingWindow).mean()

frdf['rolling_new_tests'] = frdf.loc[:,['t']].rolling(window=rollingWindow).mean()



fig.add_trace(go.Bar(x=frdf.date, y=frdf['t'], name=('Tests '),  hovertext='label',

                   ),

             1, 1, secondary_y=False)

fig.add_trace(go.Bar(x=frdf.date, y=frdf['rolling_new_tests'], name=('rolling_new_tests '),  hovertext='label',

                   ),

             1, 1, secondary_y=False)



fig.add_trace(go.Bar(x=frdf.date, y=frdf['new_tests_smoothed'], name=('Owid smoothed Tests '),  hovertext='label',

                   ),

             1, 1, secondary_y=False)



fig.add_trace(go.Bar(x=frdf.date, y=frdf['p'], name=('Positive '),  hovertext='label',

                   ),

             2, 1, secondary_y=False)

fig.add_trace(go.Bar(x=frdf.date, y=frdf['new_cases'], name=('owid Positive '),  hovertext='label',

                   ),

             2, 1, secondary_y=False)

fig.add_trace(go.Bar(x=frdf.date, y=frdf['rolling_new_cases'], name=('rolling_new_cases '),  hovertext='label',

                   ),

             2, 1, secondary_y=False)











fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), 

                  showlegend=True,

                  yaxis=dict(

                        title='Number of ',

                        titlefont_size=16,

                        tickfont_size=14,

                    ),

                  #yaxis_type="log",

                  title_text="France")



#fig.update_layout(barmode='stack', plot_bgcolor='rgb(250, 242, 242)')

fig.show()
capa_frt_ds[(capa_frt_ds.jour>="2020-05-13") & (capa_frt_ds.jour<="2020-06-06")].p.sum()
owid_ds.loc[( (owid_ds['location'] == 'France')), ['date', 'location' , 'total_tests', 'new_tests',

       'total_tests_per_thousand', 'new_tests_per_thousand',

       'new_tests_smoothed', 'new_tests_smoothed_per_thousand', 'tests_units']].tail()
 

sp_pe_tb_quot_fr_url = "https://www.data.gouv.fr/en/datasets/r/57d44bd6-c9fd-424f-9a72-7834454f9e3c"

s = requests.get(sp_pe_tb_quot_fr_url).content

incidence_fr_ds = pd.read_csv(io.StringIO(s.decode('utf-8')))
incidence_fr_ds
(incidence_fr_ds[(incidence_fr_ds.cl_age90 == 0)].groupby("jour").p.sum()).sum()
owid_ds.loc[((owid_ds.date >= "2020-05-13") & (owid_ds['location'] == 'France'))].new_cases.sum()
owid_ds.columns
locations = sorted(owid_ds.location.unique())
# Defaults

baseLineCases=100

PerMillion=False

baseLineDate=False

confirmed=False

newCases=True

deaths=False  

liveCases=True

window_new_deaths=False

new_deaths=False

Ro_est=False

Death_trend=False

rollingWindow=7

million = 1000000

total_tests = True

new_tests = True

new_tests_smoothed = True



def pop_base(row):

    #print("Date: " + row.Date + ", Confirmed: " + str(row.Confirmed) + ", Pop: " + str(row.Population) + ", perm: " + str(row.Confirmed / (row.Population / million)))

    row.total_deaths = row.total_deaths / (row.population / million)

    row.total_cases = row.total_cases  / (row.population / million)

    row.new_cases = row.new_cases  / (row.population / million)

    row.new_deaths = row.new_deaths  / (row.population / million)

    row.rolling_deaths = row.rolling_deaths / (row.population / million)

    row.rolling_cases = row.rolling_cases / (row.population / million)

    row.total_tests = row.total_tests / (row.population / million)

    row.new_tests = row.new_tests / (row.population / million)

    row.new_tests_smoothed = row.new_tests_smoothed / (row.population / million)

    

    return row



def baselineSubplotCountry(fig, countryname, row, col):   



    newdf = owid_ds.loc[(owid_ds['location'] == countryname) & (owid_ds.date > "2020-02-28")].copy()

    

    if ((countryname == 'France') & (customFrance == True)):

        capa_frt_ds = capa_fr_ds[(capa_fr_ds.cl_age90 == 0)].copy()



        capa_frt_ds = capa_frt_ds.reset_index()

        capa_frt_ds["location"] = "France"

        capa_frt_ds["date"] = capa_frt_ds["jour"]

        #owidfr = owid_ds.loc[( (owid_ds['location'] == 'France') & (owid_ds['date'] > "2020-05-12"))].copy()



        left = newdf.set_index(['location', 'date'])

        right = capa_frt_ds.set_index(['location', 'date'])



        newdf = left.join(right, lsuffix='', rsuffix='_fr')

        newdf = newdf.reset_index()

        

        newdf['new_tests'] = newdf['t']

        newdf['new_tests'] = newdf['new_tests'] / newdf['population'] * 1000

        newdf['new_tests_smoothed'] = newdf.loc[:,['t']].rolling(window=rollingWindow).mean()

        newdf['new_tests_smoothed_per_thousand'] = newdf['new_tests_smoothed'] / newdf['population'] * 1000

        

        

    newdf['rolling_deaths'] = newdf.loc[:,['new_deaths']].rolling(window=rollingWindow).mean()

    newdf['rolling_cases'] = newdf.loc[:,['new_cases']].rolling(window=rollingWindow).mean()







    newdf["Label"] = ""

    # Keep only the period from when the number of Confirmed cases / Death is over baseLineCases      

    if (baseLineDate == True):

        if ((deaths==True) | (window_new_deaths == True) | (new_deaths == True)):  

            newdf = newdf.loc[newdf['total_deaths'] >= baseLineCases].reset_index()        

        else:

            newdf = newdf.loc[newdf['total_cases'] >= baseLineCases].reset_index()

    else:

        newdf = newdf.loc[newdf['total_cases'] >= baseLineCases].reset_index()

        #newdf = newdf.reset_index()



    

    if baseLineDate == True:

        xaxis = newdf.index

    else:

        xaxis = newdf['date']  

        

    def mlabel(row):

        row.Label = "Total Deaths: {:,} <br> Total Cases: {:,} <br> Date: {}".format(int(row.total_deaths), int(row.total_cases), row.date)

        return row



    newdf = newdf.apply(mlabel, axis='columns')        



    if PerMillion == True:

        newdf = newdf.apply(pop_base, axis='columns')

    if newCases==True:

        fig.add_trace(go.Bar(x=xaxis, y=newdf['new_cases'], name=('New Cases ' + countryname), hovertext=newdf["Label"],

                            ),

                      row, col, secondary_y=False)



    if confirmed == True:

        fig.add_trace(go.Bar(x=xaxis, y=newdf['total_cases'], name=('Total Cases ' + countryname), hovertext=newdf["Label"],

                            ),

                      row, col, secondary_y=False) 

        

    if liveCases==True:        

        fig.add_trace(go.Scatter(x=xaxis, y=newdf['rolling_cases'], mode='lines', name=('' + countryname + ' New Cases Moving Average '),  hovertext=newdf["Label"],

                            line=dict(width=2)),

                      row, col, secondary_y=False) 

        

    if total_tests == True:

        fig.add_trace(go.Bar(x=xaxis, y=newdf['total_tests'], name=('total_tests ' + countryname), hovertext=newdf["Label"],

                            ),

                      row, col, secondary_y=False) 

        

    if new_tests==True:        

        fig.add_trace(go.Bar(x=xaxis, y=newdf['new_tests'], name=('' + countryname + ' new_tests '), hovertext=newdf["Label"],

                            ),

                      row, col, secondary_y=False) 

        

    if new_tests_smoothed==True:        

        fig.add_trace(go.Bar(x=xaxis, y=newdf['new_tests_smoothed'], name=('' + countryname + ' new_tests_smoothed '), hovertext=newdf["Label"],

                            ),

                      row, col, secondary_y=False) 

        

    if (new_deaths == True):        

        fig.add_trace(go.Bar(x=xaxis, y=newdf['new_deaths'], name=('New Deaths ' + countryname),  hovertext=newdf["Label"],

                            ),

                      row, col, secondary_y=False)



    if (window_new_deaths == True):         

        fig.add_trace(go.Scatter(x=xaxis, y=newdf['rolling_deaths'], mode='lines', name=('' + countryname + ' New Deaths Moving Average '),  hovertext=newdf["Label"],

                            line=dict(width=2)),

                      row, col, secondary_y=False)

            

        

    return newdf

    

countrysToCheck = ['Ireland', 'France',  'United Kingdom', 'United States', 'Netherlands', 'Italy', 'Germany', 'Spain', 'Portugal', 'Australia']

counter = 0

fig = go.Figure()



baseLineCases=5

PerMillion=True

baseLineDate=False

confirmed=False

newCases=False

deaths=False  

liveCases=False

window_new_deaths=False

new_deaths=False

Ro_est = False

rollingWindow=7

million = 1000000

total_tests = False

new_tests = False

new_tests_smoothed = True





fig = make_subplots(

    rows=5, cols=2,

    specs=[[{"secondary_y": True}, {"secondary_y": True}],

           [{"secondary_y": True}, {"secondary_y": True}],

           [{"secondary_y": True}, {"secondary_y": True}],

           [{"secondary_y": True}, {"secondary_y": True}],

           [{"secondary_y": True}, {"secondary_y": True}]],

    subplot_titles=countrysToCheck)





dfcs = [["", 0, 0, 0, ""]]

row = 1

col = 1

for currCountry in countrysToCheck:

    currdfc = baselineSubplotCountry(fig, currCountry, row, col)

    #max_date = np.where(currdfc["short_date"] == currdfc["short_date"].max())[0]

    #currdfc = currdfc.iloc[max_date]

    

    currdfc['Country']= currCountry

    if counter == 0:

        dfcs = currdfc

    else:

        dfcs = dfcs.append(currdfc)

    counter = counter + 1

    if (col == 2):

        row = row + 1

        col = 1

    else:

        col = col + 1

            

chart_date = currdfc["date"].max()



#dfcs = dfcs.reset_index()





fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), 

                  showlegend=False,

                  title_text="New Test Smoothed (rolling 7 days window) per million to date: " + chart_date)



fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
countrysToCheck = ['Ireland', 'France',  'United Kingdom', 'United States', 'Netherlands', 'Italy', 'Germany', 'Spain', 'Portugal', 'Australia']

counter = 0

fig = go.Figure()



baseLineCases=5

PerMillion=True

baseLineDate=False

confirmed=False

newCases=False

deaths=False  

liveCases=False

window_new_deaths=False

new_deaths=False

Ro_est = False

rollingWindow=7

million = 1000000

total_tests = True

new_tests = False

new_tests_smoothed = False





fig = make_subplots(

    rows=5, cols=2,

    specs=[[{"secondary_y": True}, {"secondary_y": True}],

           [{"secondary_y": True}, {"secondary_y": True}],

           [{"secondary_y": True}, {"secondary_y": True}],

           [{"secondary_y": True}, {"secondary_y": True}],

           [{"secondary_y": True}, {"secondary_y": True}]],

    subplot_titles=("Ireland", "France", "UK", 'US', "Netherlands","Italy","Germany", "Spain"))





dfcs = [["", 0, 0, 0, ""]]

row = 1

col = 1

for currCountry in countrysToCheck:

    currdfc = baselineSubplotCountry(fig, currCountry, row, col)

    #max_date = np.where(currdfc["short_date"] == currdfc["short_date"].max())[0]

    #currdfc = currdfc.iloc[max_date]

    

    currdfc['Country']= currCountry

    if counter == 0:

        dfcs = currdfc

    else:

        dfcs = dfcs.append(currdfc)

    counter = counter + 1

    if (col == 2):

        row = row + 1

        col = 1

    else:

        col = col + 1

            

chart_date = currdfc["date"].max()



#dfcs = dfcs.reset_index()





fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), 

                  showlegend=False,

                  title_text="Total Number of Tests per million people to date: " + chart_date)



fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
countrysToCheck = ['Ireland', 'France',  'United Kingdom', 'United States', 'Netherlands', 'Italy', 'Germany', 'Spain', 'Portugal', 'Australia']

counter = 0

fig = go.Figure()



baseLineCases=5

PerMillion=True

baseLineDate=False

confirmed=False

newCases=False

deaths=False  

liveCases=False

window_new_deaths=True

new_deaths=True

Ro_est = False

rollingWindow=7

million = 1000000

total_tests = False

new_tests = False

new_tests_smoothed = False





fig = make_subplots(

    rows=5, cols=2,

    specs=[[{"secondary_y": True}, {"secondary_y": True}],

           [{"secondary_y": True}, {"secondary_y": True}],

           [{"secondary_y": True}, {"secondary_y": True}],

           [{"secondary_y": True}, {"secondary_y": True}],

           [{"secondary_y": True}, {"secondary_y": True}]],

    subplot_titles=countrysToCheck)





dfcs = [["", 0, 0, 0, ""]]

row = 1

col = 1

for currCountry in countrysToCheck:

    currdfc = baselineSubplotCountry(fig, currCountry, row, col)

    #max_date = np.where(currdfc["short_date"] == currdfc["short_date"].max())[0]

    #currdfc = currdfc.iloc[max_date]

    

    currdfc['Country']= currCountry

    

    #max_date = currdfc["date"].max()

    max_date = currdfc[currdfc.new_tests_smoothed.isnull() == False]["date"].max()

    if counter == 0:

        dfcs = currdfc.loc[currdfc.date == max_date]

    else:

        dfcs = dfcs.append(currdfc.loc[currdfc.date == max_date])

    counter = counter + 1

    if (col == 2):

        row = row + 1

        col = 1

    else:

        col = col + 1

            





#dfcs = dfcs.reset_index()





fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), 

                  showlegend=False,

                  title_text="Number of New Deaths per million people to date: " + max_date)



fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
countrysToCheck = ['Ireland', 'France',  'United Kingdom', 'United States', 'Netherlands', 'Italy', 'Germany', 'Spain', 'Portugal', 'Australia']

counter = 0

fig = go.Figure()



baseLineCases=5

PerMillion=True

baseLineDate=False

confirmed=False

newCases=True

deaths=False  

liveCases=True

window_new_deaths=False

new_deaths=False

Ro_est = False

rollingWindow=7

million = 1000000

total_tests = False

new_tests = False

new_tests_smoothed = False





fig = make_subplots(

    rows=5, cols=2,

    specs=[[{"secondary_y": True}, {"secondary_y": True}],

           [{"secondary_y": True}, {"secondary_y": True}],

           [{"secondary_y": True}, {"secondary_y": True}],

           [{"secondary_y": True}, {"secondary_y": True}],

           [{"secondary_y": True}, {"secondary_y": True}]],

    subplot_titles=countrysToCheck)





dfcs = [["", 0, 0, 0, ""]]

row = 1

col = 1

for currCountry in countrysToCheck:

    currdfc = baselineSubplotCountry(fig, currCountry, row, col)

    #max_date = np.where(currdfc["short_date"] == currdfc["short_date"].max())[0]

    #currdfc = currdfc.iloc[max_date]

    

    currdfc['Country']= currCountry

    

    #max_date = currdfc["date"].max()

    max_date = currdfc[currdfc.new_tests_smoothed.isnull() == False]["date"].max()

    if counter == 0:

        dfcs = currdfc.loc[currdfc.date == max_date]

    else:

        dfcs = dfcs.append(currdfc.loc[currdfc.date == max_date])

    counter = counter + 1

    if (col == 2):

        row = row + 1

        col = 1

    else:

        col = col + 1

            





#dfcs = dfcs.reset_index()





fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), 

                  showlegend=False,

                  title_text="Number of New Cases per million people to date: " + max_date)



fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
currdfc[currdfc.new_tests_smoothed.isnull() == False]["date"].max()
currdfc.loc[: , ['date', 'location', 'population', 'stringency_index', 'new_tests_smoothed', 'new_cases', 'rolling_cases']].tail()
dfcs.columns
dfcs.loc[:, ['date', 'location', 'total_cases', 'population', 'stringency_index', 'new_tests_smoothed', 'rolling_cases', 'rolling_deaths']]


def shortendate(row):

    year, day, month = row.date.split('-') 

    row.date = "/".join([month, day])

    return row



dfcs = dfcs.apply(shortendate, axis='columns')        

fig = go.Figure()



size = (dfcs.population / 1000000)

rates = (dfcs.rolling_deaths)



fig.add_trace(go.Scatter(

    x=(dfcs.rolling_cases),

    y=(dfcs.new_tests_smoothed ),

    name="plot",

    text=dfcs.location + ": " + dfcs.date,

    hovertemplate=

    "<b>%{text}</b><br><br>" +

    "Daily Cases: %{x:.0f} per million <br>" +

    "Daily Tests: %{y:.0f} per million <br>" +

    "Daily Deaths: %{marker.color:.0f} per million <br>" +

    "Population: %{marker.size:.0f} millions" +

    "<extra></extra>",

    mode='markers+text',

    marker=dict(

        size=size,

        sizemode='area',

        sizeref=2.*max(size)/(40.**2),

        sizemin=4,

        color=rates,

        showscale=True

    )

    #(dfcs.Population / 1000000),

    ))    

    



fig.update_traces(textposition='top center')

    



fig.update_layout(

    xaxis={

        'title':'New Cases per million people',

        'type':'log'},

    yaxis={'title':'New Tests per milion people ',

        'type':'log'},

    title_text="COVID-19: Daily tests vs. Daily new confirmed cases per million",

    showlegend=True,

)



fig.show()
