# !pip install plotly
import math

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import warnings

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.express as px

from scipy.stats import pearsonr

from sklearn import linear_model
warnings.filterwarnings("ignore", 'Boolean Series key will be reindexed to match DataFrame index')



train_data_path = "../input/training/train.csv"



countries=['Austria','Germany','United Kingdom','Vietnam','South Korea','Singapore','Israel','Japan','Sweden','San Marino','Slovenia','Canada','Hong Kong','Taiwan','United States','European Union','Thailand','Italy','Czechia','Australia','Trinidad and Tobago','Qatar','North Korea','New Zealand','Colombia','Romania','France','Portugal','Spain','Belgium','Luxembourg','Albania','Andorra','Azerbaijan','Belarus','Bosnia and Herzegovina','Bulgaria','Denmark','Estonia','Cyprus','Croatia','Finland','Georgia','Hungary','Latvia','Lithuania','Greece','Moldova','Malta','Monaco','Netherlands','Iceland','Guernsey','Macedonia','Ireland','Vatican City','Jersey','Kosovo','Kazakhstan','Poland','Turkey','Ukraine','Slovakia','Serbia','Switzerland','Norway','Montenegro','Iran','Liechtenstein','Russia','Mexico','Egypt','Palestine','Malaysia','Nepal','Afghanistan','Iraq','Faroe Islands','Philippines','Kuwait','South Africa','Armenia','Pakistan','Brazil','Costa Rica','Panama','India','Bahrain','United Arab Emirates','Kyrgyzstan','Indonesia','Namibia','Morocco','Uganda']
# Read and parse cases data

cases_data = pd.read_csv(train_data_path,

                    sep=',',

                    dtype={'Country_Region': str, 'ConfirmedCases': int, 'Fatalities': int},

                    parse_dates=['Date'])

cases_data = cases_data.rename(columns={"Country_Region": "Country"})

cases_data = cases_data[pd.isnull(cases_data["Province_State"])]



# Drop fields

cases_data = cases_data[["Country", "Date", "ConfirmedCases", "Fatalities"]]



# Filter on countries

cases_data = cases_data[cases_data["Country"].str.contains("|".join(countries))]



# Calculate cases/fatalities per day

cases_data["ConfirmedCasesDay"] = cases_data.groupby("Country")["ConfirmedCases"].shift(0) - cases_data.groupby("Country")["ConfirmedCases"].shift(1) 

cases_data["FatalitiesDay"] = cases_data.groupby("Country")["Fatalities"].shift(0) - cases_data.groupby("Country")["Fatalities"].shift(1) 



# To reduce clutter, limit to 5 countries

countries_subset = ["Germany", "Italy", "Spain", "Netherlands","Belgium"]

cases_data_subset = cases_data[cases_data['Country'].isin(countries_subset)]



cases_data
display(cases_data.info())

display(cases_data.describe())
df_all = cases_data.groupby(['Date']).sum()



# cumulative

fig = go.Figure(data=[

    go.Bar(name='Confirmed cases', x=df_all.index, y=df_all['ConfirmedCases']),

    go.Bar(name='Fatalities', x=df_all.index, y=df_all['Fatalities'])

])



fig.update_layout(barmode='overlay',

                  title='Cumulative confirmed cases and fatalities worldwide', 

                  yaxis={'title':'Confirmed cases and fatalities'})

fig.show()



# daily

fig = go.Figure(data=[

    go.Bar(name='Confirmed cases per day', x=df_all.index, y=df_all['ConfirmedCasesDay']),

    go.Bar(name='Fatalities per day', x=df_all.index, y=df_all['FatalitiesDay'])

])



fig.update_layout(barmode='overlay',

                  title='Daily confirmed cases and fatalities worldwide', 

                  yaxis={'title':'Confirmed cases and fatalities'})

fig.show()
# Cumulative



# Plot confirmed cases per country

fig = px.line(cases_data_subset, x='Date', y='ConfirmedCases', color='Country')

fig.update_layout(title="Cumulative confirmed cases for " +  ", ".join(countries_subset),

                  font={'size':10})

fig.show()



# Plot fatalities per country

fig = px.line(cases_data_subset, x='Date', y='Fatalities', color='Country')

fig.update_layout(title="Cumulative fatalities for " +  ", ".join(countries_subset), 

                  font={'size':10})

fig.show()
# Daily



# Plot confirmed cases per country

fig = px.line(cases_data_subset, x='Date', y='ConfirmedCasesDay', color='Country')

fig.update_layout(title="Daily confirmed cases for " +  ", ".join(countries_subset), 

                  font={'size':10}, 

                  yaxis={'title':'Confirmed cases'})

fig.show()



# Plot fatalities per country

fig = px.line(cases_data_subset, x='Date', y='FatalitiesDay', color='Country')

fig.update_layout(title="Daily fatalities for " +  ", ".join(countries_subset), 

                  font={'size':10}, 

                  yaxis={'title':'Fatalities'})

fig.show()
# Load containment measures 

containment_measures = pd.read_csv("../input/export-containment-measures/export_containment_measures.csv",parse_dates=['StartDate'])[["Country", "StartDate", "Label", "Severity"]]
# Smooth data

cases_data["ConfirmedCasesDaySmooth"] = cases_data["ConfirmedCasesDay"].rolling(window=7).mean()

cases_data["FatalitiesDaySmooth"] = cases_data["FatalitiesDay"].rolling(window=7).mean()
# Calculate growth of cases/fatalities per day

cases_data["ConfirmedCasesDeltaGrowth"] = cases_data.groupby("Country")["ConfirmedCasesDaySmooth"].shift(-1) - cases_data.groupby("Country")["ConfirmedCasesDaySmooth"].shift(0) 

cases_data["FatalitiesDeltaGrowth"] = cases_data.groupby("Country")["FatalitiesDaySmooth"].shift(-1) - cases_data.groupby("Country")["FatalitiesDaySmooth"].shift(0) 
# Plot confirmed cases growth per country

fig = go.Figure()



for country in countries_subset:

    df_c = cases_data[cases_data["Country"] == country]

    df_m = containment_measures[containment_measures["Country"] == country]



    fig.add_trace(go.Scatter(x=df_c['Date'], 

                             y=df_c['ConfirmedCasesDeltaGrowth'],

                             mode='lines',

                             name=country))

    fig.add_trace(go.Scatter(x=df_m['StartDate'],

                             y=df_m['Severity']*333,

                             mode='markers', 

                             name=country,

                             hoverinfo='name+text+x', 

                             showlegend=False,

                             text=df_m['Label']))

fig.update_layout(title="Daily growth of confirmed cases growth for " +  ", ".join(countries_subset), 

                  font={'size':10}, 

                  yaxis={'title':'Growth of confirmed cases'})

fig.show()





# Plot the growth of fatality per country

fig = go.Figure()

for country in countries_subset:

    df_c = cases_data[cases_data["Country"] == country]

    df_m = containment_measures[containment_measures["Country"] == country]



    fig.add_trace(go.Scatter(x=df_c['Date'], 

                             y=df_c['FatalitiesDeltaGrowth'],

                             mode='lines',

                             name=country))

    fig.add_trace(go.Scatter(x=df_m['StartDate'],

                             y=df_m['Severity']*33,

                             mode='markers', 

                             name=country, 

                             hoverinfo='name+text+x', 

                             showlegend=False,

                             text=df_m['Label']))

fig.update_layout(title="Daily growth fatalities growth for " +  ", ".join(countries_subset), 

                  font={'size':10}, 

                  yaxis={'title':'Growth of fatalities'})

fig.show()
cases_data["FatalitiesDeltaGrowthSmooth"] = cases_data["FatalitiesDeltaGrowth"].rolling(window=4).mean()

cases_data["ConfirmedCasesDeltaGrowthSmooth"] = cases_data["ConfirmedCasesDeltaGrowth"].rolling(window=4).mean()



# Plot the growth of confirmed cases per country

fig = go.Figure()

for country in countries_subset:

    df_c = cases_data[cases_data["Country"] == country]

    df_m = containment_measures[containment_measures["Country"] == country]



    fig.add_trace(go.Scatter(x=df_c['Date'], 

                             y=df_c['ConfirmedCasesDeltaGrowthSmooth'],

                             mode='lines',

                             name=country))

    fig.add_trace(go.Scatter(x=df_m['StartDate'],

                             y=df_m['Severity']*cases_data['ConfirmedCasesDeltaGrowthSmooth'].max()/3,

                             mode='markers', 

                             name=country,

                             hoverinfo='name+text+x', 

                             showlegend=False,

                             text=df_m['Label']))

fig.update_layout(title="Daily growth of confirmed cases growth for " +  ", ".join(countries_subset), 

                  font={'size':10}, 

                  yaxis={'title':'Growth of confirmed cases'})

fig.show()





# Plot the growth of fatality per country

fig = go.Figure()

for country in countries_subset:

    df_c = cases_data[cases_data["Country"] == country]

    df_m = containment_measures[containment_measures["Country"] == country]



    fig.add_trace(go.Scatter(x=df_c['Date'], 

                             y=df_c['FatalitiesDeltaGrowthSmooth'],

                             mode='lines',

                             name=country))

    fig.add_trace(go.Scatter(x=df_m['StartDate'],

                             y=df_m['Severity']*cases_data['FatalitiesDeltaGrowthSmooth'].max()/3,

                             mode='markers', 

                             name=country, 

                             hoverinfo='name+text+x', 

                             showlegend=False,

                             text=df_m['Label']))

fig.update_layout(title="Daily growth fatalities growth for " +  ", ".join(countries_subset), 

                  font={'size':10}, 

                  yaxis={'title':'Growth of fatalities'})

fig.show()
results = []



for country in countries:

    df_m = containment_measures[containment_measures["Country"] == country]

    for i,measure in df_m.iterrows():

        

        field = "ConfirmedCasesDeltaGrowthSmooth"

        offset_days = 14 # 12

        days_range = 14 # 14

        

        country = measure[0]

        

        date_before = str(measure[1] + pd.Timedelta(days=-(days_range/2)))[0:10]

        date_after = str(measure[1] + pd.Timedelta(days=days_range/2))[0:10]

        

        avg_growth_before = cases_data[cases_data["Country"] == country][(date_before < cases_data["Date"]) & (cases_data["Date"] < date_after)][field].mean()

        

        date_before = str(measure[1] + pd.Timedelta(days=offset_days))[0:10]

        date_after = str(measure[1] + pd.Timedelta(days=offset_days+days_range))[0:10]

        

        avg_growth_after = cases_data[cases_data["Country"] == country][(date_before < cases_data["Date"]) & (cases_data["Date"] < date_after)][field].mean()

        

        results.append([country, str(measure[1])[0:10], measure[2], measure[3], avg_growth_before, avg_growth_after]) 

        

df_performance = pd.DataFrame(results, columns=["Country", "DateStart", "Measure", "Severity", "Avg growth before", "Avg Growth After"])

df_performance = df_performance.dropna()

df_performance["Worked"] = df_performance.apply(lambda x: x["Avg growth before"] > x["Avg Growth After"],axis=1)



df_performance = df_performance.sort_values(by=['Measure'])





performance_2 = []

for measure in list(containment_measures["Label"].drop_duplicates()):

    worked = len(df_performance[(df_performance["Measure"] == measure) & (df_performance["Worked"] == True)])

    n_worked = len(df_performance[(df_performance["Measure"] == measure) & (df_performance["Worked"] == False)])

    performance_2.append([measure, worked, n_worked])



df_performance_2 = pd.DataFrame(performance_2, columns=["Measure", "Success", "Failure"])

df_performance_2
fig = go.Figure(data=[

    go.Bar(name='Decreased', x=df_performance_2["Measure"], y=df_performance_2["Success"]),

    go.Bar(name='Increased', x=df_performance_2["Measure"], y=df_performance_2["Failure"])

])



# Change the bar mode

fig.update_layout(barmode='group',

                 yaxis={'title':'Number of countries'})

fig.show()
# Fix nan in FatalitiesDeltaGrowthSmooth

cases_data["FatalitiesDeltaGrowthSmooth_nan"] = cases_data.apply(lambda x: 0 if str(x["FatalitiesDeltaGrowthSmooth"]) == "nan" else x["FatalitiesDeltaGrowthSmooth"] ,axis=1)

cases_data["ConfirmedCasesDeltaGrowthSmooth_nan"] = cases_data.apply(lambda x: 0 if str(x["ConfirmedCasesDeltaGrowthSmooth"]) == "nan" else x["ConfirmedCasesDeltaGrowthSmooth"] ,axis=1)



# Add numbers to containment measures

def get_from_cases_data(x,field):

    cd = cases_data[(cases_data["Country"] == x["Country"]) & (cases_data["Date"] == x["StartDate"])][field]

    if len(cd) == 0:

        return 0

    else:

        return int(cd)



containment_measures["FatalitiesGrowthStartDate"] = containment_measures.apply(lambda x: get_from_cases_data(x,"FatalitiesDeltaGrowthSmooth_nan") ,axis=1)

containment_measures["ConfirmedCasesGrowthStartDate"] = containment_measures.apply(lambda x:  get_from_cases_data(x,"ConfirmedCasesDeltaGrowthSmooth_nan"),axis=1)



containment_measures.head()
measures = list(containment_measures["Label"].drop_duplicates())



fig = make_subplots(rows=3, cols=2, subplot_titles=(range(1, 7)),

                   vertical_spacing=0.1)



for i,measure in zip(range(len(measures)),measures):

    x,y,z=[],[],[]

    for country in countries:

        try:

            c_start = int(containment_measures[(containment_measures["Country"] == country) & (containment_measures["Label"] == measure)]["ConfirmedCasesGrowthStartDate"])

            c_now = max(cases_data[cases_data["Country"] == country]["ConfirmedCases"])

            x.append(c_start)

            y.append(c_now)

            z.append(country)

        except:

            continue



    fig.add_trace(go.Scatter(x=[i+1 for i in x], 

                             y=y,

                             text=z,

                             mode='markers',

                             hoverinfo='x+y+text',

                             showlegend=False,

                             name="\n" + measure + " - " + str(round(pearsonr(x,y)[0],4))),

                  row=int(i/2)+1,

                  col=(i%2)+1)



    fig.layout.annotations[i].update(text="\n" + measure + " - " + 'Pearson corr. = '+ 

                                 str(round(pearsonr(np.log(np.array([i+-1*min(x)+1 for i in x])),

                                                    np.log(np.array([i+1 for i in y])))[0],4)))





    fig.update_xaxes(title_text="Growth of confirmed cases at start of measure", 

                     type='log',

                     row=int(i/2)+1, 

                     col=(i%2)+1)

    fig.update_yaxes(title_text="Growth of confirmed cases now", 

                     type='log',

                     row=int(i/2)+1, 

                     col=(i%2)+1)



fig.update_layout(title='Growth of confirmed cases at start of measure vs the growth of confirmed cases now',

                  height=900,

                  title_font_size=16)



fig.show()
measures = list(containment_measures["Label"].drop_duplicates())



fig = make_subplots(rows=3, cols=2, subplot_titles=(range(1, 7)),

                   vertical_spacing=0.1)



for i,measure in zip(range(len(measures)),measures):

    x,y,z=[],[],[]

    for country in countries:

        try:

            c_start = int(containment_measures[(containment_measures["Country"] == country) & (containment_measures["Label"] == measure)]["ConfirmedCasesGrowthStartDate"])

            c_now = max(cases_data[cases_data["Country"] == country]["ConfirmedCases"])

            x.append(c_start)

            y.append(c_now)

            z.append(country)

        except:

            continue



    fig.add_trace(go.Scatter(x=[i+1 for i in x], 

                             y=y,

                             text=z,

                             mode='markers',

                             hoverinfo='x+y+text',

                             showlegend=False,

                             name="\n" + measure + " - " + str(round(pearsonr(x,y)[0],4))),

                  row=int(i/2)+1,

                  col=(i%2)+1)

    

    lm = linear_model.LinearRegression()

    _x = [[i] for i in x]

    model = lm.fit(_x,y)



    x2 =[i for i in range(0,6)]

    y2 = [model.intercept_ + i*model.coef_[0] for i in [1,max(x)]]

    

    fig.add_trace(go.Scatter(x=[1,max(x)],

                             y=y2,

                             showlegend=False,

                            mode='lines'),

                  row=int(i/2)+1,

                  col=(i%2)+1)

    

    fig.layout.annotations[i].update(text="\n" + measure + " - " + 'Pearson corr. = '+ 

                                 str(round(pearsonr(np.log(np.array([i+-1*min(x)+1 for i in x])),

                                                    np.log(np.array([i+1 for i in y])))[0],4)))





    fig.update_xaxes(title_text="Growth of confirmed cases at start of measure", 

                     type='log',

                     row=int(i/2)+1, 

                     col=(i%2)+1)

    fig.update_yaxes(title_text="Growth of confirmed cases now", 

                     type='log',

                     row=int(i/2)+1, 

                     col=(i%2)+1)



fig.update_layout(title='Growth of confirmed cases at start of measure vs the growth of confirmed cases on April 9th',

                  height=900,

                  title_font_size=16)



fig.show()