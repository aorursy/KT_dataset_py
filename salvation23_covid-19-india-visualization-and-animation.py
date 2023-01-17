# import libs

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import Markdown, display

%matplotlib inline



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import cufflinks as cf

cf.go_offline()



import plotly.express as px

import plotly.graph_objects as go



# Load covid_19_india

covid_19_india_df = pd.read_csv('../input/covid19-in-india/covid_19_india.csv', index_col='Sno')



covid_19_india_df.Date = pd.to_datetime(covid_19_india_df.Date, format='%d/%m/%y')

covid_19_india_df = covid_19_india_df.dropna()
date_table = covid_19_india_df.groupby(['Date', 'State/UnionTerritory']).sum().reset_index().set_index('Date')

total = date_table.loc[date_table.last_valid_index()].sum()

confirmed_count = int(total.Confirmed)

death_count = int(total.Deaths)

cured_count = int(total.Cured)



html_template = "<span style='color:{}; font-size:1.4em;'>{}</span>"

cured = html_template.format('green', 'Cured - '+str(cured_count))    

confirmed = html_template.format('blue', 'Confirmed cases - '+str(confirmed_count))    

deaths = html_template.format('red', 'Deaths - '+str(death_count))



display(Markdown(html_template.format('black', 'Summary :')))

display(Markdown(cured))

display(Markdown(confirmed))

display(Markdown(deaths))
today_states = date_table.loc[date_table.last_valid_index()].reset_index()

max_confrim = today_states[today_states.Confirmed == today_states.Confirmed.max()]

max_deaths = today_states[today_states.Deaths == today_states.Deaths.max()]

max_cured = today_states[today_states.Cured == today_states.Cured.max()]



display(Markdown(html_template.format('black', 'States summary :')))

display(Markdown(html_template.format('DodgerBlue','Maximum confirmed cases - '+ str(max_confrim.Confirmed.iloc[0]) + ', '+max_confrim['State/UnionTerritory'].iloc[0])))

display(Markdown(html_template.format('MediumSeaGreen','Maximum cured - '+ str(max_cured.Cured.iloc[0]) + ', '+max_cured['State/UnionTerritory'].iloc[0])))

display(Markdown(html_template.format('Tomato','Maximum deaths - '+ str(max_deaths.Deaths.iloc[0]) + ', '+max_deaths['State/UnionTerritory'].iloc[0])))
date_group = covid_19_india_df.groupby(['Date']).sum()

date_group.reset_index(inplace=True)



date_group.sort_values('Date',inplace=True)



from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=3, shared_yaxes=True,

                    subplot_titles=("Total deaths - " + str(death_count),"Total confirmed - " + str(confirmed_count),

                                    "Total cured - " + str(cured_count)))



# Create and style traces

fig.add_trace(go.Scatter(x=date_group.Date, y=date_group.Deaths, name='Deaths',

                         line=dict(color='firebrick', width=1), mode='lines+markers'), row=1, col=1)

fig.add_trace(go.Scatter(x=date_group.Date, y=date_group.Confirmed, name = 'Confirmed',

                         line=dict(color='royalblue', width=1), mode='lines+markers',), row=1, col=2)

fig.add_trace(go.Scatter(x=date_group.Date, y=date_group.Cured, name='Cured',

                         line=dict(color='green', width=1), mode='lines+markers',), row=1, col=3)



fig.update_layout(

    title="All Cases",

    xaxis_title="Date",

    yaxis_title="Count",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="black"

    )

)



fig.update_xaxes(ticks="inside")

fig.update_yaxes(ticks="inside", col=1)



fig.show()



fig = go.Figure(data=[

    go.Bar(name='Confirmed', x=date_group.Date, y=date_group.Confirmed, marker_color='green'),

    go.Bar(name='Deaths', x=date_group.Date, y=date_group.Deaths, marker_color='firebrick'),

    go.Bar(name='Cured', x=date_group.Date, y=date_group.Cured, marker_color='royalblue')

])

# Change the bar mode

fig.update_layout(autosize=False,

    width=1000,

    height=750,

    title="All Cases",

    xaxis_title="Date",

    yaxis_title="Count",

    barmode='relative',

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="black"

    ))

fig.show()




import plotly.express as px

fig = px.bar(date_group, x='Date', y='Confirmed', color='Confirmed', color_continuous_scale=px.colors.sequential.Viridis)



fig.update_layout(

    title="Confirmed cases",    

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="black"

    )

)



fig.show()
fig = px.bar(date_group, x='Date', y='Deaths', color='Deaths', color_continuous_scale=px.colors.sequential.Viridis)



fig.update_layout(

    title="Deaths",    

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="black"

    )

)



fig.show()
fig = px.bar(date_group, x='Date', y='Cured', color='Cured', color_continuous_scale=px.colors.sequential.Viridis)



fig.update_layout(

    title="Cured",    

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="black"

    )

)



fig.show()
fig = go.Figure(data=[

    go.Bar(name='Confirmed', x=date_table.loc[date_table.last_valid_index()]['State/UnionTerritory'], y=date_table.loc[date_table.last_valid_index()].Confirmed, marker_color='green'),

    go.Bar(name='Deaths', x=date_table.loc[date_table.last_valid_index()]['State/UnionTerritory'], y=date_table.loc[date_table.last_valid_index()].Deaths, marker_color='firebrick'),

    go.Bar(name='Cured', x=date_table.loc[date_table.last_valid_index()]['State/UnionTerritory'], y=date_table.loc[date_table.last_valid_index()].Cured, marker_color='royalblue')

])

# Change the bar mode

fig.update_layout(

    autosize=False,

    width=1000,

    height=750,

    title="State wise cases",

    xaxis_title="State",

    yaxis_title="Count",

    barmode='stack',

    )

fig.show()
fig = px.pie(date_table.loc[date_table.last_valid_index()], values='Confirmed', names='State/UnionTerritory', title='Confirmed cases')

fig.update_layout(

    autosize=False,

    width=900,

    height=650)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
fig = px.pie(date_table.loc[date_table.last_valid_index()], values='Deaths', names='State/UnionTerritory', title='Deaths')

fig.update_layout(

    autosize=False,

    width=900,

    height=650,

    )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
fig = px.pie(date_table.loc[date_table.last_valid_index()], values='Cured', names='State/UnionTerritory', title='Cured')

fig.update_layout(

    autosize=False,

    width=900,

    height=650)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
dummy = pd.DataFrame()



dummy['State'] = covid_19_india_df['State/UnionTerritory'].unique()



#colors = sns.dark_palette("purple", 33).as_hex()

#colors = sns.diverging_palette(250, 15, s=75, l=40, n=33, center="dark").as_hex()

#colors = sns.color_palette("Blues_d", 33).as_hex()

colors =sns.color_palette("viridis", 33).as_hex()



def animate_bar_chart():

    frames = []

    grouped = covid_19_india_df[covid_19_india_df['Date'] > '2020/03'][['Date', 'State/UnionTerritory', 'Confirmed', 'Deaths', 'Cured']].groupby(['Date'])    

    for name, group in iter(grouped):

        merged = pd.merge(group, dummy, how='outer', left_on='State/UnionTerritory', right_on='State')        

        merged.fillna(0, inplace=True)

        merged.sort_values('State', inplace=True)

        frames.append(go.Frame(data = [go.Bar(x = merged['State'].tolist(), y=merged['Confirmed'].tolist(), marker_color=colors)], 

                              layout=go.Layout(title='Confirmed cases - '+group.Date.iloc[0].strftime('%Y-%m-%d'))))

    

    fig = go.Figure(

        data = [go.Bar(x = merged['State'].tolist(), y = [0] * len(merged['State'].tolist()))],

        frames=frames, 

        layout=go.Layout(

            width=1000,

            height=750,

            xaxis=dict(type='category'),

            yaxis=dict(range=[0, 7000], autorange=False),            

            title="Confirmed cases",

            xaxis_title="State",

            yaxis_title="Count",

            updatemenus=[dict(

                type="buttons",

                buttons=[dict(label="Play",

                              method="animate",

                              args=[None])])]))

    fig.show()

                                       

                                       



animate_bar_chart()


hospital_df = pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv', index_col='Sno')

hospital_df['TotalBeds'] = hospital_df.sum(axis=1)

hospital_df = hospital_df[hospital_df['State/UT'] != 'All India']

fig = px.bar(hospital_df.sort_values('TotalBeds', ascending=True), x="TotalBeds", y="State/UT", orientation='h', color='TotalBeds', color_continuous_scale='Viridis')

fig.update_layout(

    autosize=False,

    width=1200,

    height=750,

    title="Total hospital beds in states",

    yaxis_title="State",

    xaxis_title="Bed Count"    

    )

fig.show()
age_df = pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv', index_col='Sno')

cm = sns.light_palette("green", as_cmap=True)

age_df.style.background_gradient(cmap=cm)
px.bar(age_df, x='AgeGroup', y='TotalCases', color='TotalCases', 

       hover_data=['AgeGroup', 'TotalCases', 'Percentage'], 

       height=750, width=1000, color_continuous_scale='Viridis', title='Age group Cases')
icmr_testing_df = pd.read_csv('../input/covid19-in-india/ICMRTestingDetails.csv', index_col='SNo', parse_dates=[1], dayfirst=True)
fig = go.Figure()

fig.add_trace(go.Scatter(x=icmr_testing_df.DateTime, y=icmr_testing_df.TotalSamplesTested,

                    mode='lines+markers',

                    name='Total Samples Tested', connectgaps=True))



fig.add_trace(go.Scatter(x=icmr_testing_df.DateTime, y=icmr_testing_df.TotalIndividualsTested,

                    mode='lines+markers',

                    name='Total Individuals Tested', connectgaps=True))



fig.add_trace(go.Scatter(x=icmr_testing_df.DateTime, y=icmr_testing_df.TotalPositiveCases,

                    mode='lines+markers',

                    name='Total Positive Cases', connectgaps=True))



#fig.update_layout(yaxis_type="log")

fig.update_layout(title='ICMR Test result',

                   xaxis_title='Date',

                   yaxis_title='Total Cases')

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=icmr_testing_df.DateTime, y=icmr_testing_df.TotalSamplesTested,

                    mode='lines+markers',

                    name='Total Samples Tested', connectgaps=True))



fig.add_trace(go.Scatter(x=icmr_testing_df.DateTime, y=icmr_testing_df.TotalIndividualsTested,

                    mode='lines+markers',

                    name='Total Individuals Tested', connectgaps=True))



fig.add_trace(go.Scatter(x=icmr_testing_df.DateTime, y=icmr_testing_df.TotalPositiveCases,

                    mode='lines+markers',

                    name='Total Positive Cases', connectgaps=True))



fig.update_layout(yaxis_type="log")

fig.update_layout(title='ICMR Test result - log scale',

                   xaxis_title='Date',

                   yaxis_title='Total Cases')

fig.show()