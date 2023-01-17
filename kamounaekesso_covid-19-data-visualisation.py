import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

import plotly.express as px

import plotly.graph_objects as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot, plot_mpl

import plotly.offline as py

init_notebook_mode(connected=True)

plt.rcParams.update({'font.size': 14})
#confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')

#deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')

#recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')
listes1=['Algeria','Angola','Benin','Botswana','Burkina Faso','Burundi','Cameroon','Cabo Verde','Central African Republic','Chad',

'Camoros','Congo (Kinshasa)','Congo (Brazzaville)','Djibouti','Egypt','Equatorial Guinea','Eritrea','Ethiopia','Gabon','Gambia','Ghana','Guinea',

'Guinea-Bissau',"Cote d'Ivoire",'Kenya','Lesotho','Liberia','Libya','Madagascar','Malawi','Mali','Mauritania','Mauritius','Morocco','Mozambique',

'Namibia','Niger','Nigeria','Rwanda','Sao Tome and Principe','Senegal','Seychelles','Sierra Leone','Somalia','South Africa','South Sudan','Sudan',

'Tanzania','Togo','Tunisia','Uganda','Zambia','Zimbabwe','Western Sahara','Eswatini']
confirmed_df = pd.read_csv("../input/les-datasets-utiliss/confirmes_du_covid.csv")

confirmed_df.drop(["Unnamed: 0","Province/State"],axis=1,inplace =True)

confirmed_df.drop(confirmed_df.iloc[:,3:26],axis=1,inplace=True)

confirmed_df=confirmed_df[confirmed_df['Country/Region'].isin(listes1)]



deaths_df = pd.read_csv("../input/les-datasets-utiliss/morts_du_covid.csv")

deaths_df.drop(["Unnamed: 0","Province/State"],axis=1,inplace =True)

deaths_df.drop(deaths_df.iloc[:,3:26],axis=1,inplace=True)

deaths_df=deaths_df[deaths_df['Country/Region'].isin(listes1)]



recoveries_df = pd.read_csv("../input/les-datasets-utiliss/guris_du_covid.csv")

recoveries_df.drop(["Unnamed: 0","Province/State"],axis=1,inplace =True)

recoveries_df.drop(recoveries_df.iloc[:,3:26],axis=1,inplace=True)

recoveries_df=recoveries_df[recoveries_df['Country/Region'].isin(listes1)]



confirmed_df = confirmed_df.melt(id_vars=['Country/Region', 'Lat', 'Long'], var_name='date', value_name='confirmed')

deaths_df = deaths_df.melt(id_vars=['Country/Region', 'Lat', 'Long'], var_name='date', value_name='deaths')

recoveries_df = recoveries_df.melt(id_vars=['Country/Region', 'Lat', 'Long'], var_name='date', value_name='recoveries')

recoveries_df.dtypes
confirmed_df['date'] = pd.to_datetime(confirmed_df['date'])

deaths_df['date'] = pd.to_datetime(deaths_df['date'])

recoveries_df['date'] = pd.to_datetime(recoveries_df['date'])

complet_df = confirmed_df.merge(recoveries_df).merge(deaths_df)

#renommer les colonnes pour un travail plus allégé

complet_df = complet_df.rename(columns={'Country/Region': 'Country', 'date': 'Date', 'confirmed': "Confirmed", "recoveries": "Recoveries", "deaths": "Deaths"})

# verifier les valeurs nulles

complet_df.isnull().sum()

complet_df
def plot_treemap(col):

    fig = px.treemap(complet_df.iloc[-54: ,: ], path=['Country'], values=col, height=900,

                 title=col[-54:], color_discrete_sequence = px.colors.qualitative.Dark2)

    fig.data[0].textinfo = 'label+text+value'

    fig.show()

print('                                                                             ')

print('=====Total confirmed cases in Africa :',complet_df['Confirmed'][-54:].sum(),'=========' )

plot_treemap('Confirmed')
print('                                                                             ')

print('=====Total de cas guéris en Afrique:',complet_df['Recoveries'][-54: ].sum(),'=========' )

plot_treemap('Recoveries')
complet_df["Active"]=complet_df['Confirmed']-(complet_df["Recoveries"]+complet_df["Deaths"])

print('                                                                             ')

print('=====Total de cas actif en Afrique:',(complet_df['Confirmed']-(complet_df["Recoveries"]+complet_df["Deaths"][-54: ])).sum(),'====' )

plot_treemap('Active')
print('                                                                             ')

print('=====Le nombre total de décès en Afrique est :',complet_df['Deaths'][-54:].sum(),'====' )

plot_treemap('Deaths')
complet_df = confirmed_df.merge(recoveries_df).merge(deaths_df)

#renommer les colonnes pour un travail plus allégé

complet_df = complet_df.rename(columns={'Country/Region': 'Country', 'date': 'Date', 'confirmed': "Confirmed", "recoveries": "Recoveries", "deaths": "Deaths"})

# verifier les valeurs nulles

#complet_df.isnull().sum()



Africa_df = complet_df.groupby(['Date']).agg({'Confirmed': ['sum'], 'Recoveries': ['sum'], 'Deaths': ['sum']}).reset_index()

Africa_df.columns = Africa_df.columns.get_level_values(0)



def ajout_taux(data):

    data['Confirmed Change'] = data['Confirmed'].diff().shift(-1)

    data['Mortality Rate'] = data['Deaths'] / data['Confirmed']

    data['Recovery Rate'] = data['Recoveries'] / data['Confirmed']

    data['Growth Rate'] = data['Confirmed Change'] / data['Confirmed']

    data['Growth Rate Change'] = data['Growth Rate'].diff().shift(-1)

    data['Growth Rate Accel'] = data['Growth Rate Change'] / data['Growth Rate']

    return data



Africa_df = ajout_taux(Africa_df)



def plot_map(df, col, pal):

    df = df[df[col]>0]

    fig = px.choropleth(df, locations="Country", locationmode='country names', 

                  color=col, hover_name="Country", 

                  title=col, hover_data=[col], color_continuous_scale=pal)

    fig.update_layout(coloraxis_showscale=True)

    fig.show()

plot_map(complet_df.iloc[-54:,:],'Confirmed','matter')
def plot_aggregate_curve(df, fig=None):

    if fig is None:

        fig = go.Figure()

    fig.update_layout(template='plotly_dark')

    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Confirmed'],

                             mode='lines+markers',

                             name='Cas confirmés',

                             line=dict(color='Yellow', width=2)

                            ))

    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Deaths'],

                             mode='lines+markers',

                             name='Morts',

                             line=dict(color='Red', width=2)

                            ))

    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Recoveries'],

                             mode='lines+markers',

                             name='Guérisons',

                             line=dict(color='Green', width=2)

                            ))

    return fig
plot_aggregate_curve(Africa_df).show()
def plot_diff_curve(df, fig=None):

    if fig is None:

        fig = go.Figure()



    fig.update_layout(template='plotly_dark')

    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Mortality Rate'],

                             mode='lines+markers',

                             name='Taux de mortalité',

                             line=dict(color='red', width=2)))



    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Recovery Rate'],

                             mode='lines+markers',

                             name='Taux de guérison',

                             line=dict(color='Green', width=2)))



    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Growth Rate'],

                             mode='lines+markers',

                             name='Taux de contamination',

                             line=dict(color='Yellow', width=2)))

    fig.update_layout(yaxis=dict(tickformat=".2%"))

    

    return fig
plot_diff_curve(Africa_df).show()
fig = go.Figure()

fig.update_layout(template='plotly_dark')



tmp_df = Africa_df.copy()

tmp_df = tmp_df[tmp_df['Growth Rate Accel'] < 10]



fig.add_trace(go.Scatter(x=tmp_df['Date'], 

                         y=tmp_df['Growth Rate Accel'],

                         mode='lines+markers',

                         name='accélération de la cr',

                         line=dict(color='Green', width=3)))

fig.update_layout(yaxis=dict(tickformat=".2%"))



fig.show()
confirmed_by_country_df = complet_df.groupby(['Date','Country']).sum().reset_index()
fig = px.line(confirmed_by_country_df, x='Date', y='Confirmed', color='Country', line_group="Country", hover_name="Country")

fig.update_layout(template='plotly_dark')

fig.show()
#échelle logarithmique. toute tendance linéaire implique une rapide croissance du nomdre de cas confirmés sur la période dans le 



fig = px.line(confirmed_by_country_df, x='Date', y='Confirmed', color='Country', line_group="Country", hover_name="Country")

fig.update_layout(

    template='plotly_dark',

    yaxis_type="log"

)

fig.show()
confirmed_by_country_df.groupby('Country').max().sort_values(by='Confirmed', ascending=False)[:10]
k_layout_kwargs = {

    'font': dict(size=12,),

    'legend': dict(x=0, y=-0.7),

}
Togo_df = confirmed_by_country_df[confirmed_by_country_df['Country'] == 'Togo'].copy()

Togo_df = ajout_taux(Togo_df)

tmp_df = Togo_df[Togo_df['Confirmed'] > 0]

plot_aggregate_curve(tmp_df).show()
plot_diff_curve(tmp_df).show()

Algeria_df = confirmed_by_country_df[confirmed_by_country_df['Country'] == 'Algeria'].copy()

Algeria_df = ajout_taux(Algeria_df)

tmp_df1 = Algeria_df[Algeria_df['Confirmed'] > 0]

plot_aggregate_curve(tmp_df1).show()
plot_diff_curve(tmp_df1).show()
from fbprophet.plot import plot_plotly

from fbprophet import Prophet

from fbprophet.plot import add_changepoints_to_plot
total_pop = 7706000



floor_model = lambda x: max(x - 1, 0) 

cap_model = lambda x: round(min(total_pop, 1.5 * x + 1000))



# Modeling Togo confirmed cases 

confirmed_training_df = confirmed_by_country_df[(confirmed_by_country_df['Country'] == 'Togo') & (confirmed_by_country_df['Confirmed'] > 0)]

confirmed_training_df = confirmed_training_df.rename(columns={'Date': 'ds', 'Confirmed': 'y'}).reset_index(drop=True)



confirmed_training_df['floor'] = confirmed_training_df.y.apply(floor_model)

confirmed_training_df['cap'] = confirmed_training_df.y.apply(cap_model)
confirmed_training_df.y = confirmed_training_df.y.apply(np.log10)

confirmed_training_df.floor = confirmed_training_df.floor.apply(np.log10)

confirmed_training_df.cap = confirmed_training_df.cap.apply(np.log10)

# Total confirmed model 

m = Prophet(

    growth='linear',

    interval_width=0.90,

    changepoint_prior_scale=0.5,

    changepoint_range=0.7,

    yearly_seasonality=False,

    weekly_seasonality=True,

    daily_seasonality=True,

    #n_changepoints=1

)

m.fit(confirmed_training_df)

future = m.make_future_dataframe(periods=14)

future['floor'] = confirmed_training_df.floor

future['cap'] = confirmed_training_df.cap

confirmed_forecast = m.predict(future)
for kwargs in [{}, {"yaxis_type": "log"}]:

    fig = plot_plotly(m, confirmed_forecast, plot_cap=False, changepoints=True)

    annotations = []

    annotations.append(dict(

        xref='paper',

        yref='paper',

        x=0.0,

        y=1.15,

        xanchor='left',

        yanchor='bottom',

        text='Predictions for log10 Confirmed cases Togo',

        font=dict(

            family='Arial',

            size=30,

            color='rgb(37,37,37)'),

        showarrow=False))

    fig.update_layout(

        annotations=annotations,

        **kwargs

    )

    fig.show()
total_pop = 43820839



floor_model = lambda x: max(x - 1, 0) 

cap_model = lambda x: round(min(total_pop, 1.5 * x + 1000))



# Modeling Algeria confirmed cases 

confirmed_training_dfa = confirmed_by_country_df[(confirmed_by_country_df['Country'] == 'Algeria') & (confirmed_by_country_df['Confirmed'] > 0)]

confirmed_training_dfa = confirmed_training_dfa.rename(columns={'Date': 'ds', 'Confirmed': 'y'}).reset_index(drop=True)



confirmed_training_dfa['floor'] = confirmed_training_dfa.y.apply(floor_model)

confirmed_training_dfa['cap'] = confirmed_training_dfa.y.apply(cap_model)
confirmed_training_dfa.y = confirmed_training_dfa.y.apply(np.log10)

confirmed_training_dfa.floor = confirmed_training_dfa.floor.apply(np.log10)

confirmed_training_dfa.cap = confirmed_training_dfa.cap.apply(np.log10)
# modelisation du total confirmé

m = Prophet(

    growth='linear',

    interval_width=0.97,

    changepoint_prior_scale=0.5,

    changepoint_range=0.3,

    yearly_seasonality=False,

    weekly_seasonality=True,

    daily_seasonality=True,

    #n_changepoints=0

)

m.fit(confirmed_training_dfa)

future1 = m.make_future_dataframe(periods=14)

future1['floor'] = confirmed_training_dfa.floor

future1['cap'] = confirmed_training_dfa.cap

confirmed_forecasta = m.predict(future1)
for kwargs in [{}, {"yaxis_type": "log"}]:

    fig = plot_plotly(m, confirmed_forecasta, plot_cap=False, changepoints=True)

    annotations = []

    annotations.append(dict(

        xref='paper',

        yref='paper',

        x=0.0,

        y=1.15,

        xanchor='left',

        yanchor='bottom',

        text='Predictions for log10 Confirmed cases Algeria',

        font=dict(

            family='Arial',

            size=30,

            color='rgb(37,37,37)'),

        showarrow=False))

    fig.update_layout(

        annotations=annotations,

        **kwargs

    )

    

    fig.show()