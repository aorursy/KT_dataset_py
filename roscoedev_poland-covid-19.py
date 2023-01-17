import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures



pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style("whitegrid")



from plotly.offline import init_notebook_mode, iplot, plot

import plotly.express as px

import plotly.graph_objs as go



import datetime



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

poland_covid_19_summary = pd.read_csv('../input/poland-covid19-summary/poland_covid-19_summary.csv', index_col='date', parse_dates=True)

poland_covid_19_summary['active'] = poland_covid_19_summary.sick - (poland_covid_19_summary.dead + poland_covid_19_summary.cured)

poland_covid_19_summary.head()
poland_covid_19_summary.tail()
poland_covid_19_summary["change"] = poland_covid_19_summary.active.diff()

poland_covid_19_summary["growth"] = poland_covid_19_summary.active.div(other=poland_covid_19_summary.active.shift(1))



poland_covid_19_summary.tail()
poland_covid_19_tests = pd.read_csv('../input/polandcovid19tests/poland_covid-19_tests.csv', index_col='date', parse_dates=True)



tests_scatter = go.Scatter(

    x = poland_covid_19_tests.index,

    y = poland_covid_19_tests.tests,

    mode = "lines",

    name = "Testy lacznie",

    marker = dict(color = 'green'))



data = [tests_scatter]

layout = dict(title = 'Statystyka testow COVID-19 w Polsce',

              xaxis= dict(title= 'Data',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Testy',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)

pass
poland_covid_19_tests['factor'] = poland_covid_19_tests.tests / poland_covid_19_tests.positive

poland_covid_19_tests
poland_covid_19_summary.rename(columns={'sick': 'Chorzy', 'dead': 'Zmarli', 'cured': 'Wyleczeni'}, inplace=True)



sick_scatter = go.Scatter(

    x = poland_covid_19_summary.index,

    y = poland_covid_19_summary.Chorzy,

    mode = "lines",

    name = "Wszystkie przypadki",

    marker = dict(color = 'blue'))



active_scatter = go.Scatter(

    x = poland_covid_19_summary.index,

    y = poland_covid_19_summary.active,

    mode = "lines",

    name = "Aktywne przypadki",

    marker = dict(color = 'orange'))



dead_scatter = go.Scatter(

    x = poland_covid_19_summary.index,

    y = poland_covid_19_summary.Zmarli,

    mode = "lines",

    name = "Zmarli",

    marker = dict(color = 'red'))



cured_scatter = go.Scatter(

    x = poland_covid_19_summary.index,

    y = poland_covid_19_summary.Wyleczeni,

    mode = "lines",

    name = "Wyleczeni",

    marker = dict(color = 'green'))



#plt.figure(figsize=(16,8))

#sns.lineplot(data=poland_covid_19_summary).set(title = 'COVID-19 - Podsumowanie', xlabel = 'Data', ylabel = 'Ludzie' )



data = [sick_scatter, active_scatter, dead_scatter, cured_scatter]

layout = dict(title = 'Statystyka COVID-19 w Polsce',

              xaxis= dict(title= 'Data',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Ludzie',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)

pass
tests_scatter_normalized = go.Scatter(

    x = poland_covid_19_tests.index,

    y = poland_covid_19_tests.tests / max(poland_covid_19_tests.tests),

    mode = "lines",

    name = "Testy lacznie",

    marker = dict(color = 'green'))



sick_scatter_normalized = go.Scatter(

    x = poland_covid_19_summary.index,

    y = poland_covid_19_summary.Chorzy / max(poland_covid_19_summary.Chorzy),

    mode = "lines",

    name = "Wszystkie przypadki",

    marker = dict(color = 'blue'))



data = [sick_scatter_normalized, tests_scatter_normalized]

layout = dict(title = 'Znormalizowane porownanie testow i potwierdzonych przypadkow COVID-19 w Polsce',

              xaxis= dict(title= 'Data',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Ludzie',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)

pass
poland_covid_19_tests_diff = poland_covid_19_tests.diff()



tests_daily = {

  'x': poland_covid_19_tests_diff.index,

  'y': poland_covid_19_tests_diff.tests,

  'name': 'Testy',

  'type': 'bar',

  'marker': dict(color = 'blue')

};



data = [tests_daily];

layout = {

  'xaxis': {'title': 'Data'},

  'barmode': 'relative',

  'title': 'Ilosc testow dziennie'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)



pass
data = [sick_scatter, active_scatter]

layout = dict(title = 'Statystyka COVID-19 w Polsce (log)',

              xaxis= dict(title= 'Data',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Ludzie (log)',ticklen= 5,zeroline= False),

              yaxis_type='log'

             )

fig = dict(data = data, layout = layout)

iplot(fig)

pass
data = [dead_scatter, cured_scatter]

layout = dict(title = 'Statystyka COVID-19 w Polsce (Wyleczeni vs. Zgony)',

              xaxis= dict(title= 'Data',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Ludzie',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)

pass
resp_scatter = go.Scatter(

    x = poland_covid_19_summary.index,

    y = [10000] * len(poland_covid_19_summary.index),

    mode = "lines",

    name = "Respiratory",

    marker = dict(color = 'black'),

    line = dict(dash='dash'))



sick_hard_scatter = go.Scatter(

    x = poland_covid_19_summary.index,

    y = poland_covid_19_summary.Chorzy * 0.12,

    mode = "lines",

    name = "Ciezko chorzy",

    marker = dict(color = 'navy'))



sick_light_scatter = go.Scatter(

    x = poland_covid_19_summary.index,

    y = poland_covid_19_summary.Chorzy * 0.88,

    mode = "lines",

    name = "Lekko chorzy",

    marker = dict(color = 'blue'))



data = [sick_hard_scatter, sick_light_scatter, resp_scatter]

layout = dict(title = 'Statystyka COVID-19 w Polsce (Chorzy i ciezko chorzy)',

              xaxis= dict(title= 'Data',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Ludzie',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)

pass
sick_trace = {

  'x': poland_covid_19_summary.index,

  'y': poland_covid_19_summary.active,

  'name': 'Chorzy',

  'type': 'bar',

  'marker': dict(color = 'blue')

};



dead_trace = {

  'x': poland_covid_19_summary.index,

  'y': poland_covid_19_summary.Zmarli,

  'name': 'Zmarli',

  'type': 'bar',

  'marker': dict(color = 'red')

};



cured_trace = {

  'x': poland_covid_19_summary.index,

  'y': poland_covid_19_summary.Wyleczeni,

  'name': 'Wyleczeni',

  'type': 'bar',

  'marker': dict(color = 'green')

};



data = [dead_trace, cured_trace, sick_trace];

layout = {

  'xaxis': {'title': 'Data'},

  'barmode': 'relative',

  'title': 'Statystyka COVID-19 w Polsce'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)



pass
poland_covid_19_summary_diff = poland_covid_19_summary.diff()



sick_diff_scatter = go.Scatter(

    x = poland_covid_19_summary_diff.index,

    y = poland_covid_19_summary_diff.active,

    mode = "lines",

    name = "Chorzy",

    marker = dict(color = 'blue'))



dead_diff_scatter = go.Scatter(

    x = poland_covid_19_summary_diff.index,

    y = poland_covid_19_summary_diff.Zmarli,

    mode = "lines",

    name = "Zmarli",

    marker = dict(color = 'red'))



cured_diff_scatter = go.Scatter(

    x = poland_covid_19_summary_diff.index,

    y = poland_covid_19_summary_diff.Wyleczeni,

    mode = "lines",

    name = "Wyleczeni",

    marker = dict(color = 'green'))



data = [sick_diff_scatter, dead_diff_scatter, cured_diff_scatter]

layout = dict(title = 'COVID-19 w Polsce - Zmiana dzienna',

              xaxis= dict(title= 'Data',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Ludzie',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)

pass
growth_scatter = go.Scatter(

    x = poland_covid_19_summary.index[4:],

    y = poland_covid_19_summary.growth[4:],

    mode = "lines",

    name = "Wskaznik wzrostu",

    marker = dict(color = 'green'))



data = [growth_scatter]

layout = dict(title = 'Wskaznik wzrostu zachorowan',

              xaxis= dict(title= 'Data',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Wskaznik wzrostu',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)

pass
from fbprophet import Prophet



df_sick = pd.DataFrame([poland_covid_19_summary.Chorzy]).transpose()

df_sick.reset_index(inplace=True)

df_sick = df_sick.rename(columns={'Chorzy': 'y', 'date': 'ds'})







m = Prophet()

m.fit(df_sick)



future_sick = m.make_future_dataframe(periods=10)

forecast_sick = m.predict(future_sick)



fig1 = m.plot(forecast_sick)
fig2 = m.plot_components(forecast_sick)

def add_dates(df, days_column, start_date='2020-03-03'):

    """Addd 'Dates' column to df."""

    start_date = pd.to_datetime(start_date)

    

    df['Date'] = df.apply(lambda x: start_date + pd.DateOffset(days=int(x[days_column])), axis=1)



pass
NUM_OF_DAYES = 60

ORDER = 1
from scipy.interpolate import InterpolatedUnivariateSpline



x = np.linspace(0, NUM_OF_DAYES, NUM_OF_DAYES + 1)

s = InterpolatedUnivariateSpline(range(0, len(df_sick.y)), df_sick.y, k=ORDER)

y = s(x)



sick_interpolated = pd.DataFrame(y, x, columns=['sick'])

sick_interpolated.index.name = 'day'

sick_interpolated.reset_index(inplace=True)



add_dates(sick_interpolated, 'day')



pass
PERIODS = 10
from fbprophet import Prophet



df_dead = pd.DataFrame([poland_covid_19_summary.Zmarli]).transpose()

df_dead.reset_index(inplace=True)

df_dead = df_dead.rename(columns={'Zmarli': 'y', 'date': 'ds'})



m = Prophet()

m.fit(df_dead)



future_dead = m.make_future_dataframe(periods=PERIODS)

forecast_dead = m.predict(future_dead)



fig3 = m.plot(forecast_dead)
fig4 = m.plot_components(forecast_dead)
from scipy.interpolate import InterpolatedUnivariateSpline



x = np.linspace(0, NUM_OF_DAYES, NUM_OF_DAYES + 1)

s = InterpolatedUnivariateSpline(range(0, len(df_dead.y)), df_dead.y, k=ORDER)

y = s(x)



dead_interpolated = pd.DataFrame(y, x, columns=['dead'])

dead_interpolated.index.name = 'day'

dead_interpolated.reset_index(inplace=True)



add_dates(dead_interpolated, 'day')



pass
sick_prediction = go.Scatter(

    x = sick_interpolated.Date,

    y = sick_interpolated.sick,

    mode = "lines",

    name = "Chorzy",

    marker = dict(color = 'blue'))



dead_prediction = go.Scatter(

    x = dead_interpolated.Date,

    y = dead_interpolated.dead,

    mode = "lines",

    name = "Zgony",

    marker = dict(color = 'red'))



data = [sick_prediction, dead_prediction]

layout = dict(title = 'COVID-19 w Polsce - Prognoza',

              xaxis= dict(title= 'Data',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Ludzie',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
poland_covid_19_summary.insert(0, 'Dzien', range(0, len(poland_covid_19_summary)))
sick_model = LinearRegression(fit_intercept=True)

poly = PolynomialFeatures(degree=8)

num_days_poly = poly.fit_transform(poland_covid_19_summary.Dzien.values.reshape(-1,1))

poly_reg = sick_model.fit(num_days_poly, poland_covid_19_summary.active.values.reshape(-1,1))

predictions_for_given_days = sick_model.predict(num_days_poly)



print("coef_ :",sick_model.coef_,"intercept_:",sick_model.intercept_)

pass
today = poland_covid_19_summary.Dzien.iloc[-1]

print(f'Today is {today} day of pandemic in Poland')



tomorrow_value = today



for i in range(0, 2):

    tomorrow_value += 1



    value_prediction = poly.fit_transform(np.array([[tomorrow_value]]))

    prediction = sick_model.predict(value_prediction)

    print(f'Prediction for day number {tomorrow_value} : {prediction} cases ')

dead_model = LinearRegression(fit_intercept=True)

poly = PolynomialFeatures(degree=8)

num_days_poly = poly.fit_transform(poland_covid_19_summary.Dzien.values.reshape(-1,1))

poly_reg = dead_model.fit(num_days_poly, poland_covid_19_summary.Zmarli.values.reshape(-1,1))

predictions_for_given_days = dead_model.predict(num_days_poly)



print("coef_ :",dead_model.coef_,"intercept_:",dead_model.intercept_)

pass
today = poland_covid_19_summary.Dzien.iloc[-1]

print(f'Today is {today} day of pandemic in Poland')



tomorrow_value = today



for i in range(0, 2):

    tomorrow_value += 1

    value_prediction = poly.fit_transform(np.array([[tomorrow_value]]))

    prediction = dead_model.predict(value_prediction)

    print(f'Prediction for day number {tomorrow_value} : {prediction} dead ')
import math

def model(N, a, alpha, t):

    # we enforce N, a and alpha to be positive numbers using min and max functions

    return max(N, 0) * (1 - math.e ** (min(-a, 0) * t)) ** max(alpha, 0)
def model_loss(params):

    N, a, alpha = params

    model_x = []

    r = 0

    for t in range(len(poland_covid_19_summary)):

        r += (model(N, a, alpha, t) - poland_covid_19_summary.active.iloc[t]) ** 2

#         print(model(N, a, alpha, t), df.iloc[t, 0])

    return r 
import numpy as np

from scipy.optimize import minimize

opt = minimize(model_loss, x0=np.array([200000, 0.1, 15]), method='Nelder-Mead', tol=1e-5).x

opt
model_x = []

for t in range(len(poland_covid_19_summary)):

    model_x.append([poland_covid_19_summary.index[t], model(*opt, t)])

model_sim = pd.DataFrame(model_x, dtype=int)

model_sim.set_index(0, inplace=True)

model_sim.columns = ['Model']

pd.concat([poland_covid_19_summary, model_sim], axis=1)



model_sick_scatter = go.Scatter(

    x = model_sim.index,

    y = model_sim.Model,

    mode = "lines",

    name = "Chorzy (model)",

    marker = dict(color = 'red'))



data = [active_scatter, model_sick_scatter]

layout = dict(title = 'Model vs Rzeczywiste Dane COVID-19 w Polsce',

              xaxis= dict(title= 'Data',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Ludzie',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)

pass
import datetime

start_date = poland_covid_19_summary.index[0]

n_days = 110

extended_model_x = []

for t in range(n_days):

    extended_model_x.append([start_date + datetime.timedelta(days=t), model(*opt, t)])

extended_model_sim = pd.DataFrame(extended_model_x, dtype=int)

extended_model_sim.set_index(0, inplace=True)

extended_model_sim.columns = ['Model']

pd.concat([extended_model_sim, poland_covid_19_summary], axis=1)



model_extended_sick_scatter = go.Scatter(

    x = extended_model_sim.index,

    y = extended_model_sim.Model,

    mode = "lines",

    name = "Chorzy (model)",

    marker = dict(color = 'red'))



beds_scatter = go.Scatter(

    x = extended_model_sim.index,

    y = [10000] * len(extended_model_sim.index),

    mode = "lines",

    name = "Respiratory",

    marker = dict(color = 'black'),

    line = dict(dash='dot'))



data = [active_scatter, model_extended_sick_scatter, beds_scatter]

layout = dict(title = 'Model dlugoterminowy COVID-19 w Polsce',

              xaxis= dict(title= 'Data',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Ludzie',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)

pass
model_extended_sick_hard_scatter = go.Scatter(

    x = extended_model_sim.index,

    y = extended_model_sim.Model * 0.12,

    mode = "lines",

    name = "Ciezko chorzy (model)",

    marker = dict(color = 'red'))



resp_scatter = go.Scatter(

    x = extended_model_sim.index,

    y = [10000] * len(extended_model_sim.index),

    mode = "lines",

    name = "Respiratory",

    marker = dict(color = 'black'),

    line = dict(dash='dash'))



data = [resp_scatter, model_extended_sick_hard_scatter]

layout = dict(title = 'Model ciezko chorych COVID-19 w Polsce',

              xaxis= dict(title= 'Data',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Ludzie',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)

pass
extended_model_sim[70:110]