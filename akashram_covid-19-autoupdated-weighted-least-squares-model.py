import numpy as np 

import pandas as pd

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Visualisation libraries



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

import pycountry

import plotly.express as px

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objs as go

import plotly.offline as py

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

import plotly.io as pls





py.init_notebook_mode(connected=True)

import folium 

from folium import plugins

plt.style.use("fivethirtyeight")



plt.rcParams['figure.figsize'] = 8, 5
data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")



print(f"Dataset is of {data.shape[0]} rows and {data.shape[1]} columns")
#open = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")

# ll =  pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")
data.head()
data.ObservationDate = pd.to_datetime(data.ObservationDate)



data['Last Update'] = pd.to_datetime(data['Last Update'])
data['Total cases'] = data['Confirmed']



data['Active cases'] = data['Total cases'] - (data['Recovered'] + data['Deaths'])



print(f'Total number of Confirmed cases:', data['Total cases'].sum())



print(f'Total number of Active cases:', data['Active cases'].sum())
grouped_conf = data.groupby('ObservationDate')['Confirmed'].sum().reset_index()



grouped_Act = data.groupby('ObservationDate')['Active cases'].sum().reset_index()
pls.templates.default = "plotly_dark"



fig = px.line(grouped_conf, x="ObservationDate", y="Confirmed", 

              title="Worldwide Confirmed Cases Over Time")

fig.show()



fig = px.line(grouped_Act, x="ObservationDate", y="Active cases", 

              title="Worldwide Active Cases Over Time")



fig.show()
cd_group = data.groupby(['Country/Region','ObservationDate'])['Confirmed'].sum().reset_index()

cd_act = data.groupby(['Country/Region', 'ObservationDate'])['Active cases'].sum().reset_index()



ch_data = data[data['Country/Region'].str.contains("China")] 

ch_data = ch_data.groupby(['ObservationDate'])['Active cases'].sum().reset_index()



row_data = data[~data['Country/Region'].str.contains("China")] 

row_data = row_data.groupby(['ObservationDate'])['Active cases'].sum().reset_index()
row_data['ObservationDate'] = row_data['ObservationDate'].dt.date

ch_data['ObservationDate'] = ch_data['ObservationDate'].dt.date



from datetime import date

today = date.today()
lrow = row_data.tail(1)

lc =  ch_data.tail(1)

all_act = lc.merge(lrow, on='ObservationDate')

all_act.columns = ['Date', 'China', 'Rest-of-the-World']
pls.templates.default = "ggplot2"



fig = px.line(ch_data, x="ObservationDate", y="Active cases", 

              title="Active Cases in CHINA Over Time")

fig.show()
pls.templates.default = "seaborn"



fig = px.line(row_data, x="ObservationDate", y="Active cases", 

              title="Active Cases in Rest of the world Over Time")

fig.show()
colors = ['#7FB3D5', '#D5937F']



China = all_act['China'].sum()

RoW =  all_act['Rest-of-the-World'].sum()



fig = go.Figure(data=[go.Pie(labels=['China','Rest of the World'],

                             values= [China,RoW],hole =.3)])

                          

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)))



fig.show()
ggdf = data.groupby(['ObservationDate', 'Country/Region'])['Confirmed'].max()

ggdf = ggdf.reset_index()



ggdf['ObservationDate'] = pd.to_datetime(ggdf['ObservationDate'])

ggdf['date'] = ggdf['ObservationDate'].dt.strftime('%m/%d/%Y')

ggdf['size'] = ggdf['Confirmed'].pow(0.4)



fig = px.scatter_geo(ggdf, locations="Country/Region", locationmode='country names', 

                     color="Confirmed",  size='size', hover_name="Country/Region", 

                     range_color= [0, 1500], 

                     projection="natural earth", animation_frame="date", 

                     title='COVID-19 Confirmed Cases Spread Over Time Globally', color_continuous_scale=px.colors.sequential.Viridis)



fig.show()
ggdf_a = data.groupby(['ObservationDate', 'Country/Region'])['Active cases'].max()

ggdf_a = ggdf_a.reset_index()



ggdf_a['ObservationDate'] = pd.to_datetime(ggdf['ObservationDate'])

ggdf_a['date'] = ggdf_a['ObservationDate'].dt.strftime('%m/%d/%Y')

ggdf_a['size'] = ggdf_a['Active cases'].pow(0.4)



fig = px.scatter_geo(ggdf_a, locations="Country/Region", locationmode='country names', 

                     color="Active cases",  size='size', hover_name="Country/Region", 

                     range_color= [0, 1500], 

                     projection="natural earth", animation_frame="date", 

                     title='COVID-19 Active Cases Spread Over Time Globally', color_continuous_scale="Inferno")



fig.show()
# We model only the active cases



da =  data.groupby(['Country/Region','ObservationDate'])['Active cases'].sum().reset_index()

# data.groupby(['Country/Region'])['Active cases'].sum().sort_values(ascending=False).head(20).reset_index()



da_last_day = da[da['ObservationDate'] > '2020-02-20']

daa = da_last_day.sort_values(by='Active cases', ascending=False)
countries = ['Mainland China', 'Italy', 'South Korea','Iran', 'Spain', 'Germany' , 'France', 'US'

, 'Switzerland', 'UK','Netherlands', 'Norway','Japan', 'Singapore', 'India', 'Canada']



most_affected_countries = daa[daa['Country/Region'].isin(countries)]
# t_countries = ['Mainland China', 'Italy', 'South Korea','Singapore', 'India']

# most_affected_countries = daa[daa['Country/Region'].isin(t_countries)]
fig = px.line(most_affected_countries, x="ObservationDate", y="Active cases", color='Country/Region')



fig.show()
fig = px.line(most_affected_countries, x="ObservationDate", y="Active cases", color='Country/Region')

fig.update_layout(yaxis_type="log")



fig.show()
## The windows that we use are weighted: we give more weight to the last day, and less to the days further away in time.



import numpy as np



def ramp_window(start=14, middle=7):

    window = np.ones(start)

    window[:middle] = np.arange(middle) / float(middle)

    window /= window.sum()

    return window



def exp_window(start=14, growth=1.1):

    window = growth ** np.arange(start)

    window /= window.sum()

    return window
window_size = 15

weighted_window = exp_window(start=window_size, growth=1.6)
import plotly.graph_objects as go

fig = go.Figure(data=go.Scatter(x=weighted_window))

fig.show()
# We model only the active cases



da_lf =  data.groupby(['Country/Region','ObservationDate'])['Active cases'].sum().reset_index()

# data.groupby(['Country/Region'])['Active cases'].sum().sort_values(ascending=False).head(20).reset_index()



dalf_last_day = da_lf[da_lf['ObservationDate'] > '2020-03-05']

da_lff = dalf_last_day.sort_values(by='Active cases', ascending=False)
countries = ['Mainland China', 'Italy', 'South Korea','Iran', 'Spain', 'Germany' , 'France', 'US'

, 'Switzerland', 'UK','Netherlands', 'Norway','Japan', 'Singapore', 'India', 'Canada']



most_affected_countries_last_ft = da_lff[da_lff['Country/Region'].isin(countries)]
fig = px.line(most_affected_countries_last_ft, x="ObservationDate", y="Active cases", color='Country/Region')

fig.update_layout(yaxis_type="log")

fig.show()
newf = most_affected_countries_last_ft.pivot(index='ObservationDate', columns='Country/Region')
import statsmodels.api as sm



def fit_on_window(data, window):

    """ Fit the last window of the data

    """

    window_size = len(window)

    last_fortnight = data.iloc[-window_size:]

    log_last_fortnight = np.log(last_fortnight)

    log_last_fortnight[log_last_fortnight == -np.inf] = 0



    design = pd.DataFrame({'linear': np.arange(window_size),

                           'const': np.ones(window_size)})



    growth_rate = pd.DataFrame(data=np.zeros((1, len(data.columns))),

                               columns=data.columns)



    predicted_cases = pd.DataFrame()

    predicted_cases_lower = pd.DataFrame()

    predicted_cases_upper = pd.DataFrame()

    prediction_dates = pd.date_range(data.index[-window_size],

                                    periods=window_size + 7)

    

    

    for country in data.columns:

        mod_wls = sm.WLS(log_last_fortnight[country].values, design,

                         weights=window, hasconst=True)

        res_wls = mod_wls.fit()

        growth_rate[country] = np.exp(res_wls.params.linear)

        predicted_cases[country] = np.exp(res_wls.params.const +

                res_wls.params.linear * np.arange(len(prediction_dates))

            )

        

        # 1st and 3rd quartiles in the confidence intervals

        conf_int = res_wls.conf_int(alpha=.25)

        

        

        predicted_cases_lower[country] = np.exp(res_wls.params.const +

                conf_int[0].linear * np.arange(len(prediction_dates))

            )

        predicted_cases_upper[country] = np.exp(res_wls.params.const +

                conf_int[1].linear * np.arange(len(prediction_dates))

            )



    predicted_cases = pd.concat(dict(prediction=predicted_cases,

                                     lower_bound=predicted_cases_lower,

                                     upper_bound=predicted_cases_upper),

                                axis=1)

    predicted_cases['date'] = prediction_dates

    predicted_cases = predicted_cases.set_index('date')

    if window_size > 10:

        predicted_cases  = predicted_cases.iloc[window_size - 10:]

        

    return growth_rate, predicted_cases
growth_rate, predicted_cases = fit_on_window(newf, weighted_window)
pls.templates.default = "plotly_dark"

grt = growth_rate.T.sort_values(by=0).reset_index()

grt.columns = ['level', 'country', 'growth_rate']



fig = px.bar(grt, x="country", y="growth_rate", orientation='h')

fig.show()
mt = most_affected_countries_last_ft[most_affected_countries_last_ft['ObservationDate'] > '2020-03-10']
pcs = predicted_cases.stack().reset_index()



pcs = pcs[pcs['date'] > '2020-03-21']



pcs.columns = ['date', 'country', 'lower_bound', 'prediction', 'upper_bound']
fig = px.line(mt, x="ObservationDate", y="Active cases", color='Country/Region')



fig.update_layout(yaxis_type="log")



fig.show()
fig = px.line(pcs, x="date", y="prediction", color='country')



fig.update_layout(yaxis_type="log")



fig.show()
#plt.figure(figsize=(20,12))

#ax = sns.lineplot(x="date", y="prediction", data=pcs, hue='country')

#ax = sns.lineplot(x="date", y="prediction", data=pcs, hue='country')

#plt.legend(loc=(.8, -.6))

#ax.set_yscale('log')

#ax.set_title('Prediction of active cases in major countries + lower and upper bounds')