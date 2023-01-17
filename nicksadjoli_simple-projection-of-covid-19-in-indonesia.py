import numpy as np

import pandas as pd

import time

from datetime import date

%matplotlib inline



date_today = date.today()
#global_confirmed_dataset  = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv', index_col='Country/Region')

#global_deaths_dataset     = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv', index_col='Country/Region')



global_confirmed_dataset  = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv', index_col='Country/Region')

global_deaths_dataset     = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv', index_col='Country/Region')



pd.set_option('display.max_columns', None)
global_confirmed_dataset.loc[['Indonesia']] 
global_deaths_dataset.loc[['Indonesia']]
print("Today's date:",date_today.strftime("%m/%d/%Y"))

latest_id_gov_data = pd.read_json("https://opendata.arcgis.com/datasets/db3be1cadcf44b6fa709274c12726c59_0.geojson")['features']

days_recorded = len(latest_id_gov_data) - 1

latest_id_confirmed, latest_id_deaths, latest_id_recovered = None, None, None



#len(latest_id_gov_data)#[0]['properties']



for i in range(days_recorded, 0, -1):

    data_date = latest_id_gov_data[i]['properties']['Tanggal'].split('T')[0]

    if str(date_today) == str(data_date):

        latest_id_confirmed_val = latest_id_gov_data[i]['properties']['Jumlah_Kasus_Kumulatif']

        latest_id_deaths_val = latest_id_gov_data[i]['properties']['Jumlah_Pasien_Meninggal']

        latest_id_recovered_val = latest_id_gov_data[i]['properties']['Jumlah_Pasien_Sembuh'] 

        break



#get today's date, but match the date format used by John Hopkin's data

today_date_key = date_today.strftime("%m/%d/%Y")[1:-2]

latest_id_confirmed_data = pd.Series([latest_id_confirmed_val], index=[today_date_key])

latest_id_deaths_data = pd.Series([latest_id_deaths_val], index=[today_date_key])



if (latest_id_confirmed_val is None) or (latest_id_deaths_val is None) or (latest_id_recovered_val is None):

    print("Latest {} data from Indonesia Government's website (http://covid19.bnpb.go.id/) NOT YET AVAILABLE".format(date_today.strftime("%m/%d/%Y")))

    indonesia_confirmed_data = global_confirmed_dataset.loc['Indonesia', '3/1/20':]

    indonesia_deaths_data = global_deaths_dataset.loc['Indonesia', '3/1/20' :]

else:

    print("Latest {} data from Indonesia Government's website (http://covid19.bnpb.go.id/) IS AVAILABLE".format(date_today.strftime("%m/%d/%Y")))

    indonesia_confirmed_data = global_confirmed_dataset.loc['Indonesia', '3/1/20':].append(latest_id_confirmed_data)

    indonesia_deaths_data = global_deaths_dataset.loc['Indonesia', '3/1/20' :].append(latest_id_deaths_data)



print()

print("Dataset of Indonesian confirmed cases, imputed with latest data pulled from Indonesian Governemnt's website")

print(indonesia_confirmed_data)

from scipy.optimize import curve_fit

import datetime

import plotly.offline as plotly_off

from plotly.offline import init_notebook_mode, iplot, plot

import plotly.graph_objs as go

from sklearn.metrics import mean_squared_error



def richard_curve(t, K, r, alpha, tm):

    return K / ((1 + alpha * np.exp(-r * (t - tm)))**(1/alpha))



def plotly_graph(plot_data, title, height=700, width=None, case='Confirmed'):

    if width is None:

        layout = go.Layout(

                    title= title,

                    plot_bgcolor = 'rgb(229, 229, 229)',

                    xaxis = dict(title= f'Days after 1 March 2020', ticklen= 10, zeroline= False),

                    yaxis = dict(title= 'Number of {}'.format(case),ticklen= 10,zeroline= False),

                    height = height,

                    legend = dict(x=-.1, y=-.4)

                    )

    else:

        layout = go.Layout(

                    title = title,

                    plot_bgcolor = 'rgb(229, 229, 229)',

                    xaxis = dict(title= f'Days after 1 March 2020', ticklen= 10, zeroline= False),

                    yaxis = dict(title= 'Number of {}'.format(case),ticklen= 10,zeroline= False),

                    height = height,

                    width = width,

                    legend = dict(x=-.1, y=-.4)

                    )

    fig = go.Figure(data=plot_data, layout=layout)

    fig.show()



who_confirmed_dataset = pd.read_csv('../input/total-cases-covid-19-who.csv', index_col='Entity')

who_id_confirmed_data = who_confirmed_dataset.loc[['Indonesia']]

who_id_confirmed_data_day = who_id_confirmed_data['Year'][61:].values

who_id_confirmed_march_data = who_id_confirmed_data['Total confirmed cases of COVID-19 (cases)'][61:].values



#Note that WHO seem to only record cumulative data if there is an increase. Hence, adjustments to this dataset are required to fill in the 'missing' days

print("WHO record of confirmed Indonesian COVID-19 in March 2020:")

print(who_id_confirmed_data[61:])



#days_length = who_id_confirmed_data_day[-1] - who_id_confirmed_data_day[0] 



t_itb = who_id_confirmed_data_day - who_id_confirmed_data_day[0]

y_confirmed_itb = who_id_confirmed_march_data





#parameter values that are used by the ITB research team

richard_curve_itb_param = (8495, 0.2, 0.410, 40.12)



#Fitting into the data that seem to be used by the ITB research

popt_confirmed_itb, pcov_confirmed_itb = curve_fit(richard_curve, t_itb[:8], y_confirmed_itb[:8], \

                                                   p0=richard_curve_itb_param) 

projection_times_itb = t_itb

projection_confirmed_itb = richard_curve(projection_times_itb-t_itb[0], *popt_confirmed_itb) 



trace_confirmed_cur_itb = go.Scatter(

                      x=t_itb[2:8],

                      y=y_confirmed_itb[2:8],

                      mode='lines',

                      marker=go.scatter.Marker(color='rgb(255, 127, 14)',

                                      size=7),

                      name='Current Confirmed'

                      )



trace_confirmed_itb_proj = go.Scatter(

                      x=projection_times_itb[2:8],#projection_times_itb[35:81],

                      y=projection_confirmed_itb[2:8] - ( projection_confirmed_itb[2] - 2),

                      mode='lines',

                      marker=go.scatter.Marker(color='rgb(10, 250, 50)',

                                      size=7),

                      name='Projected Confirmed Cases (ITB model)'

                      )



plot_data = [trace_confirmed_cur_itb, trace_confirmed_itb_proj] 

plotly_graph(plot_data, 'Recreation of ITB Research\'s Richard\'s Curve Fit for Indonesian COVID-19', case='Confirmed')



print("Average RMSE obtained:", np.sqrt(mean_squared_error(y_confirmed_itb[2:8], projection_confirmed_itb[2:8])), "VS Reported RMSE:", 8.5051)
#all confirmed, death, and recoveries has been sliced to have same dates

t = np.arange(0, len(indonesia_confirmed_data))

date_strings = indonesia_confirmed_data.keys()

y_confirmed = indonesia_confirmed_data.values



#get a set of new optimized parameters, using the proposed p0 in ITB 15 March 2020 report

popt_confirmed_itb_new, pcov_confirmed_itb_new = curve_fit(richard_curve, t, y_confirmed, p0=richard_curve_itb_param) #specified that initial number of confirmed is put at 2 



#try using our own p0 parameters for the Richard Curve, to see whehter we get different results or not

#Noted that this is still close to parameters chosen by ITB researchers

custom_richard_p0 = (8800, 0.1, 0.50, 42)

popt_confirmed_custom, pcov_confirmed_custom = curve_fit(richard_curve, t, y_confirmed, p0=custom_richard_p0) 



days_to_project = 10



projection_times = np.linspace(t[0], t[-1] + days_to_project, num=len(t) + days_to_project)

projection_confirmed_itb_newdata = richard_curve(projection_times, *popt_confirmed_itb)

projection_confirmed_itb_refit = richard_curve(projection_times, *popt_confirmed_itb_new)

projection_confirmed_customp0 = richard_curve(projection_times, *popt_confirmed_custom) 





trace_confirmed_cur = go.Scatter(

                      x=t,

                      y=y_confirmed,

                      mode='markers',

                      marker=go.scatter.Marker(color='rgb(255, 127, 14)',

                                      size=7),

                      name='Current Confirmed cases (up to {})'.format(date_strings[-1])

                      )



trace_confirmed_itb_new_proj = go.Scatter(

                      x=projection_times, #projection_times_itb,

                      y=projection_confirmed_itb_newdata,

                      mode='lines',

                      marker=go.scatter.Marker(color='rgb(10, 50, 250)',

                                      size=7),

                      name='Projected # Confirmed (Richard Curve, previous ITB params)'

                      )



trace_confirmed_itb_refit = go.Scatter(

                      x=projection_times, #projection_times_itb,

                      y=projection_confirmed_itb_refit,

                      mode='lines',

                      marker=go.scatter.Marker(color='rgb(200, 200, 20)',

                                      size=7),

                      name='Projected # Confirmed (Richard Curve, re-trained params with ITB\'s p0)'

                      )



trace_confirmed_custom_proj = go.Scatter(

                       x=projection_times,

                       y=projection_confirmed_customp0,

                       mode='lines',

                       marker=go.scatter.Marker(color='rgb(200, 50, 200)'),

                       name='Projected # Confirmed Cases (Richard Curve, custom params & p0)'

                       )



plot_data = [trace_confirmed_cur, trace_confirmed_itb_new_proj, trace_confirmed_itb_refit, \

             trace_confirmed_custom_proj] 

plotly_graph(plot_data, 'Estimated Projection of COVID-19 in Indonesia, Richard\'s Curve Fitting', case='Confirmed')



print("Average RMSE obtained for Confirmed:")

print("> Previous ITB params: ", np.sqrt(mean_squared_error(y_confirmed, projection_confirmed_itb_newdata[:len(t)])) )

print("> Re-trained params w/ ITB p0:", np.sqrt(mean_squared_error(y_confirmed, projection_confirmed_itb_refit[:len(t)])) )

print("> Custom params & p0]", np.sqrt(mean_squared_error(y_confirmed, projection_confirmed_customp0[:len(t)])) ) 
def exp_function(x, a, b, c):

    return a * np.exp(b*x) + c



# popt = fitted(a,b,c) coefficients to minimize square error; pcov = optimized covariance of returned popt

popt_confirmed_exp, pcov_confirmed_exp = curve_fit(exp_function, t, y_confirmed, p0=(1, 1e-6, 1)) 



projection_confirmed_exp = exp_function(projection_times, *popt_confirmed_exp)





trace_confirmed_proj_exp = go.Scatter(

                       x=projection_times,

                       y=projection_confirmed_exp,

                       mode='lines',

                       marker=go.scatter.Marker(color='rgb(31, 119, 180)'),

                       name='Projected Confirmed Cases (Exponential Fit)'

                       )



plot_data = [trace_confirmed_cur, trace_confirmed_proj_exp]

plotly_graph(plot_data, 'Estimated Projection of COVID-19 in Indonesia, Exponential Function Fitting',case='Confirmed')



print("Average RMSE obtained for Confirmed: [Exponential Fit]", \

      np.sqrt(mean_squared_error(y_confirmed, projection_confirmed_exp[:len(t)])))
y_deaths = indonesia_deaths_data.values

custom_richard_p0_deaths = (8000, 0.05, 0.5,40)



# popt = fitted(a,b,c) coefficients to minimize square error; pcov = optimized covariance of returned popt

popt_deaths_richard, pcov_deaths_richard = curve_fit(richard_curve, t, y_deaths, p0=custom_richard_p0_deaths)

popt_deaths_exp    , pcov_deaths_exp     = curve_fit(exp_function, t, y_deaths, p0=(1, 1e-6, 1))

#popt_recovered, pcov_recovered = curve_fit(exp_function, x, y_recovered, p0=(1, 1e-6, 1))



projection_deaths_exp = exp_function(projection_times, *popt_deaths_exp)

projection_deaths_richard = richard_curve(projection_times, *popt_deaths_richard)



trace_deaths_cur = go.Scatter(

                    x = t,

                    y = y_deaths,

                    mode = 'markers',

                    marker = go.scatter.Marker(color='rgb(250, 10, 10)'),

                    name='Current number of Deaths'

                    )



trace_deaths_proj_exp = go.Scatter(

                       x=projection_times,

                       y=projection_deaths_exp,

                       mode='lines',

                       marker=go.scatter.Marker(color='rgb(200, 100, 100)'),

                       name='Projected Deaths (Exponential Fit)'

                       )



trace_deaths_proj_richard = go.Scatter(

                       x=projection_times,

                       y=projection_deaths_richard,

                       mode='lines',

                       marker=go.scatter.Marker(color='rgb(250, 50, 150)'),

                       name='Projected Deaths (Richard\'s Curve, custom params & p0)'

                       )

plot_data = [trace_deaths_cur, trace_deaths_proj_exp, trace_deaths_proj_richard]

plotly_graph(plot_data, "Estimated Projection of COVID-19 Deaths in Indonesia", case="Deaths")



print("Average RMSE obtained for Death cases:")

print("> Exponential Fit:", \

      np.sqrt(mean_squared_error(y_deaths, projection_deaths_exp[:len(t)])))

print("> Richard Curve:", \

      np.sqrt(mean_squared_error(y_deaths, projection_deaths_richard[:len(t)])))
projected_months = 7

extended_days_to_project = projected_months * 31



popt_confirmed_custom_old, pcov_confirmed_custom_old = curve_fit(richard_curve, t[:21], y_confirmed[:21],p0=custom_richard_p0)



projection_times_ext = np.linspace(t[0], t[-1] + extended_days_to_project, num=200)

projection_confirmed_itb_new_ext = richard_curve(projection_times_ext, *popt_confirmed_itb) 

projection_confirmed_customp0_ext = richard_curve(projection_times_ext, *popt_confirmed_custom) 

projection_confirmed_customp0_ext_old = richard_curve(projection_times_ext, *popt_confirmed_custom_old)



trace_confirmed_custom_proj_ext = go.Scatter(

                       x=projection_times_ext,

                       y=projection_confirmed_customp0_ext,

                       mode='lines',

                       marker=go.Marker(color='rgb(200, 50, 200)'),

                       name='Extended Projected # Confirmed - (Richard Curve, custom params & p0) [24 March 2020]'

                       )



trace_confirmed_custom_proj_ext_old = go.Scatter(

                       x=projection_times_ext,

                       y=projection_confirmed_customp0_ext_old,

                       mode='lines',

                       marker=go.Marker(color='rgb(50, 200, 200)'),

                       name='Extended Projected # Confirmed - (Richard Curve, custom params & p0) [21 March 2020]'

                       )



plot_data = [trace_confirmed_cur, trace_confirmed_custom_proj_ext, trace_confirmed_custom_proj_ext_old] 

plotly_graph(plot_data, 'Extended Projection numbers of COVID-19 in Indonesia, up to 7 months from 1 March 2020', case="Confirmed")

print("Last Date in dataset:", date_strings[-1])

last_days = 4

print("Re-training up to last {} days of data, up to:".format(str(last_days)), date_strings[-(last_days + 1)] )
popt_conf_custom_partial, pcov_conf_custom_partial = curve_fit(richard_curve, t[:-last_days], \

                                                               y_confirmed[:-last_days], p0=custom_richard_p0)



popt_conf_itb_partial, pcov_conf_itb_partial = curve_fit(richard_curve, t[:-last_days], \

                                                               y_confirmed[:-last_days], p0=richard_curve_itb_param)



popt_conf_exp_partial, pcov_conf_exp_partial = curve_fit(exp_function, t[:-last_days], y_confirmed[:-last_days], \

                                                         p0=(1, 1e-6, 1)) 



popt_deaths_rich_partial, pcov_deaths_rich_partial = curve_fit(richard_curve, t[:-last_days], \

                                                               y_deaths[:-last_days], p0=custom_richard_p0_deaths)



popt_deaths_exp_partial, pconv_deaths_exp_partial = curve_fit(exp_function, t[:-last_days], y_deaths[:-last_days],\

                                                             p0=(1, 1e-6, 1))





projection_confirmed_custom_partial = richard_curve(projection_times, *popt_conf_custom_partial)

projection_confirmed_itb_partial = richard_curve(projection_times, *popt_conf_itb_partial)

projection_confirmed_exp_partial = exp_function(projection_times, *popt_conf_exp_partial)

projection_deaths_richard_partial = richard_curve(projection_times, *popt_deaths_rich_partial)

projection_deaths_exp_partial  = exp_function(projection_times, *popt_deaths_exp_partial)



trace_confirmed_custom_proj_partial = go.Scatter(

                       x=projection_times,

                       y=projection_confirmed_custom_partial,

                       mode='lines',

                       marker=go.scatter.Marker(color='rgb(255, 100, 255)'),

                       name='Projected # Confirmed Cases (Richard, Partial Data)'

                       )



trace_confirmed_itb_proj_partial = go.Scatter(

                       x=projection_times,

                       y=projection_confirmed_itb_partial,

                       mode='lines',

                       marker=go.scatter.Marker(color='rgb(255, 255, 100)'),

                       name='Projected # Confirmed Cases (Richard, Partial Data)'

                       )



trace_confirmed_proj_exp_partial = go.Scatter(

                       x=projection_times,

                       y=projection_confirmed_exp_partial,

                       mode='lines',

                       marker=go.scatter.Marker(color='rgb(31, 200, 250)'),

                       name='Projected # Confirmed Cases (Exp Fit, partial data)'

                       )



trace_deaths_proj_exp_partial = go.Scatter(

                       x=projection_times,

                       y=projection_deaths_exp_partial,

                       mode='lines',

                       marker=go.scatter.Marker(color='rgb(200, 130, 150)'),

                       name='Projected Deaths (Exp Fit, Partial Data)'

                       )



trace_deaths_proj_richard_partial = go.Scatter(

                       x=projection_times,

                       y=projection_deaths_richard_partial,

                       mode='lines',

                       marker=go.scatter.Marker(color='rgb(250, 100, 250)'),

                       name='Projected Deaths (Richard, Partial Data)'

                       )



plot_data_confirmed = [trace_confirmed_cur, trace_confirmed_custom_proj, trace_confirmed_itb_proj_partial, \

                        trace_confirmed_custom_proj_partial, trace_confirmed_proj_exp, \

                       trace_confirmed_proj_exp_partial]

plot_data_deaths = [trace_deaths_cur, trace_deaths_proj_richard, trace_deaths_proj_richard_partial,\

                    trace_deaths_proj_exp, trace_deaths_proj_exp_partial]

plotly_graph(plot_data_confirmed, "Comparison of models trained on full vs partial Confirmed data", case="Confirmed")

print("Last day Confirmed ({}) Comparisons:   Actual   |   Full_Fit   |   Partial_Fit".format(date_strings[-1]))

print("> Richard, Custom p0 =>                ", y_confirmed[-1], "|", projection_confirmed_customp0[len(t)-1], "|",

       projection_confirmed_custom_partial[len(t)-1])

print("> Richard, ITB param =>                ", y_confirmed[-1], "|", projection_confirmed_itb_newdata[len(t)-1], "|",

       projection_confirmed_itb_partial[len(t)-1])

print("> Exponential =>                       ", y_confirmed[-1], "|", projection_confirmed_exp[len(t)-1], "|",

       projection_confirmed_exp_partial[len(t)-1])

print()



plotly_graph(plot_data_deaths, "Comparison of models t rained on full vs partial Deaths data", case="Deaths")

print("Last day Deaths ({}) Comparisons:   Actual   |   Full_Fit   |   Partial_Fit".format(date_strings[-1]))

print("> Richard, Custom p0 =>                ", y_deaths[-1],"|", projection_deaths_richard[len(t)-1],"|", 

       projection_deaths_richard_partial[len(t)-1])

print("> Exponential =>                       ", y_deaths[-1], "|", projection_deaths_exp[len(t)-1], "|",

       projection_deaths_exp_partial[len(t)-1])
projection_confirmed_custom_partial_ext = richard_curve(projection_times_ext, *popt_conf_custom_partial)

projection_confirmed_itb_partial_ext = richard_curve(projection_times_ext, *popt_conf_itb_partial)

projection_confirmed_exp_partial_ext = exp_function(projection_times_ext, *popt_conf_exp_partial)

projection_confirmed_exp_ext = exp_function(projection_times_ext, *popt_confirmed_exp)

projection_deaths_richard_partial_ext = richard_curve(projection_times_ext, *popt_deaths_rich_partial)

projection_deaths_richard_ext = richard_curve(projection_times_ext, *popt_deaths_richard)



trace_confirmed_custom_proj_ext = go.Scatter(

                       x=projection_times_ext,

                       y=projection_confirmed_customp0_ext,

                       mode='lines',

                       marker=go.scatter.Marker(color='rgb(100, 30, 250)'),

                       name='Extended Projected # Confirmed- (Richard Curve, custom params & p0)'

                       )



trace_confirmed_exp_ext = go.Scatter(

                          x=projection_times_ext,

                          y=projection_confirmed_exp_ext,

                          mode='lines',

                          marker=go.scatter.Marker(color='rgb(150, 10, 250)'),

                          name='Extended Projected # Confirmed- (Exponential, Full Data)'

                          )



trace_confirmed_custom_partial_ext = go.Scatter(

                       x=projection_times_ext,

                       y=projection_confirmed_custom_partial_ext,

                       mode='lines',

                       marker=go.scatter.Marker(color='rgb(70, 80, 200)'),

                       name='Extended Projected # Confirmed (Richard, Partial Data)'

                       )



trace_confirmed_exp_partial_ext = go.Scatter(

                       x=projection_times_ext,

                       y=projection_confirmed_exp_partial_ext,

                       mode='lines',

                       marker=go.scatter.Marker(color='rgb(31, 200, 250)'),

                       name='Extended Projected # Confirmed (Exponential, Partial Data)'

                       )



trace_confirmed_itb_proj_partial_ext = go.Scatter(

                       x=projection_times_ext,

                       y=projection_confirmed_itb_partial_ext,

                       mode='lines',

                       marker=go.scatter.Marker(color='rgb(255, 255, 100)'),

                       name='Projected # Confirmed Cases (ITB, Partial Data)'

                       )



trace_deaths_proj_richard_ext = go.Scatter(

                       x=projection_times_ext,

                       y=projection_deaths_richard_ext,

                       mode='lines',

                       marker=go.scatter.Marker(color='rgb(250, 10, 10)'),

                       name='Extended Projected # Deaths - (Richard)'

                       )



trace_deaths_proj_richard_partial_ext = go.Scatter(

                       x=projection_times_ext,

                       y=projection_deaths_richard_partial_ext,

                       mode='lines',

                       marker=go.scatter.Marker(color='rgb(250, 100, 250)'),

                       name='Extended Projected # Deaths - (Richard, Partial Data)'

                       )





plot_data = [trace_confirmed_custom_proj_ext, trace_confirmed_custom_partial_ext, trace_confirmed_itb_proj_partial_ext,\

             trace_deaths_proj_richard_ext, trace_deaths_proj_richard_partial_ext] #, trace_confirmed_exp_ext, trace_confirmed_exp_partial_ext]

plotly_graph(plot_data, "Extended projections, comparison of Fitting on Partial vs Full data - Richard Functions")



plot_data = [trace_confirmed_exp_ext, trace_confirmed_exp_partial_ext]

plotly_graph(plot_data, "Extended projections, comparison of Fitting on Partial vs Full data - Exponential Functions")



plot_data = [trace_confirmed_custom_partial_ext, trace_deaths_proj_richard_ext, trace_deaths_proj_richard_partial_ext]

plotly_graph(plot_data, "Extended projections, comparison of Fitting on Partial vs Full data (minus Confirmed cases with Partial Data Fitting)")