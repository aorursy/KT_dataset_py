import numpy as np

from numpy import inf

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from plotly.graph_objs.scatter.marker import Line

from plotly.graph_objs import Line

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

import warnings

warnings.filterwarnings('ignore')
worldwide = pd.read_csv('https://covid.ourworldindata.org/data/ecdc/total_cases.csv')
worldwide['date'] = pd.to_datetime(worldwide['date'])

worldwide = worldwide.reset_index()

a = worldwide.iloc[worldwide.shape[0]-1]

a = list(a)[0]

top_10 = worldwide.tail(1)

top_10 = top_10.transpose()

top_10 = top_10.reset_index()

top_10 = top_10.rename(columns = {'index' : 'Country'})

top_10 = top_10.rename(columns = {a : 'Cases'})

top_10.drop([0, 1], inplace = True)

top_10 = top_10.reset_index()

top_10 = top_10[top_10['Country'] != 'World']

top_10.drop(columns = ['index'], inplace = True)

top_10 = top_10.sort_values(by = 'Cases', ascending = False).head(10)

top_10
country_list = np.array(top_10['Country'])
country_wise = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')
country_wise = country_wise[country_wise['location'] != 'world']
world = country_wise.sort_values(by = 'date')
world_data = px.choropleth(world, locations="iso_code",

                    color="total_cases",

                    hover_name="location",

                    animation_frame="date",

                    title = "Total new COVID-19 cases",

                   color_continuous_scale=px.colors.sequential.Reds)

world_data["layout"].pop("updatemenus")

world_data.show()
world_data = px.choropleth(world, locations="iso_code",

                    color="total_deaths",

                    hover_name="location",

                    animation_frame="date",

                    title = "Total COVID-19 deaths",

                   color_continuous_scale=px.colors.sequential.Reds)

world_data["layout"].pop("updatemenus")

world_data.show()
world_data = px.choropleth(world, locations="iso_code",

                    color="new_tests_per_thousand",

                    hover_name="location",

                    animation_frame="date",

                    title = "Tests Per Thousand",

                   color_continuous_scale=px.colors.sequential.Plasma)

world_data["layout"].pop("updatemenus")

world_data.show()
country_wise['date'] = pd.to_datetime(country_wise['date'])

country_wise['Mortality_Rate'] = (country_wise['total_deaths']/country_wise['total_cases'])*100
cases = []

for col in country_wise:

    for col in country_list:

        cases.append(country_wise.loc[country_wise['location'] == col])
country_1 = cases[0]

country_2 = cases[1]

country_3 = cases[2]

country_4 = cases[3]

country_5 = cases[4]

country_6 = cases[5]

country_7 = cases[6]

country_8 = cases[7]

country_9 = cases[8]

country_10 = cases[9]
x = country_1.merge(country_2, on = 'date', how = 'left')

x = x[['date', 'new_cases_x', 'new_cases_y']]

x.rename(columns = {'new_cases_x' : 'country_1', 'new_cases_y' : 'country_2'}, inplace = True)

x = x.merge(country_3, on = 'date', how = 'left')

x = x[['date', 'country_1', 'country_2', 'new_cases']]

x.rename(columns = {'new_cases' : 'country_3'}, inplace = True)

x = x.merge(country_4, on = 'date', how = 'left')

x = x[['date', 'country_1', 'country_2', 'country_3', 'new_cases']]

x.rename(columns = {'new_cases' : 'country_4'}, inplace = True)

x = x.merge(country_5, on = 'date', how = 'left')

x = x[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'new_cases']]

x.rename(columns = {'new_cases' : 'country_5'}, inplace = True)

x = x.merge(country_6, on = 'date', how = 'left')

x = x[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'new_cases']]

x.rename(columns = {'new_cases' : 'country_6'}, inplace = True)

x = x.merge(country_7, on = 'date', how = 'left')

x = x[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'new_cases']]

x.rename(columns = {'new_cases' : 'country_7'}, inplace = True)

x = x.merge(country_8, on = 'date', how = 'left')

x = x[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'country_7', 'new_cases']]

x.rename(columns = {'new_cases' : 'country_8'}, inplace = True)

x = x.merge(country_9, on = 'date', how = 'left')

x = x[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'country_7', 'country_8', 'new_cases']]

x.rename(columns = {'new_cases' : 'country_9'}, inplace = True)

x = x.merge(country_10, on = 'date', how = 'left')

x = x[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'country_7', 'country_8', 'country_9', 'new_cases']]

x.rename(columns = {'new_cases' : 'country_10'}, inplace = True)
x = x.fillna(0)
x = x.reset_index()
x.rename(columns = {'index':'Days'}, inplace = True)
top_10 = top_10[top_10['Country'] != 'Rest']
fig = px.bar(top_10, 

            x = 'Cases',

            y = 'Country', 

            orientation = 'h', 

            height = 500, 

            title = '10 Worst Affected Nations', 

            color = 'Country')

fig.show()
total_affected = worldwide.tail(1)

total_affected = int(total_affected['World'])
top_10['Percentage'] = (top_10['Cases']/total_affected)*100
top_10_cases = top_10['Cases'].sum()

rest_cases = total_affected - top_10_cases

per = (rest_cases/total_affected)*100

new_row = pd.DataFrame({'Country' : ['Rest'], 'Cases' : [rest_cases], 'Percentage' : [per]})

top_10 = pd.concat([top_10, new_row])

top_10.style.background_gradient(cmap = 'reds')
fig = px.pie(top_10,

            values = 'Percentage',

            names = 'Country', 

            title = 'Percentage Of Global Cases')

fig.show()
top_10_total_cases = country_1.merge(country_2, on = 'date', how = 'left')

top_10_total_cases = top_10_total_cases[['date', 'total_cases_x', 'total_cases_y']]

top_10_total_cases.rename(columns = {'total_cases_x' : 'country_1', 'total_cases_y' : 'country_2'}, inplace = True)

top_10_total_cases = top_10_total_cases.merge(country_3, on = 'date', how = 'left')

top_10_total_cases = top_10_total_cases[['date', 'country_1', 'country_2', 'total_cases']]

top_10_total_cases.rename(columns = {'total_cases' : 'country_3'}, inplace = True)

top_10_total_cases = top_10_total_cases.merge(country_4, on = 'date', how = 'left')

top_10_total_cases = top_10_total_cases[['date', 'country_1', 'country_2', 'country_3', 'total_cases']]

top_10_total_cases.rename(columns = {'total_cases' : 'country_4'}, inplace = True)

top_10_total_cases = top_10_total_cases.merge(country_5, on = 'date', how = 'left')

top_10_total_cases = top_10_total_cases[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'total_cases']]

top_10_total_cases.rename(columns = {'total_cases' : 'country_5'}, inplace = True)

top_10_total_cases = top_10_total_cases.merge(country_6, on = 'date', how = 'left')

top_10_total_cases = top_10_total_cases[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'total_cases']]

top_10_total_cases.rename(columns = {'total_cases' : 'country_6'}, inplace = True)

top_10_total_cases = top_10_total_cases.merge(country_7, on = 'date', how = 'left')

top_10_total_cases = top_10_total_cases[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'total_cases']]

top_10_total_cases.rename(columns = {'total_cases' : 'country_7'}, inplace = True)

top_10_total_cases = top_10_total_cases.merge(country_8, on = 'date', how = 'left')

top_10_total_cases = top_10_total_cases[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'country_7', 'total_cases']]

top_10_total_cases.rename(columns = {'total_cases' : 'country_8'}, inplace = True)

top_10_total_cases = top_10_total_cases.merge(country_9, on = 'date', how = 'left')

top_10_total_cases = top_10_total_cases[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'country_7', 'country_8', 'total_cases']]

top_10_total_cases.rename(columns = {'total_cases' : 'country_9'}, inplace = True)

top_10_total_cases = top_10_total_cases.merge(country_10, on = 'date', how = 'left')

top_10_total_cases = top_10_total_cases[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'country_7', 'country_8', 'country_9', 'total_cases']]

top_10_total_cases.rename(columns = {'total_cases' : 'country_10'}, inplace = True)

top_10_total_cases.fillna(0, inplace = True)
top_10_total_cases = top_10_total_cases.reset_index()

top_10_total_cases.drop(columns = ['date'], inplace = True)
top_10_total_cases.rename(columns = {'index' : 'Days'}, inplace = True)
top_10_total_cases.rename(columns = {'country_1' : country_list[0]}, inplace = True)

top_10_total_cases.rename(columns = {'country_2' : country_list[1]}, inplace = True)

top_10_total_cases.rename(columns = {'country_3' : country_list[2]}, inplace = True)

top_10_total_cases.rename(columns = {'country_4' : country_list[3]}, inplace = True)

top_10_total_cases.rename(columns = {'country_5' : country_list[4]}, inplace = True)

top_10_total_cases.rename(columns = {'country_6' : country_list[5]}, inplace = True)

top_10_total_cases.rename(columns = {'country_7' : country_list[6]}, inplace = True)

top_10_total_cases.rename(columns = {'country_8' : country_list[7]}, inplace = True)

top_10_total_cases.rename(columns = {'country_9' : country_list[8]}, inplace = True)

top_10_total_cases.rename(columns = {'country_10' : country_list[9]}, inplace = True)
fig_inc = go.Figure(go.Line(x = top_10_total_cases['Days'], 

                            y = top_10_total_cases[country_list[0]],

                            name = country_list[0], 

                            mode = 'lines+markers'))

fig_inc.add_trace(go.Line(x = top_10_total_cases['Days'],

                          y = top_10_total_cases[country_list[1]], 

                          name = country_list[1], 

                          mode = 'lines+markers'))

fig_inc.add_trace(go.Line(x = top_10_total_cases['Days'],

                          y = top_10_total_cases[country_list[2]], 

                          name = country_list[2], 

                          mode = 'lines+markers'))

fig_inc.add_trace(go.Line(x = top_10_total_cases['Days'],

                          y = top_10_total_cases[country_list[3]], 

                          name = country_list[3], 

                          mode = 'lines+markers'))

fig_inc.add_trace(go.Line(x = top_10_total_cases['Days'],

                          y = top_10_total_cases[country_list[4]], 

                          name = country_list[4], 

                          mode = 'lines+markers'))

fig_inc.add_trace(go.Line(x = top_10_total_cases['Days'],

                          y = top_10_total_cases[country_list[5]], 

                          name = country_list[5], 

                          mode = 'lines+markers'))

fig_inc.add_trace(go.Line(x = top_10_total_cases['Days'],

                          y = top_10_total_cases[country_list[6]], 

                          name = country_list[6], 

                          mode = 'lines+markers'))

fig_inc.add_trace(go.Line(x = top_10_total_cases['Days'],

                          y = top_10_total_cases[country_list[7]], 

                          name = country_list[7], 

                          mode = 'lines+markers'))

fig_inc.add_trace(go.Line(x = top_10_total_cases['Days'],

                          y = top_10_total_cases[country_list[8]], 

                          name = country_list[8], 

                          mode = 'lines+markers'))

fig_inc.add_trace(go.Line(x = top_10_total_cases['Days'],

                          y = top_10_total_cases[country_list[9]], 

                          name = country_list[9], 

                          mode = 'lines+markers'))

fig_inc.update_layout(title = 'Incremental Cases')

fig_inc.update_xaxes(title= '------>Timeline (Days since 31-12-2019)' ,showline=False)

fig_inc.update_yaxes(title= '------>Number of incremental cases', showline=False)

fig_inc.show()
confirmed = pd.read_csv('https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv&filename=time_series_covid19_confirmed_global.csv')

death = pd.read_csv('https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_deaths_global.csv&filename=time_series_covid19_deaths_global.csv')

recovered = pd.read_csv('https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_recovered_global.csv&filename=time_series_covid19_recovered_global.csv')
a = confirmed.groupby(['Country/Region']).agg(sum)

a = a.sort_values(a.columns[a.shape[1]-1], ascending = False)

a = a.transpose()

a.drop(a.index[[0, 1]], inplace = True)

a.rename(columns = {'US' : 'United States'}, inplace = True)

a = a[[country_list[0], country_list[1], country_list[2], country_list[3], country_list[4], country_list[5], country_list[6], country_list[7], country_list[8], country_list[9]]]

b = death.groupby(['Country/Region']).agg(sum)

b = b.transpose()

b.rename(columns = {'US' : 'United States'}, inplace = True)

b = b[[country_list[0], country_list[1], country_list[2], country_list[3], country_list[4], country_list[5], country_list[6], country_list[7], country_list[8], country_list[9]]]

b.drop(b.index[[0, 1]], inplace = True)
c = recovered.groupby(['Country/Region']).agg(sum)

c = c.transpose()

c.rename(columns = {'US' : 'United States'}, inplace = True)

c = c[[country_list[0], country_list[1], country_list[2], country_list[3], country_list[4], country_list[5], country_list[6], country_list[7], country_list[8], country_list[9]]]

c.drop(c.index[[0, 1]], inplace = True)
country_one = pd.DataFrame({'Confirmed' : a[country_list[0]], 'Death' : b[country_list[0]], 'Recovered' : c[country_list[0]]})

country_one['Active'] = country_one['Confirmed']-(country_one['Death']+country_one['Recovered'])

country_one['Mortality_Rate'] = (country_one['Death']/country_one['Confirmed'])*100

country_one['Recovery_Rate'] = (country_one['Recovered']/country_one['Confirmed'])*100

country_one = country_one.fillna(0)

country_one = country_one.reset_index()

country_one.rename(columns = {'index' : 'Date'}, inplace = True)



country_two = pd.DataFrame({'Confirmed' : a[country_list[1]], 'Death' : b[country_list[1]], 'Recovered' : c[country_list[1]]})

country_two['Active'] = country_two['Confirmed']-(country_two['Death']+country_two['Recovered'])

country_two['Mortality_Rate'] = (country_two['Death']/country_two['Confirmed'])*100

country_two['Recovery_Rate'] = (country_two['Recovered']/country_two['Confirmed'])*100

country_two = country_two.fillna(0)

country_two = country_two.reset_index()

country_two.rename(columns = {'index' : 'Date'}, inplace = True)



country_three = pd.DataFrame({'Confirmed' : a[country_list[2]], 'Death' : b[country_list[2]], 'Recovered' : c[country_list[2]]})

country_three['Active'] = country_three['Confirmed']-(country_three['Death']+country_three['Recovered'])

country_three['Mortality_Rate'] = (country_three['Death']/country_three['Confirmed'])*100

country_three['Recovery_Rate'] = (country_three['Recovered']/country_three['Confirmed'])*100

country_three = country_three.fillna(0)

country_three = country_three.reset_index()

country_three.rename(columns = {'index' : 'Date'}, inplace = True)



country_four = pd.DataFrame({'Confirmed' : a[country_list[3]], 'Death' : b[country_list[3]], 'Recovered' : c[country_list[3]]})

country_four['Active'] = country_four['Confirmed']-(country_four['Death']+country_four['Recovered'])

country_four['Mortality_Rate'] = (country_four['Death']/country_four['Confirmed'])*100

country_four['Recovery_Rate'] = (country_four['Recovered']/country_four['Confirmed'])*100

country_four = country_four.fillna(0)

country_four = country_four.reset_index()

country_four.rename(columns = {'index' : 'Date'}, inplace = True)



country_five = pd.DataFrame({'Confirmed' : a[country_list[4]], 'Death' : b[country_list[4]], 'Recovered' : c[country_list[4]]})

country_five['Active'] = country_five['Confirmed']-(country_five['Death']+country_five['Recovered'])

country_five['Mortality_Rate'] = (country_five['Death']/country_five['Confirmed'])*100

country_five['Recovery_Rate'] = (country_five['Recovered']/country_five['Confirmed'])*100

country_five = country_five.fillna(0)

country_five = country_five.reset_index()

country_five.rename(columns = {'index' : 'Date'}, inplace = True)



country_six = pd.DataFrame({'Confirmed' : a[country_list[5]], 'Death' : b[country_list[5]], 'Recovered' : c[country_list[5]]})

country_six['Active'] = country_six['Confirmed']-(country_six['Death']+country_six['Recovered'])

country_six['Mortality_Rate'] = (country_six['Death']/country_six['Confirmed'])*100

country_six['Recovery_Rate'] = (country_six['Recovered']/country_six['Confirmed'])*100

country_six = country_six.fillna(0)

country_six = country_six.reset_index()

country_six.rename(columns = {'index' : 'Date'}, inplace = True)



country_seven = pd.DataFrame({'Confirmed' : a[country_list[6]], 'Death' : b[country_list[6]], 'Recovered' : c[country_list[6]]})

country_seven['Active'] = country_seven['Confirmed']-(country_seven['Death']+country_seven['Recovered'])

country_seven['Mortality_Rate'] = (country_seven['Death']/country_seven['Confirmed'])*100

country_seven['Recovery_Rate'] = (country_seven['Recovered']/country_seven['Confirmed'])*100

country_seven = country_seven.fillna(7)

country_seven = country_seven.reset_index()

country_seven.rename(columns = {'index' : 'Date'}, inplace = True)



country_eight = pd.DataFrame({'Confirmed' : a[country_list[7]], 'Death' : b[country_list[7]], 'Recovered' : c[country_list[7]]})

country_eight['Active'] = country_eight['Confirmed']-(country_eight['Death']+country_eight['Recovered'])

country_eight['Mortality_Rate'] = (country_eight['Death']/country_eight['Confirmed'])*100

country_eight['Recovery_Rate'] = (country_eight['Recovered']/country_eight['Confirmed'])*100

country_eight = country_eight.fillna(0)

country_eight = country_eight.reset_index()

country_eight.rename(columns = {'index' : 'Date'}, inplace = True)



country_nine = pd.DataFrame({'Confirmed' : a[country_list[8]], 'Death' : b[country_list[8]], 'Recovered' : c[country_list[8]]})

country_nine['Active'] = country_nine['Confirmed']-(country_nine['Death']+country_nine['Recovered'])

country_nine['Mortality_Rate'] = (country_nine['Death']/country_nine['Confirmed'])*100

country_nine['Recovery_Rate'] = (country_nine['Recovered']/country_nine['Confirmed'])*100

country_nine = country_nine.fillna(0)

country_nine = country_nine.reset_index()

country_nine.rename(columns = {'index' : 'Date'}, inplace = True)



country_ten = pd.DataFrame({'Confirmed' : a[country_list[9]], 'Death' : b[country_list[9]], 'Recovered' : c[country_list[9]]})

country_ten['Active'] = country_ten['Confirmed']-(country_ten['Death']+country_ten['Recovered'])

country_ten['Mortality_Rate'] = (country_ten['Death']/country_ten['Confirmed'])*100

country_ten['Recovery_Rate'] = (country_ten['Recovered']/country_ten['Confirmed'])*100

country_ten = country_ten.fillna(0)

country_ten = country_ten.reset_index()

country_ten.rename(columns = {'index' : 'Date'}, inplace = True)
fig_log_one = go.Figure(go.Line(x = country_one['Date'], 

                            y = np.log1p(country_one['Confirmed']),

                            name = 'Confirmed', 

                            mode = 'lines'))

fig_log_one.add_trace(go.Line(x = country_one['Date'],

                          y = np.log1p(country_one['Death']), 

                          name = 'Death', 

                          mode = 'lines'))

fig_log_one.add_trace(go.Line(x = country_one['Date'],

                          y = np.log1p(country_one['Recovered']), 

                          name = 'Recovered', 

                          mode = 'lines'))

fig_log_one.add_trace(go.Line(x = country_one['Date'],

                          y = np.log1p(country_one['Active']), 

                          name = 'Active', 

                          mode = 'lines'))

fig_log_one.update_layout(title = country_list[0])

fig_log_one.update_xaxes(title= '------>Timeline' ,showline=False)

fig_log_one.update_yaxes(title= '------>Lograthmic increment in cases', showline=False)

fig_log_one.show()



fig_log_two = go.Figure(go.Line(x = country_two['Date'], 

                            y = np.log1p(country_two['Confirmed']),

                            name = 'Confirmed', 

                            mode = 'lines'))

fig_log_two.add_trace(go.Line(x = country_two['Date'],

                          y = np.log1p(country_two['Death']), 

                          name = 'Death', 

                          mode = 'lines'))

fig_log_two.add_trace(go.Line(x = country_two['Date'],

                          y = np.log1p(country_two['Recovered']), 

                          name = 'Recovered', 

                          mode = 'lines'))

fig_log_two.add_trace(go.Line(x = country_two['Date'],

                          y = np.log1p(country_two['Active']), 

                          name = 'Active', 

                          mode = 'lines'))

fig_log_two.update_layout(title = country_list[1])

fig_log_two.update_xaxes(title= '------>Timeline' ,showline=False)

fig_log_two.update_yaxes(title= '------>Lograthmic increment in cases', showline=False)

fig_log_two.show()



fig_three = go.Figure(go.Line(x = country_three['Date'], 

                            y = np.log1p(country_three['Confirmed']),

                            name = 'Confirmed', 

                            mode = 'lines'))

fig_three.add_trace(go.Line(x = country_three['Date'],

                          y = np.log1p(country_three['Death']), 

                          name = 'Death', 

                          mode = 'lines'))

fig_three.add_trace(go.Line(x = country_three['Date'],

                          y = np.log1p(country_three['Recovered']), 

                          name = 'Recovered', 

                          mode = 'lines'))

fig_three.add_trace(go.Line(x = country_three['Date'],

                          y = np.log1p(country_three['Active']), 

                          name = 'Active', 

                          mode = 'lines'))

fig_three.update_layout(title = country_list[2])

fig_three.update_xaxes(title= '------>Timeline' ,showline=False)

fig_three.update_yaxes(title= '------>Lograthmic increment in cases', showline=False)

fig_three.show()



fig_four = go.Figure(go.Line(x = country_four['Date'], 

                            y = np.log1p(country_four['Confirmed']),

                            name = 'Confirmed', 

                            mode = 'lines'))

fig_four.add_trace(go.Line(x = country_four['Date'],

                          y = np.log1p(country_four['Death']), 

                          name = 'Death', 

                          mode = 'lines'))

fig_four.add_trace(go.Line(x = country_four['Date'],

                          y = np.log1p(country_four['Recovered']), 

                          name = 'Recovered', 

                          mode = 'lines'))

fig_four.add_trace(go.Line(x = country_four['Date'],

                          y = np.log1p(country_four['Active']), 

                          name = 'Active', 

                          mode = 'lines'))

fig_four.update_layout(title = country_list[3])

fig_four.update_xaxes(title= '------>Timeline' ,showline=False)

fig_four.update_yaxes(title= '------>Lograthmic increment in cases', showline=False)

fig_four.show()



fig_five = go.Figure(go.Line(x = country_five['Date'], 

                            y = np.log1p(country_five['Confirmed']),

                            name = 'Confirmed', 

                            mode = 'lines'))

fig_five.add_trace(go.Line(x = country_five['Date'],

                          y = np.log1p(country_five['Death']), 

                          name = 'Death', 

                          mode = 'lines'))

fig_five.add_trace(go.Line(x = country_five['Date'],

                          y = np.log1p(country_five['Recovered']), 

                          name = 'Recovered', 

                          mode = 'lines'))

fig_five.add_trace(go.Line(x = country_five['Date'],

                          y = np.log1p(country_five['Active']), 

                          name = 'Active', 

                          mode = 'lines'))

fig_five.update_layout(title = country_list[4])

fig_five.update_xaxes(title= '------>Timeline' ,showline=False)

fig_five.update_yaxes(title= '------>Lograthmic increment in cases', showline=False)

fig_five.show()



fig_six = go.Figure(go.Line(x = country_six['Date'], 

                            y = np.log1p(country_six['Confirmed']),

                            name = 'Confirmed', 

                            mode = 'lines'))

fig_six.add_trace(go.Line(x = country_six['Date'],

                          y = np.log1p(country_six['Death']), 

                          name = 'Death', 

                          mode = 'lines'))

fig_six.add_trace(go.Line(x = country_six['Date'],

                          y = np.log1p(country_six['Recovered']), 

                          name = 'Recovered', 

                          mode = 'lines'))

fig_six.add_trace(go.Line(x = country_six['Date'],

                          y = np.log1p(country_six['Active']), 

                          name = 'Active', 

                          mode = 'lines'))

fig_six.update_layout(title = country_list[5])

fig_six.update_xaxes(title= '------>Timeline' ,showline=False)

fig_six.update_yaxes(title= '------>Lograthmic increment in cases', showline=False)

fig_six.show()



fig_seven = go.Figure(go.Line(x = country_seven['Date'], 

                            y = np.log1p(country_seven['Confirmed']),

                            name = 'Confirmed', 

                            mode = 'lines'))

fig_seven.add_trace(go.Line(x = country_seven['Date'],

                          y = np.log1p(country_seven['Death']), 

                          name = 'Death', 

                          mode = 'lines'))

fig_seven.add_trace(go.Line(x = country_seven['Date'],

                          y = np.log1p(country_seven['Recovered']), 

                          name = 'Recovered', 

                          mode = 'lines'))

fig_seven.add_trace(go.Line(x = country_seven['Date'],

                          y = np.log1p(country_seven['Active']), 

                          name = 'Active', 

                          mode = 'lines'))

fig_seven.update_layout(title = country_list[6])

fig_seven.update_xaxes(title= '------>Timeline' ,showline=False)

fig_seven.update_yaxes(title= '------>Lograthmic increment in cases', showline=False)

fig_seven.show()



fig_eight = go.Figure(go.Line(x = country_eight['Date'], 

                            y = np.log1p(country_eight['Confirmed']),

                            name = 'Confirmed', 

                            mode = 'lines'))

fig_eight.add_trace(go.Line(x = country_eight['Date'],

                          y = np.log1p(country_eight['Death']), 

                          name = 'Death', 

                          mode = 'lines'))

fig_eight.add_trace(go.Line(x = country_eight['Date'],

                          y = np.log1p(country_eight['Recovered']), 

                          name = 'Recovered', 

                          mode = 'lines'))

fig_eight.add_trace(go.Line(x = country_eight['Date'],

                          y = np.log1p(country_eight['Active']), 

                          name = 'Active', 

                          mode = 'lines'))

fig_eight.update_layout(title = country_list[7])

fig_eight.update_xaxes(title= '------>Timeline' ,showline=False)

fig_eight.update_yaxes(title= '------>Lograthmic increment in cases', showline=False)

fig_eight.show()





fig_nine = go.Figure(go.Line(x = country_nine['Date'], 

                            y = np.log1p(country_nine['Confirmed']),

                            name = 'Confirmed', 

                            mode = 'lines'))

fig_nine.add_trace(go.Line(x = country_nine['Date'],

                          y = np.log1p(country_nine['Death']), 

                          name = 'Death', 

                          mode = 'lines'))

fig_nine.add_trace(go.Line(x = country_nine['Date'],

                          y = np.log1p(country_nine['Recovered']), 

                          name = 'Recovered', 

                          mode = 'lines'))

fig_nine.add_trace(go.Line(x = country_nine['Date'],

                          y = np.log1p(country_nine['Active']), 

                          name = 'Active', 

                          mode = 'lines'))

fig_nine.update_layout(title = country_list[8])

fig_nine.update_xaxes(title= '------>Timeline' ,showline=False)

fig_nine.update_yaxes(title= '------>Lograthmic increment in cases', showline=False)

fig_nine.show()





fig_ten = go.Figure(go.Line(x = country_ten['Date'], 

                            y = np.log1p(country_ten['Confirmed']),

                            name = 'Confirmed', 

                            mode = 'lines'))

fig_ten.add_trace(go.Line(x = country_ten['Date'],

                          y = np.log1p(country_ten['Death']), 

                          name = 'Death', 

                          mode = 'lines'))

fig_ten.add_trace(go.Line(x = country_ten['Date'],

                          y = np.log1p(country_ten['Recovered']), 

                          name = 'Recovered', 

                          mode = 'lines'))

fig_ten.add_trace(go.Line(x = country_ten['Date'],

                          y = np.log1p(country_ten['Active']), 

                          name = 'Active', 

                          mode = 'lines'))

fig_ten.update_layout(title = country_list[9])

fig_ten.update_xaxes(title= '------>Timeline' ,showline=False)

fig_ten.update_yaxes(title= '------>Lograthmic increment in cases', showline=False)

fig_ten.show()

fig_one = px.line(country_one, 

             x = 'Date', 

             y = 'Mortality_Rate')

fig_one.update_layout(title = country_list[0])

fig_one.show()



fig_two = px.line(country_two, 

             x = 'Date', 

             y = 'Mortality_Rate')

fig_two.update_layout(title = country_list[1])

fig_two.show()



fig_three = px.line(country_three, 

             x = 'Date', 

             y = 'Mortality_Rate')

fig_three.update_layout(title = country_list[2])

fig_three.show()



fig_four = px.line(country_four, 

             x = 'Date', 

             y = 'Mortality_Rate')

fig_four.update_layout(title = country_list[3])

fig_four.show()



fig_five = px.line(country_five, 

             x = 'Date', 

             y = 'Mortality_Rate')

fig_five.update_layout(title = country_list[4])

fig_five.show()



fig_six = px.line(country_six, 

             x = 'Date', 

             y = 'Mortality_Rate')

fig_six.update_layout(title = country_list[5])

fig_six.show()



fig_seven = px.line(country_seven, 

             x = 'Date', 

             y = 'Mortality_Rate')

fig_seven.update_layout(title = country_list[6])

fig_seven.show()



fig_eight = px.line(country_eight, 

             x = 'Date', 

             y = 'Mortality_Rate')

fig_eight.update_layout(title = country_list[7])

fig_eight.show()



fig_nine = px.line(country_nine, 

             x = 'Date', 

             y = 'Mortality_Rate')

fig_nine.update_layout(title = country_list[8])

fig_nine.show()



fig_ten = px.line(country_ten, 

             x = 'Date', 

             y = 'Mortality_Rate')

fig_ten.update_layout(title = country_list[9])

fig_ten.show()
fig_one = px.line(country_one, 

             x = 'Date', 

             y = 'Recovery_Rate')

fig_one.update_layout(title = country_list[0])

fig_one.show()



fig_two = px.line(country_two, 

             x = 'Date', 

             y = 'Recovery_Rate')

fig_two.update_layout(title = country_list[1])

fig_two.show()



fig_three = px.line(country_three, 

             x = 'Date', 

             y = 'Recovery_Rate')

fig_three.update_layout(title = country_list[2])

fig_three.show()



fig_four = px.line(country_four, 

             x = 'Date', 

             y = 'Recovery_Rate')

fig_four.update_layout(title = country_list[3])

fig_four.show()



fig_five = px.line(country_five, 

             x = 'Date', 

             y = 'Recovery_Rate')

fig_five.update_layout(title = country_list[4])

fig_five.show()



fig_six = px.line(country_six, 

             x = 'Date', 

             y = 'Recovery_Rate')

fig_six.update_layout(title = country_list[5])

fig_six.show()



fig_seven = px.line(country_seven, 

             x = 'Date', 

             y = 'Recovery_Rate')

fig_seven.update_layout(title = country_list[6])

fig_seven.show()



fig_eight = px.line(country_eight, 

             x = 'Date', 

             y = 'Recovery_Rate')

fig_eight.update_layout(title = country_list[7])

fig_eight.show()



fig_nine = px.line(country_nine, 

             x = 'Date', 

             y = 'Recovery_Rate')

fig_nine.update_layout(title = country_list[8])

fig_nine.show()



fig_ten = px.line(country_ten, 

             x = 'Date', 

             y = 'Recovery_Rate')

fig_ten.update_layout(title = country_list[9])

fig_ten.show()
hospital_beds_per_thousand = country_1.merge(country_2, on = 'date', how = 'left')

hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'hospital_beds_per_thousand_x', 'hospital_beds_per_thousand_y']]

hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand_x' : 'country_1', 'hospital_beds_per_thousand_y' : 'country_2'}, inplace = True)

hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_3, on = 'date', how = 'left')

hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'hospital_beds_per_thousand']]

hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_3'}, inplace = True)

hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_4, on = 'date', how = 'left')

hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'country_3', 'hospital_beds_per_thousand']]

hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_4'}, inplace = True)

hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_5, on = 'date', how = 'left')

hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'hospital_beds_per_thousand']]

hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_5'}, inplace = True)

hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_6, on = 'date', how = 'left')

hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'hospital_beds_per_thousand']]

hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_6'}, inplace = True)

hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_7, on = 'date', how = 'left')

hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'hospital_beds_per_thousand']]

hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_7'}, inplace = True)

hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_8, on = 'date', how = 'left')

hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'country_7', 'hospital_beds_per_thousand']]

hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_8'}, inplace = True)

hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_9, on = 'date', how = 'left')

hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'country_7', 'country_8', 'hospital_beds_per_thousand']]

hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_9'}, inplace = True)

hospital_beds_per_thousand = hospital_beds_per_thousand.merge(country_10, on = 'date', how = 'left')

hospital_beds_per_thousand = hospital_beds_per_thousand[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'country_7', 'country_8', 'country_9', 'hospital_beds_per_thousand']]

hospital_beds_per_thousand.rename(columns = {'hospital_beds_per_thousand' : 'country_10'}, inplace = True)

hospital_beds_per_thousand = hospital_beds_per_thousand.fillna(0)

hospital_beds_per_thousand = hospital_beds_per_thousand.reset_index()

hospital_beds_per_thousand.rename(columns = {'index':'Days'}, inplace = True)
beds = hospital_beds_per_thousand.tail(1)

beds.rename(columns = {'country_1' : country_list[0], 

                       'country_2' : country_list[1],

                       'country_3' : country_list[2],

                       'country_4' : country_list[3],

                       'country_5' : country_list[4],

                       'country_6' : country_list[5],

                       'country_7' : country_list[6],

                       'country_8' : country_list[7],

                       'country_9' : country_list[8],

                       'country_10' : country_list[9],},

            inplace = True)

a = int(beds['Days'])

beds = beds.transpose()

beds = beds.reset_index()

beds.drop([0, 1], inplace = True)

beds.rename(columns = {'index' : 'Country', a : 'Hospital_beds_per_thousand'}, inplace = True)
beds_fig = px.bar(beds,

             x = 'Country',

             y = 'Hospital_beds_per_thousand',

             height = 500,

             title = 'Hospital Beds Per Thousand',

             color = 'Country')

beds_fig.show()
country_1['test_per_confirmed'] = country_1['new_tests_smoothed']/country_1['new_cases']

country_2['test_per_confirmed'] = country_2['new_tests_smoothed']/country_2['new_cases']

country_3['test_per_confirmed'] = country_3['new_tests_smoothed']/country_3['new_cases']

country_4['test_per_confirmed'] = country_4['new_tests_smoothed']/country_4['new_cases']

country_5['test_per_confirmed'] = country_5['new_tests_smoothed']/country_5['new_cases']

country_6['test_per_confirmed'] = country_6['new_tests_smoothed']/country_6['new_cases']

country_7['test_per_confirmed'] = country_7['new_tests_smoothed']/country_7['new_cases']

country_8['test_per_confirmed'] = country_8['new_tests_smoothed']/country_8['new_cases']

country_9['test_per_confirmed'] = country_9['new_tests_smoothed']/country_9['new_cases']

country_10['test_per_confirmed'] = country_10['new_tests_smoothed']/country_10['new_cases']

test_per_confirmed = country_1.merge(country_2, on = 'date', how = 'left')

test_per_confirmed = test_per_confirmed[['date', 'test_per_confirmed_x', 'test_per_confirmed_y']]

test_per_confirmed.rename(columns = {'test_per_confirmed_x' : 'country_1', 'test_per_confirmed_y' : 'country_2'}, inplace = True)

test_per_confirmed = test_per_confirmed.merge(country_3, on = 'date', how = 'left')

test_per_confirmed = test_per_confirmed[['date', 'country_1', 'country_2', 'test_per_confirmed']]

test_per_confirmed.rename(columns = {'test_per_confirmed' : 'country_3'}, inplace = True)

test_per_confirmed = test_per_confirmed.merge(country_4, on = 'date', how = 'left')

test_per_confirmed = test_per_confirmed[['date', 'country_1', 'country_2', 'country_3', 'test_per_confirmed']]

test_per_confirmed.rename(columns = {'test_per_confirmed' : 'country_4'}, inplace = True)

test_per_confirmed = test_per_confirmed.merge(country_5, on = 'date', how = 'left')

test_per_confirmed = test_per_confirmed[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'test_per_confirmed']]

test_per_confirmed.rename(columns = {'test_per_confirmed' : 'country_5'}, inplace = True)

test_per_confirmed = test_per_confirmed.merge(country_6, on = 'date', how = 'left')

test_per_confirmed = test_per_confirmed[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'test_per_confirmed']]

test_per_confirmed.rename(columns = {'test_per_confirmed' : 'country_6'}, inplace = True)

test_per_confirmed = test_per_confirmed.merge(country_7, on = 'date', how = 'left')

test_per_confirmed = test_per_confirmed[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'test_per_confirmed']]

test_per_confirmed.rename(columns = {'test_per_confirmed' : 'country_7'}, inplace = True)

test_per_confirmed = test_per_confirmed.merge(country_8, on = 'date', how = 'left')

test_per_confirmed = test_per_confirmed[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'country_7', 'test_per_confirmed']]

test_per_confirmed.rename(columns = {'test_per_confirmed' : 'country_8'}, inplace = True)

test_per_confirmed = test_per_confirmed.merge(country_9, on = 'date', how = 'left')

test_per_confirmed = test_per_confirmed[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'country_7', 'country_8', 'test_per_confirmed']]

test_per_confirmed.rename(columns = {'test_per_confirmed' : 'country_9'}, inplace = True)

test_per_confirmed = test_per_confirmed.merge(country_10, on = 'date', how = 'left')

test_per_confirmed = test_per_confirmed[['date', 'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'country_7', 'country_8', 'country_9', 'test_per_confirmed']]

test_per_confirmed.rename(columns = {'test_per_confirmed' : 'country_10'}, inplace = True)

test_per_confirmed = test_per_confirmed.fillna(0)

test_per_confirmed = test_per_confirmed.reset_index()

test_per_confirmed.rename(columns = {'index':'Days'}, inplace = True)
test_per_confirmed.set_index('date', inplace = True)

test_per_confirmed = test_per_confirmed.rolling(7).mean()

test_per_confirmed.reset_index(inplace = True)

test_per_confirmed.rename(columns = {'index' : 'date'})
test_per_confirmed = test_per_confirmed.fillna(0)
fig = go.Figure()

fig.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_1'],

                    mode='lines',

                    name=country_list[0]))

fig.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_2'],

                    mode='lines',

                    name=country_list[1]))

fig.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_3'],

                    mode='lines',

                    name=country_list[2]))

fig.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_4'],

                    mode='lines',

                    name=country_list[3]))

fig.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_5'],

                    mode='lines',

                    name=country_list[4]))

fig.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_6'],

                    mode='lines', 

                    name=country_list[5]))

fig.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_7'],

                    mode='lines',

                    name=country_list[6]))

fig.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_8'],

                    mode='lines',

                    name=country_list[7]))

fig.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_9'],

                    mode='lines', 

                    name=country_list[8]))

fig.add_trace(go.Scatter(x=test_per_confirmed['date'], y=test_per_confirmed['country_10'],

                    mode='lines',

                    name=country_list[9]))

fig.update_layout(title = 'Test conducted per confirmed case (7 day rolling average)')

fig.update_xaxes(title= '------>Timeline' ,showline=False)

fig.update_yaxes(title= '------>Tests / Confirmed Case', showline=False)

fig.show()
confirmed_daily_pct_change = confirmed.groupby('Country/Region').agg(sum)

confirmed_daily_pct_change = confirmed_daily_pct_change.drop(columns = ['Lat', 'Long'])

confirmed_daily_pct_change = confirmed_daily_pct_change.diff(axis = 1, periods = 1)

confirmed_daily_pct_change = confirmed_daily_pct_change.pct_change(axis = 1, periods = 1)*100

confirmed_daily_pct_change = confirmed_daily_pct_change.fillna(0)

confirmed_daily_pct_change[confirmed_daily_pct_change == inf] = 0

confirmed_daily_pct_change[confirmed_daily_pct_change == -inf] = 0

confirmed_daily_pct_change = confirmed_daily_pct_change.reset_index()

confirmed_daily_pct_change.loc[confirmed_daily_pct_change['Country/Region'] == 'US', 'Country/Region'] = 'United States'

confirmed_daily_pct_change = confirmed_daily_pct_change.loc[confirmed_daily_pct_change['Country/Region'].isin(country_list)]

confirmed_daily_pct_change = confirmed_daily_pct_change.transpose()

confirmed_daily_pct_change.columns = confirmed_daily_pct_change.iloc[0]

confirmed_daily_pct_change = confirmed_daily_pct_change[1:]

confirmed_daily_pct_change = confirmed_daily_pct_change.reset_index()

confirmed_daily_pct_change.rename(columns = {'index' : 'date'}, inplace = True)
# fig_confirmed_daily_pct_change = go.Figure()

# fig_confirmed_daily_pct_change.add_trace(go.Scatter(x=confirmed_daily_pct_change['date'], 

#                                                 y=confirmed_daily_pct_change[country_list[0]],

#                                                 mode='lines+markers',

#                                                 name=country_list[0]))

# fig_confirmed_daily_pct_change.add_trace(go.Scatter(x=confirmed_daily_pct_change['date'],

#                                                 y=confirmed_daily_pct_change[country_list[1]],

#                                                 mode='lines+markers',

#                                                 name=country_list[1]))

# fig_confirmed_daily_pct_change.add_trace(go.Scatter(x=confirmed_daily_pct_change['date'],

#                                                 y=confirmed_daily_pct_change[country_list[2]],

#                                                 mode='lines+markers',

#                                                 name=country_list[2]))

# fig_confirmed_daily_pct_change.add_trace(go.Scatter(x=confirmed_daily_pct_change['date'],

#                                                 y=confirmed_daily_pct_change[country_list[3]],

#                                                 mode='lines+markers',

#                                                 name=country_list[3]))

# fig_confirmed_daily_pct_change.add_trace(go.Scatter(x=confirmed_daily_pct_change['date'],

#                                                 y=confirmed_daily_pct_change[country_list[4]],

#                                                 mode='lines+markers',

#                                                 name=country_list[4]))

# fig_confirmed_daily_pct_change.add_trace(go.Scatter(x=confirmed_daily_pct_change['date'],

#                                                 y=confirmed_daily_pct_change[country_list[5]],

#                                                 mode='lines+markers', 

#                                                 name=country_list[5]))

# fig_confirmed_daily_pct_change.add_trace(go.Scatter(x=confirmed_daily_pct_change['date'],

#                                                 y=confirmed_daily_pct_change[country_list[6]],

#                                                 mode='lines+markers',

#                                                 name=country_list[6]))

# fig_confirmed_daily_pct_change.add_trace(go.Scatter(x=confirmed_daily_pct_change['date'],

#                                                 y=confirmed_daily_pct_change[country_list[7]],

#                                                 mode='lines+markers',

#                                                 name=country_list[7]))

# fig_confirmed_daily_pct_change.add_trace(go.Scatter(x=confirmed_daily_pct_change['date'],

#                                                 y=confirmed_daily_pct_change[country_list[8]],

#                                                 mode='lines+markers', 

#                                                 name=country_list[8]))

# fig_confirmed_daily_pct_change.add_trace(go.Scatter(x=confirmed_daily_pct_change['date'],

#                                                 y=confirmed_daily_pct_change[country_list[9]],

#                                                 mode='lines+markers',

#                                                 name=country_list[9]))

# fig_confirmed_daily_pct_change.update_layout(title = 'Day on Day %age change in case')

# fig_confirmed_daily_pct_change.update_xaxes(title= '------>Timeline' ,showline=False)

# fig_confirmed_daily_pct_change.update_yaxes(title= '------>%age change', showline=False)

# fig_confirmed_daily_pct_change.show()
fig = make_subplots(

    rows=10, cols=1,

    subplot_titles=(country_list[0], country_list[1],

                    country_list[2], country_list[3],

                    country_list[4], country_list[5],

                    country_list[6], country_list[7],

                    country_list[8], country_list[9]

                    ))



fig.add_trace(go.Scatter(x=confirmed_daily_pct_change['date'],

                         y=confirmed_daily_pct_change[country_list[0]],

                         mode='lines+markers',

                         name=country_list[0]),

             row=1, col=1)



fig.add_trace(go.Scatter(x=confirmed_daily_pct_change['date'],

                         y=confirmed_daily_pct_change[country_list[1]],

                        mode='lines+markers',

                        name=country_list[1]),

              row=2, col=1)



fig.add_trace(go.Scatter(x=confirmed_daily_pct_change['date'],

                         y=confirmed_daily_pct_change[country_list[2]],

                        mode='lines+markers',

                        name=country_list[2]),

              row=3, col=1)



fig.add_trace(go.Scatter(x=confirmed_daily_pct_change['date'],

                         y=confirmed_daily_pct_change[country_list[3]],

                        mode='lines+markers',

                        name=country_list[3]),

              row=4, col=1)



fig.add_trace(go.Scatter(x=confirmed_daily_pct_change['date'],

                         y=confirmed_daily_pct_change[country_list[4]],

                        mode='lines+markers',

                        name=country_list[4]),

              row=5, col=1)



fig.add_trace(go.Scatter(x=confirmed_daily_pct_change['date'],

                         y=confirmed_daily_pct_change[country_list[5]],

                        mode='lines+markers',

                        name=country_list[5]),

              row=6, col=1)



fig.add_trace(go.Scatter(x=confirmed_daily_pct_change['date'],

                         y=confirmed_daily_pct_change[country_list[6]],

                        mode='lines+markers',

                        name=country_list[6]),

              row=7, col=1)



fig.add_trace(go.Scatter(x=confirmed_daily_pct_change['date'],

                         y=confirmed_daily_pct_change[country_list[7]],

                        mode='lines+markers',

                        name=country_list[7]),

              row=8, col=1)



fig.add_trace(go.Scatter(x=confirmed_daily_pct_change['date'],

                         y=confirmed_daily_pct_change[country_list[8]],

                        mode='lines+markers',

                        name=country_list[8]),

              row=9, col=1)



fig.add_trace(go.Scatter(x=confirmed_daily_pct_change['date'],

                         y=confirmed_daily_pct_change[country_list[9]],

                        mode='lines+markers',

                        name=country_list[9]),

              row=10, col=1)



fig.update_xaxes(title_text="------------> Timeline", row=1, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=2, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=3, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=4, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=5, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=6, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=7, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=8, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=9, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=10, col=1)



fig.update_yaxes(title_text="-----> Recovered", row=1, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=2, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=3, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=4, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=5, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=6, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=7, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=8, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=9, col=1)

fig.update_yaxes(title_text="----->Recovered", row=10, col=1)



fig.update_layout(height=3000, width=1300, title_text="Day on Day %age Change in number of Confirmed Patients")



fig.show()
death_daily_pct_change = death.groupby('Country/Region').agg(sum)

death_daily_pct_change = death_daily_pct_change.drop(columns = ['Lat', 'Long'])

death_daily_pct_change = death_daily_pct_change.diff(axis = 1, periods = 1)

death_daily_pct_change = death_daily_pct_change.pct_change(axis = 1, periods = 1)*100

death_daily_pct_change = death_daily_pct_change.fillna(0)

death_daily_pct_change[death_daily_pct_change == inf] = 0

death_daily_pct_change[death_daily_pct_change == -inf] = 0

death_daily_pct_change = death_daily_pct_change.reset_index()

death_daily_pct_change.loc[death_daily_pct_change['Country/Region'] == 'US', 'Country/Region'] = 'United States'

death_daily_pct_change = death_daily_pct_change.loc[death_daily_pct_change['Country/Region'].isin(country_list)]

death_daily_pct_change = death_daily_pct_change.transpose()

death_daily_pct_change.columns = death_daily_pct_change.iloc[0]

death_daily_pct_change = death_daily_pct_change[1:]

death_daily_pct_change = death_daily_pct_change.reset_index()

death_daily_pct_change.rename(columns = {'index' : 'date'}, inplace = True)
# fig_death_daily_pct_change = go.Figure()

# fig_death_daily_pct_change.add_trace(go.Scatter(x=death_daily_pct_change['date'], 

#                                                 y=death_daily_pct_change[country_list[0]],

#                                                 mode='lines+markers',

#                                                 name=country_list[0]))

# fig_death_daily_pct_change.add_trace(go.Scatter(x=death_daily_pct_change['date'],

#                                                 y=death_daily_pct_change[country_list[1]],

#                                                 mode='lines+markers',

#                                                 name=country_list[1]))

# fig_death_daily_pct_change.add_trace(go.Scatter(x=death_daily_pct_change['date'],

#                                                 y=death_daily_pct_change[country_list[2]],

#                                                 mode='lines+markers',

#                                                 name=country_list[2]))

# fig_death_daily_pct_change.add_trace(go.Scatter(x=death_daily_pct_change['date'],

#                                                 y=death_daily_pct_change[country_list[3]],

#                                                 mode='lines+markers',

#                                                 name=country_list[3]))

# fig_death_daily_pct_change.add_trace(go.Scatter(x=death_daily_pct_change['date'],

#                                                 y=death_daily_pct_change[country_list[4]],

#                                                 mode='lines+markers',

#                                                 name=country_list[4]))

# fig_death_daily_pct_change.add_trace(go.Scatter(x=death_daily_pct_change['date'],

#                                                 y=death_daily_pct_change[country_list[5]],

#                                                 mode='lines+markers', 

#                                                 name=country_list[5]))

# fig_death_daily_pct_change.add_trace(go.Scatter(x=death_daily_pct_change['date'],

#                                                 y=death_daily_pct_change[country_list[6]],

#                                                 mode='lines+markers',

#                                                 name=country_list[6]))

# fig_death_daily_pct_change.add_trace(go.Scatter(x=death_daily_pct_change['date'],

#                                                 y=death_daily_pct_change[country_list[7]],

#                                                 mode='lines+markers',

#                                                 name=country_list[7]))

# fig_death_daily_pct_change.add_trace(go.Scatter(x=death_daily_pct_change['date'],

#                                                 y=death_daily_pct_change[country_list[8]],

#                                                 mode='lines+markers', 

#                                                 name=country_list[8]))

# fig_death_daily_pct_change.add_trace(go.Scatter(x=death_daily_pct_change['date'],

#                                                 y=death_daily_pct_change[country_list[9]],

#                                                 mode='lines+markers',

#                                                 name=country_list[9]))

# fig_death_daily_pct_change.update_layout(title = 'Day on Day %age change in case')

# fig_death_daily_pct_change.update_xaxes(title= '------>Timeline' ,showline=False)

# fig_death_daily_pct_change.update_yaxes(title= '------>%age change', showline=False)

# fig_death_daily_pct_change.show()
fig = make_subplots(

    rows=10, cols=1,

    subplot_titles=(country_list[0], country_list[1],

                    country_list[2], country_list[3],

                    country_list[4], country_list[5],

                    country_list[6], country_list[7],

                    country_list[8], country_list[9]

                    ))



fig.add_trace(go.Scatter(x=death_daily_pct_change['date'],

                         y=death_daily_pct_change[country_list[0]],

                         mode='lines+markers',

                         name=country_list[0]),

             row=1, col=1)



fig.add_trace(go.Scatter(x=death_daily_pct_change['date'],

                         y=death_daily_pct_change[country_list[1]],

                        mode='lines+markers',

                        name=country_list[1]),

              row=2, col=1)



fig.add_trace(go.Scatter(x=death_daily_pct_change['date'],

                         y=death_daily_pct_change[country_list[2]],

                        mode='lines+markers',

                        name=country_list[2]),

              row=3, col=1)



fig.add_trace(go.Scatter(x=death_daily_pct_change['date'],

                         y=death_daily_pct_change[country_list[3]],

                        mode='lines+markers',

                        name=country_list[3]),

              row=4, col=1)



fig.add_trace(go.Scatter(x=death_daily_pct_change['date'],

                         y=death_daily_pct_change[country_list[4]],

                        mode='lines+markers',

                        name=country_list[4]),

              row=5, col=1)



fig.add_trace(go.Scatter(x=death_daily_pct_change['date'],

                         y=death_daily_pct_change[country_list[5]],

                        mode='lines+markers',

                        name=country_list[5]),

              row=6, col=1)



fig.add_trace(go.Scatter(x=death_daily_pct_change['date'],

                         y=death_daily_pct_change[country_list[6]],

                        mode='lines+markers',

                        name=country_list[6]),

              row=7, col=1)



fig.add_trace(go.Scatter(x=death_daily_pct_change['date'],

                         y=death_daily_pct_change[country_list[7]],

                        mode='lines+markers',

                        name=country_list[7]),

              row=8, col=1)



fig.add_trace(go.Scatter(x=death_daily_pct_change['date'],

                         y=death_daily_pct_change[country_list[8]],

                        mode='lines+markers',

                        name=country_list[8]),

              row=9, col=1)



fig.add_trace(go.Scatter(x=death_daily_pct_change['date'],

                         y=death_daily_pct_change[country_list[9]],

                        mode='lines+markers',

                        name=country_list[9]),

              row=10, col=1)



fig.update_xaxes(title_text="------------> Timeline", row=1, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=2, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=3, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=4, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=5, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=6, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=7, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=8, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=9, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=10, col=1)



fig.update_yaxes(title_text="-----> Recovered", row=1, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=2, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=3, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=4, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=5, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=6, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=7, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=8, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=9, col=1)

fig.update_yaxes(title_text="----->Recovered", row=10, col=1)



fig.update_layout(height=3000, width=1300, title_text="Day on Day %age Change in number of Patients Deceased")



fig.show()
recovered_daily_pct_change = recovered.groupby('Country/Region').agg(sum)

recovered_daily_pct_change = recovered_daily_pct_change.drop(columns = ['Lat', 'Long'])

recovered_daily_pct_change = recovered_daily_pct_change.diff(axis = 1, periods = 1)

recovered_daily_pct_change = recovered_daily_pct_change.pct_change(axis = 1, periods = 1)*100

recovered_daily_pct_change = recovered_daily_pct_change.fillna(0)

recovered_daily_pct_change[recovered_daily_pct_change == inf] = 0

recovered_daily_pct_change[recovered_daily_pct_change == -inf] = 0

recovered_daily_pct_change = recovered_daily_pct_change.reset_index()

recovered_daily_pct_change.loc[recovered_daily_pct_change['Country/Region'] == 'US', 'Country/Region'] = 'United States'

recovered_daily_pct_change = recovered_daily_pct_change.loc[recovered_daily_pct_change['Country/Region'].isin(country_list)]

recovered_daily_pct_change = recovered_daily_pct_change.transpose()

recovered_daily_pct_change.columns = recovered_daily_pct_change.iloc[0]

recovered_daily_pct_change = recovered_daily_pct_change[1:]

recovered_daily_pct_change = recovered_daily_pct_change.reset_index()

recovered_daily_pct_change.rename(columns = {'index' : 'date'}, inplace = True)
# fig = make_subplots(

#     rows=10, cols=1,

#     subplot_titles=(country_list[0], country_list[1],

#                     country_list[2], country_list[3],

#                     country_list[4], country_list[5],

#                     country_list[6], country_list[7],

#                     country_list[8], country_list[9]

#                     ))



# fig.add_trace(go.Scatter(x=recovered_daily_pct_change['date'],

#                          y=recovered_daily_pct_change[country_list[0]],

#                          mode='lines+markers',

#                          name=country_list[0]),

#              row=1, col=1)



# fig.add_trace(go.Scatter(x=recovered_daily_pct_change['date'],

#                          y=recovered_daily_pct_change[country_list[1]],

#                         mode='lines+markers',

#                         name=country_list[1]),

#               row=2, col=1)



# fig.add_trace(go.Scatter(x=recovered_daily_pct_change['date'],

#                          y=recovered_daily_pct_change[country_list[2]],

#                         mode='lines+markers',

#                         name=country_list[2]),

#               row=3, col=1)



# fig.add_trace(go.Scatter(x=recovered_daily_pct_change['date'],

#                          y=recovered_daily_pct_change[country_list[3]],

#                         mode='lines+markers',

#                         name=country_list[3]),

#               row=4, col=1)



# fig.add_trace(go.Scatter(x=recovered_daily_pct_change['date'],

#                          y=recovered_daily_pct_change[country_list[4]],

#                         mode='lines+markers',

#                         name=country_list[4]),

#               row=5, col=1)



# fig.add_trace(go.Scatter(x=recovered_daily_pct_change['date'],

#                          y=recovered_daily_pct_change[country_list[5]],

#                         mode='lines+markers',

#                         name=country_list[5]),

#               row=6, col=1)



# fig.add_trace(go.Scatter(x=recovered_daily_pct_change['date'],

#                          y=recovered_daily_pct_change[country_list[6]],

#                         mode='lines+markers',

#                         name=country_list[6]),

#               row=7, col=1)



# fig.add_trace(go.Scatter(x=recovered_daily_pct_change['date'],

#                          y=recovered_daily_pct_change[country_list[7]],

#                         mode='lines+markers',

#                         name=country_list[7]),

#               row=8, col=1)



# fig.add_trace(go.Scatter(x=recovered_daily_pct_change['date'],

#                          y=recovered_daily_pct_change[country_list[8]],

#                         mode='lines+markers',

#                         name=country_list[8]),

#               row=9, col=1)



# fig.add_trace(go.Scatter(x=recovered_daily_pct_change['date'],

#                          y=recovered_daily_pct_change[country_list[9]],

#                         mode='lines+markers',

#                         name=country_list[9]),

#               row=10, col=1)

# fig.update_xaxes(title_text="------------> Timeline", row=1, col=1)

# fig.update_xaxes(title_text="------------> Timeline", row=2, col=1)

# fig.update_xaxes(title_text="------------> Timeline", row=3, col=1)

# fig.update_xaxes(title_text="------------> Timeline", row=4, col=1)

# fig.update_xaxes(title_text="------------> Timeline", row=5, col=1)

# fig.update_xaxes(title_text="------------> Timeline", row=6, col=1)

# fig.update_xaxes(title_text="------------> Timeline", row=7, col=1)

# fig.update_xaxes(title_text="------------> Timeline", row=8, col=1)

# fig.update_xaxes(title_text="------------> Timeline", row=9, col=1)

# fig.update_xaxes(title_text="------------> Timeline", row=10, col=1)



# fig.update_yaxes(title_text="-----> Recovered", row=1, col=1)

# fig.update_yaxes(title_text="-----> Recovered", row=2, col=1)

# fig.update_yaxes(title_text="-----> Recovered", row=3, col=1)

# fig.update_yaxes(title_text="-----> Recovered", row=4, col=1)

# fig.update_yaxes(title_text="-----> Recovered", row=5, col=1)

# fig.update_yaxes(title_text="-----> Recovered", row=6, col=1)

# fig.update_yaxes(title_text="-----> Recovered", row=7, col=1)

# fig.update_yaxes(title_text="-----> Recovered", row=8, col=1)

# fig.update_yaxes(title_text="-----> Recovered", row=9, col=1)

# fig.update_yaxes(title_text="----->Recovered", row=10, col=1)



# fig.update_layout(height=3000, width=1300, title_text="Day on Day %age Change in number of Recovered Patients")



# fig.show()
fig = make_subplots(

    rows=10, cols=1,

    subplot_titles=(country_list[0], country_list[1],

                    country_list[2], country_list[3],

                    country_list[4], country_list[5],

                    country_list[6], country_list[7],

                    country_list[8], country_list[9]

                    ))



fig.add_trace(go.Scatter(x=recovered_daily_pct_change['date'],

                         y=recovered_daily_pct_change[country_list[0]],

                         mode='lines+markers',

                         name=country_list[0]),

             row=1, col=1)



fig.add_trace(go.Scatter(x=recovered_daily_pct_change['date'],

                         y=recovered_daily_pct_change[country_list[1]],

                        mode='lines+markers',

                        name=country_list[1]),

              row=2, col=1)



fig.add_trace(go.Scatter(x=recovered_daily_pct_change['date'],

                         y=recovered_daily_pct_change[country_list[2]],

                        mode='lines+markers',

                        name=country_list[2]),

              row=3, col=1)



fig.add_trace(go.Scatter(x=recovered_daily_pct_change['date'],

                         y=recovered_daily_pct_change[country_list[3]],

                        mode='lines+markers',

                        name=country_list[3]),

              row=4, col=1)



fig.add_trace(go.Scatter(x=recovered_daily_pct_change['date'],

                         y=recovered_daily_pct_change[country_list[4]],

                        mode='lines+markers',

                        name=country_list[4]),

              row=5, col=1)



fig.add_trace(go.Scatter(x=recovered_daily_pct_change['date'],

                         y=recovered_daily_pct_change[country_list[5]],

                        mode='lines+markers',

                        name=country_list[5]),

              row=6, col=1)



fig.add_trace(go.Scatter(x=recovered_daily_pct_change['date'],

                         y=recovered_daily_pct_change[country_list[6]],

                        mode='lines+markers',

                        name=country_list[6]),

              row=7, col=1)



fig.add_trace(go.Scatter(x=recovered_daily_pct_change['date'],

                         y=recovered_daily_pct_change[country_list[7]],

                        mode='lines+markers',

                        name=country_list[7]),

              row=8, col=1)



fig.add_trace(go.Scatter(x=recovered_daily_pct_change['date'],

                         y=recovered_daily_pct_change[country_list[8]],

                        mode='lines+markers',

                        name=country_list[8]),

              row=9, col=1)



fig.add_trace(go.Scatter(x=recovered_daily_pct_change['date'],

                         y=recovered_daily_pct_change[country_list[9]],

                        mode='lines+markers',

                        name=country_list[9]),

              row=10, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=1, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=2, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=3, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=4, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=5, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=6, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=7, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=8, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=9, col=1)

fig.update_xaxes(title_text="------------> Timeline", row=10, col=1)



fig.update_yaxes(title_text="-----> Recovered", row=1, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=2, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=3, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=4, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=5, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=6, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=7, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=8, col=1)

fig.update_yaxes(title_text="-----> Recovered", row=9, col=1)

fig.update_yaxes(title_text="----->Recovered", row=10, col=1)



fig.update_layout(height=3000, width=1300, title_text="Day on Day %age Change in number of Recovered Patients")



fig.show()
raw_data_1 = pd.read_csv('https://api.covid19india.org/csv/latest/raw_data1.csv')

raw_data_2 = pd.read_csv('https://api.covid19india.org/csv/latest/raw_data2.csv')

raw_data_3 = pd.read_csv('https://api.covid19india.org/csv/latest/raw_data3.csv')

raw_data_4 = pd.read_csv('https://api.covid19india.org/csv/latest/raw_data4.csv')

raw_data_5 = pd.read_csv('https://api.covid19india.org/csv/latest/raw_data5.csv')

status_1 = pd.read_csv('https://api.covid19india.org/csv/latest/death_and_recovered1.csv')

status_2 = pd.read_csv('https://api.covid19india.org/csv/latest/death_and_recovered2.csv')

statewise = pd.read_csv('https://api.covid19india.org/csv/latest/state_wise.csv')

case_time_series = pd.read_csv('https://api.covid19india.org/csv/latest/case_time_series.csv')

districtwise = pd.read_csv('https://api.covid19india.org/csv/latest/district_wise.csv')

statewise_tested = pd.read_csv('https://api.covid19india.org/csv/latest/statewise_tested_numbers_data.csv')

icmr_tested = pd.read_csv('https://api.covid19india.org/csv/latest/tested_numbers_icmr_data.csv')
total = pd.DataFrame(statewise[statewise['State'] == 'Total'])

total = total[['Recovered', 'Deaths', 'Active']]

total = total.transpose()

total = total.reset_index()

total.rename(columns = {'index' : 'Property', 0 : 'Numbers'}, inplace = True)

total
fig_total = px.pie(total, 

                  values = 'Numbers', 

                  names = 'Property', 

                  title = 'Current COVID-19 Status',

                  color_discrete_map={'Active':'blue',

                                 'Recovered':'green',

                                 'Deaths':'red'})

fig_total.show()
statewise = statewise.drop([0])
df_confirmed = statewise[['State', 'Confirmed']]

df_recovered = statewise[['State', 'Recovered']]

df_death = statewise[['State', 'Deaths']]

df_active = statewise[['State', 'Active']]
fig_statewise = make_subplots(rows=2, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}], [{"type": "pie"}, {"type": "pie"}]])



fig_statewise.add_trace(go.Pie(

     values=df_confirmed['Confirmed'],

     labels=df_confirmed['State'],

     domain=dict(x=[0, 0.5]),

     title_text="Confirmed Cases"), 

     row=1, col=1)



fig_statewise.add_trace(go.Pie(

     values=df_active['Active'],

     labels=df_active['State'],

     domain=dict(x=[0.5, 1.0]),

     title_text="Active Cases"),

    row=1, col=2)



fig_statewise.add_trace(go.Pie(

     values=df_recovered['Recovered'],

     labels=df_recovered['State'],

     domain=dict(x=[0, 0.5]),

     title_text="Recovered"),

    row=2, col=1)



fig_statewise.add_trace(go.Pie(

     values=df_death['Deaths'],

     labels=df_death['State'],

     domain=dict(x=[0.5, 1.0]),

     title_text="Deaths"),

    row=2, col=2)



fig_statewise.update_traces(hoverinfo='label+percent+name', textinfo='none')

fig_statewise.show()
statewise_5k = statewise[statewise['Confirmed'] >= 5000]

statewise_5k = statewise_5k[statewise_5k['State'] != 'State Unassigned']

statewise_5k = statewise_5k[statewise_5k['State'] != 'Total']
fig = go.Figure(data=[

    go.Bar(name='Confirmed', x=statewise_5k['State'], y=statewise_5k['Confirmed'], marker_color = 'Blue'),

    go.Bar(name='Recovered', x=statewise_5k['State'], y=statewise_5k['Recovered'], marker_color = 'Green'),

    go.Bar(name='Deceased', x=statewise_5k['State'], y=statewise_5k['Deaths'], marker_color = 'Red')

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
net_updated = statewise[['State', 'Confirmed', 'Active', 'Recovered', 'Deaths']]

net_updated['Mortaliy Rate'] = net_updated['Deaths']/net_updated['Confirmed']

net_updated['Recovery Rate'] = net_updated['Recovered']/net_updated['Confirmed']

net_updated = net_updated.fillna(0)

net_updated.style.background_gradient(cmap = 'Reds')
delta_updated = statewise[(statewise['Delta_Confirmed'] > 0) | (statewise['Delta_Recovered'] > 0) | (statewise['Delta_Deaths'] > 0)]

delta_updated = delta_updated[['State', 'Last_Updated_Time', 'Delta_Confirmed', 'Delta_Recovered', 'Delta_Deaths']]

delta_updated.style.background_gradient(cmap='Blues')
beds = pd.read_json('https://api.rootnet.in/covid19-in/hospitals/beds.json')

beds = pd.DataFrame(beds['data']['regional'])
statewise_beds = beds[['state', 'totalBeds']]

statewise_beds = statewise_beds[statewise_beds['state'] != 'INDIA']

statewise_beds = statewise_beds.sort_values(by = 'totalBeds', ascending = False)

statewise_beds.style.background_gradient(cmap='Greens')
fig = px.bar(statewise_beds, 

            x = 'totalBeds',

            y = 'state', 

            orientation = 'h',

            title = 'Hospital beds in each state', 

            color = 'state')

fig.show()
urban_rural_beds = beds[['state', 'urbanHospitals','ruralHospitals']]

urban_rural_beds = urban_rural_beds[urban_rural_beds['state'] != 'INDIA']

urban_rural_beds.style.background_gradient(cmap='Greys')
fig = go.Figure(data=[

    go.Bar(name='Urban Hospitals', x=urban_rural_beds['state'], y=urban_rural_beds['urbanHospitals']),

    go.Bar(name='Rural Hospitals', x=urban_rural_beds['state'], y=urban_rural_beds['ruralHospitals'])

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
urban_rural_beds = beds[['state', 'urbanBeds','ruralBeds']]

urban_rural_beds = urban_rural_beds[urban_rural_beds['state'] != 'INDIA']

urban_rural_beds.style.background_gradient(cmap='Blues')
fig = go.Figure(data=[

    go.Bar(name='Urban Beds', x=urban_rural_beds['state'], y=urban_rural_beds['urbanBeds']),

    go.Bar(name='Rural Beds', x=urban_rural_beds['state'], y=urban_rural_beds['ruralBeds'])

])

fig.update_layout(barmode='group')

fig.show()
samples_tested = icmr_tested[['Update Time Stamp', 'Total Samples Tested']]

samples_tested = samples_tested.set_index('Update Time Stamp')

samples_tested = samples_tested.diff()

samples_tested = samples_tested.reset_index()

samples_tested['Update Time Stamp'] = pd.to_datetime(samples_tested['Update Time Stamp'])

samples_tested['Update Time Stamp'] = samples_tested['Update Time Stamp'].dt.strftime('%d-%m-%Y')

samples_tested['Date'] = pd.DatetimeIndex(samples_tested['Update Time Stamp']).date

samples_tested['Month'] = pd.DatetimeIndex(samples_tested['Update Time Stamp']).month

samples_tested = samples_tested[['Date', 'Month', 'Total Samples Tested']]

samples_tested = samples_tested.fillna(0)

samples_tested.head()
fig_daily_tested = px.scatter(samples_tested,

                          x = 'Date',

                          y = 'Total Samples Tested',

                          color = 'Month',

                          hover_data = ['Date', 'Total Samples Tested'],

                          size = 'Total Samples Tested',

                          title = 'Number of Samples Tested Daily')

fig_daily_tested.show()
indivisuals_tested_pm = icmr_tested[['Update Time Stamp', 'Tests per million']]

indivisuals_tested_pm['Date'] = pd.to_datetime(indivisuals_tested_pm['Update Time Stamp'])

indivisuals_tested_pm['Date'] = indivisuals_tested_pm['Date'].dt.strftime('%d-%m-%Y')

indivisuals_tested_pm['Month'] = pd.DatetimeIndex(indivisuals_tested_pm['Date']).month

indivisuals_tested_pm = indivisuals_tested_pm.fillna(0)
fig_pm = px.scatter(indivisuals_tested_pm,

                   x = 'Date',

                   y = 'Tests per million', 

                   size = 'Tests per million',

                   color = 'Month',

                   hover_data = ['Date', 'Tests per million'])

fig_pm.show()

states = districtwise['State'].unique()

state = pd.DataFrame()

i = 1

for col in states:

    while(i < len(states)):

        x = districtwise[districtwise['State'] == states[i]].sort_values(by = 'Confirmed', ascending = False).head(5)[['State', 'District', 'Confirmed']]

        state = pd.concat([state, x])

        i = i+1

state = state[(state['District'] != 'Unknown')]

state.style.background_gradient(cmap = 'Reds')
trend = case_time_series[['Date', 'Total Confirmed', 'Total Recovered', 'Total Deceased']]

trend['Total Active'] = trend['Total Confirmed'] - (trend['Total Recovered'] + trend['Total Deceased'])
fig_trend = go.Figure()



fig_trend.add_trace(go.Scatter(

    x=trend['Date'], 

    y=trend['Total Confirmed'],

    mode = 'lines+markers',

    name = 'Confirmed'

    ))

fig_trend.add_trace(go.Scatter(

    x=trend['Date'], 

    y=trend['Total Active'],

    mode = 'lines+markers',

    name = 'Active'

))

fig_trend.add_trace(go.Scatter(

    x=trend['Date'],

    y=trend['Total Recovered'],

    mode = 'lines+markers',

    name = 'Recovered'

))

fig_trend.add_trace(go.Scatter(

    x=trend['Date'],

    y=trend['Total Deceased'],

    mode = 'lines+markers',

    name = 'Deceased'

))



fig_trend.update_layout(title = 'Trends', showlegend=True)

fig_trend.update_xaxes(title = '------>Timeline', showline = False)

fig_trend.update_yaxes(title = '------>Numbers', showline = False)

fig_trend.show()