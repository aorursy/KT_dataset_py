from urllib.request import urlopen
import json
import numpy as np
import pandas as pd
import plotly.express as px 

census = pd.read_csv('https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv', encoding = "ISO-8859-1")
census.head()
census.columns.values.tolist()
census.dtypes
census_state = census[census.COUNTY == 0]

census = census[census.COUNTY != 0]
census_state.head()
census.head()
#FIPS data needs to be stored as a string
census["STATE"] = census["STATE"].astype(str)
census["COUNTY"] = census["COUNTY"].astype(str)

#These loops allow us to insert any missing 0s on the beginning of state of county codes.
for i in range(len(census)):
    if len(census["STATE"].iloc[i]) < 2:
        census["STATE"].iloc[i] = "0"+census["STATE"].iloc[i]

for j in range(2):
    for i in range(len(census)):
        if len(census["COUNTY"].iloc[i]) < 3:
            census["COUNTY"].iloc[i] = "0"+census["COUNTY"].iloc[i]

census["STATE"].iloc[1]
census["FIPS"]= census["STATE"]+census["COUNTY"]
census_pop = census[["FIPS","POPESTIMATE2019","CTYNAME"]].copy()
census_pop.head()
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'   

covid = pd.read_csv(url)

covid.head()
covid = covid.dropna()

#This will get rid of the decimal point
covid["FIPS"] = covid["FIPS"].astype(int)

#FIPS need to be stored as a string
covid["FIPS"] = covid["FIPS"].astype(str)

#This allows us to place the lead 0 on if it was missing
for i in range(len(covid)):
    if len(covid["FIPS"].iloc[i]) < 5:
        covid["FIPS"].iloc[i] = "0"+covid["FIPS"].iloc[i]

covid.head()
covid_new = pd.merge(covid, census_pop, how="left", on=['FIPS'])

covid_new.head()
covid_new = covid_new.dropna()
len(covid_new)
covid_new.columns.values.tolist()
covid_data = covid_new.loc[:,"1/22/20":]

covid_data['State'] = covid_new['Province_State']

covid_data['FIPS'] = covid_new['FIPS']

covid_data.head()
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

covid_alabama = covid_data.loc[covid_data["State"]=="Alabama"]
len(covid_alabama)
covid_alabama = covid_alabama.loc[:, (covid_alabama != 0).any(axis=0)]
covid_alabama = covid_alabama.drop(["POPESTIMATE2019", "State","CTYNAME"],axis=1)
covid_alabama.head()
col_alabama_list = covid_alabama.columns.values.tolist()

del col_alabama_list[-1]
covid_alabama_tidy = pd.melt(covid_alabama,id_vars = ['FIPS'], value_vars = col_alabama_list)

covid_alabama_tidy.head()
covid_alabama_tidy.rename(columns = {'variable':'Date','value':'Cases'}, inplace = True)
covid_alabama_tidy = pd.merge(covid_alabama_tidy, census_pop, how="left", on=['FIPS'])
covid_alabama_tidy['Date'] =  pd.to_datetime(covid_alabama_tidy['Date'], format='%m/%d/%y')
covid_alabama_tidy['Date'] = covid_alabama_tidy['Date'].astype(str)
covid_alabama_tidy_sort = covid_alabama_tidy.sort_values(by=["FIPS","Date"])
covid_alabama_tidy_sort.head()
covid_alabama_tidy_sort.rename(columns = {'POPESTIMATE2019':'Population','CTYNAME':'County'}, inplace = True)
covid_alabama_tidy_sort["Cases per 100,000"] = (covid_alabama_tidy_sort["Cases"]/covid_alabama_tidy_sort["Population"])*100000

covid_alabama_tidy_sort
#The state code for Alabama is 01.  Use the appropriate state code for the state you wish to visualize.
res_alabama = [i for i in counties['features'] if not (i['properties']['STATE'] != '01')] 

alabama = {'type': 'FeatureCollection', 'features': res_alabama}
fig = px.choropleth_mapbox(covid_alabama_tidy_sort, geojson=alabama, locations='FIPS', color='Cases', animation_frame="Date",
                           color_continuous_scale="Viridis",
                           range_color=(0, covid_alabama_tidy_sort['Cases'].max()),
                           mapbox_style="carto-positron",
                           center = {'lat':32.318231,'lon':-86.902298},
                           zoom = 5,
                           opacity=0.5,
                           hover_name = 'County',
                           hover_data = ['Population'],
                           title = "COVID-19 Cases in Alabama from the Date of the First Case"
                          )
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
#This speeds up the transition when playing the animation.
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 250
fig.show()
fig = px.choropleth_mapbox(covid_alabama_tidy_sort, geojson=alabama, locations='FIPS', color='Cases per 100,000', animation_frame="Date",
                           color_continuous_scale="Viridis",
                           range_color=(0, covid_alabama_tidy_sort['Cases per 100,000'].max()),
                           mapbox_style="carto-positron",
                           center = {'lat':32.318231,'lon':-86.902298},
                           zoom = 5,
                           opacity=0.5,
                           hover_name = 'County',
                           hover_data = ['Population','Cases'],
                           title = "COVID-19 Cases Per 100,000 People in Alabama from the Date of the First Case"
                          )
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 250
fig.show()
fig = px.line(covid_alabama_tidy_sort,x = "Date", y='Cases per 100,000', line_group='County', color='County')

fig.update_layout(title = "COVID-19 Cases Per 100,000 People in Alabama from the Date of the First Case")
fig.update_xaxes(rangeslider_visible=True)
fig.show()
covid_ny = covid_data.loc[covid_data["State"]=="New York"]

covid_ny = covid_ny.loc[:, (covid_ny != 0).any(axis=0)]

covid_ny = covid_ny.drop(["POPESTIMATE2019", "State","CTYNAME"],axis=1)

col_ny_list = covid_ny.columns.values.tolist()

del col_ny_list[-1]

covid_ny_tidy = pd.melt(covid_ny,id_vars = ['FIPS'], value_vars = col_ny_list)

covid_ny_tidy.rename(columns = {'variable':'Date','value':'Cases'}, inplace = True)

covid_ny_tidy = pd.merge(covid_ny_tidy, census_pop, how="left", on=['FIPS'])

covid_ny_tidy['Date'] =  pd.to_datetime(covid_ny_tidy['Date'], format='%m/%d/%y')
covid_ny_tidy['Date'] = covid_ny_tidy['Date'].astype(str)
covid_ny_tidy_sort = covid_ny_tidy.sort_values(by=["FIPS","Date"])

covid_ny_tidy_sort.rename(columns = {'POPESTIMATE2019':'Population','CTYNAME':'County'}, inplace = True)

covid_ny_tidy_sort["Cases per 100,000"] = (covid_ny_tidy_sort["Cases"]/covid_ny_tidy_sort["Population"])*100000

covid_ny_tidy_sort
res_ny = [i for i in counties['features'] if not (i['properties']['STATE'] != '36')] 

ny = {'type': 'FeatureCollection', 'features': res_ny}
fig = px.choropleth_mapbox(covid_ny_tidy_sort, geojson=ny, locations='FIPS', color='Cases', animation_frame="Date",
                           color_continuous_scale="Viridis",
                           range_color=(0, covid_ny_tidy_sort['Cases'].max()),
                           mapbox_style="carto-positron",
                           center = {'lat':43.299428,'lon':-74.217933},
                           zoom = 5,
                           opacity=0.5,
                           hover_name = 'County',
                           hover_data = ['Population'],
                           title = "COVID-19 Cases in New York from the Date of the First Case"
                          )
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 250
fig.show()
nyc_data = pd.read_csv('https://raw.githubusercontent.com/nychealth/coronavirus-data/master/boro/boroughs-case-hosp-death.csv')
nyc_data.head()
nyc_data_case = nyc_data.drop(['BK_HOSPITALIZED_COUNT','BK_DEATH_COUNT',
                              'BX_HOSPITALIZED_COUNT','BX_DEATH_COUNT',
                              'MN_HOSPITALIZED_COUNT','MN_DEATH_COUNT',
                              'QN_HOSPITALIZED_COUNT','QN_DEATH_COUNT',
                              'SI_HOSPITALIZED_COUNT','SI_DEATH_COUNT'],axis=1)
nyc_data_case.head()
nyc_data_case['bk_cumulative_sum'] = nyc_data_case['BK_CASE_COUNT'].cumsum()
nyc_data_case['bx_cumulative_sum'] = nyc_data_case['BX_CASE_COUNT'].cumsum()
nyc_data_case['mn_cumulative_sum'] = nyc_data_case['MN_CASE_COUNT'].cumsum()
nyc_data_case['qn_cumulative_sum'] = nyc_data_case['QN_CASE_COUNT'].cumsum()
nyc_data_case['si_cumulative_sum'] = nyc_data_case['SI_CASE_COUNT'].cumsum()
nyc_data_case = nyc_data_case.drop(['BK_CASE_COUNT','BX_CASE_COUNT','MN_CASE_COUNT','QN_CASE_COUNT','SI_CASE_COUNT'],axis=1)
nyc_data_case.set_index('DATE_OF_INTEREST',inplace=True)
nyc_data_case_t = nyc_data_case.transpose()
nyc_data_case_t
nyc_data_case_t["FIPS"] = ["36047","36005","36061","36081","36085"]
nyc_data_case_t
covid_ny = covid_data.loc[covid_data["State"]=="New York"]

covid_ny = covid_ny.loc[:,"2/29/20":]
covid_ny = covid_ny.drop(["POPESTIMATE2019", "State","CTYNAME"],axis=1)

#This if statement is when you are pulling data from CSSE that was updated before nychealth
if len(nyc_data_case_t.columns.values) < len(covid_ny.columns.values):
    nyc_data_case_t.insert(len(nyc_data_case_t.columns)-1,"Temp",nyc_data_case_t.iloc[:,-2])

covid_ny.loc[covid_ny["FIPS"]=="36005"] = nyc_data_case_t.loc[nyc_data_case_t["FIPS"]=="36005"].values
covid_ny.loc[covid_ny["FIPS"]=="36047"] = nyc_data_case_t.loc[nyc_data_case_t["FIPS"]=="36047"].values
covid_ny.loc[covid_ny["FIPS"]=="36061"] = nyc_data_case_t.loc[nyc_data_case_t["FIPS"]=="36061"].values
covid_ny.loc[covid_ny["FIPS"]=="36081"] = nyc_data_case_t.loc[nyc_data_case_t["FIPS"]=="36081"].values
covid_ny.loc[covid_ny["FIPS"]=="36085"] = nyc_data_case_t.loc[nyc_data_case_t["FIPS"]=="36085"].values

covid_ny = covid_ny.loc[:, (covid_ny != 0).any(axis=0)]

col_ny_list = covid_ny.columns.values.tolist()

del col_ny_list[-1]

covid_ny_tidy = pd.melt(covid_ny,id_vars = ['FIPS'], value_vars = col_ny_list)

covid_ny_tidy.rename(columns = {'variable':'Date','value':'Cases'}, inplace = True)

covid_ny_tidy = pd.merge(covid_ny_tidy, census_pop, how="left", on=['FIPS'])

covid_ny_tidy['Date'] =  pd.to_datetime(covid_ny_tidy['Date'], format='%m/%d/%y')
covid_ny_tidy['Date'] = covid_ny_tidy['Date'].astype(str)
covid_ny_tidy_sort = covid_ny_tidy.sort_values(by=["FIPS","Date"])

covid_ny_tidy_sort.rename(columns = {'POPESTIMATE2019':'Population','CTYNAME':'County'}, inplace = True)

covid_ny_tidy_sort["Cases per 100,000"] = (covid_ny_tidy_sort["Cases"]/covid_ny_tidy_sort["Population"])*100000

covid_ny_tidy_sort
fig = px.choropleth_mapbox(covid_ny_tidy_sort, geojson=ny, locations='FIPS', color='Cases', animation_frame="Date",
                           color_continuous_scale="Viridis",
                           range_color=(0, covid_ny_tidy_sort['Cases'].max()),
                           mapbox_style="carto-positron",
                           center = {'lat':43.299428,'lon':-74.217933},
                           zoom = 5,
                           opacity=0.5,
                           hover_name = 'County',
                           hover_data = ['Population'],
                           title = "COVID-19 Cases in New York from the Date of the First Case"
                          )
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 250
fig.show()
fig = px.choropleth_mapbox(covid_ny_tidy_sort, geojson=ny, locations='FIPS', color='Cases per 100,000', animation_frame="Date",
                           color_continuous_scale="Viridis",
                           range_color=(0, covid_ny_tidy_sort['Cases per 100,000'].max()),
                           mapbox_style="carto-positron",
                           center = {'lat':43.299428,'lon':-74.217933},
                           zoom = 5,
                           opacity=0.5,
                           hover_name = 'County',
                           hover_data = ['Population','Cases'],
                           title = "COVID-19 Cases Per 100,000 people in New York from the Date of the First Case"
                          )
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 250
fig.show()
fig = px.choropleth_mapbox(covid_ny_tidy_sort, geojson=ny, locations='FIPS', color='Cases per 100,000', animation_frame="Date",
                           color_continuous_scale="Viridis",
                           #range_color=(0, covid_ny_tidy_sort['Cases per 100,000'].max()),
                           mapbox_style="carto-positron",
                           center = {'lat':43.299428,'lon':-74.217933},
                           zoom = 5,
                           opacity=0.5,
                           hover_name = 'County',
                           hover_data = ['Population','Cases'],
                           title = "COVID-19 Cases Per 100,000 people in New York from the Date of the First Case"
                          )
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 250
fig.show()
fig = px.line(covid_ny_tidy_sort,x = "Date", y='Cases per 100,000', line_group='County', color='County')

fig.update_layout(title = "COVID-19 Cases Per 100,000 People in New York from the Date of the First Case")
fig.update_xaxes(rangeslider_visible=True)
fig.show()
covid_fl = covid_data.loc[covid_data["State"]=="Florida"]

covid_fl = covid_fl.loc[:, (covid_fl != 0).any(axis=0)]

covid_fl = covid_fl.drop(["POPESTIMATE2019", "State","CTYNAME"],axis=1)

col_fl_list = covid_fl.columns.values.tolist()

del col_fl_list[-1]

covid_fl_tidy = pd.melt(covid_fl,id_vars = ['FIPS'], value_vars = col_fl_list)

covid_fl_tidy.rename(columns = {'variable':'Date','value':'Cases'}, inplace = True)

covid_fl_tidy = pd.merge(covid_fl_tidy, census_pop, how="left", on=['FIPS'])

covid_fl_tidy['Date'] =  pd.to_datetime(covid_fl_tidy['Date'], format='%m/%d/%y')
covid_fl_tidy['Date'] = covid_fl_tidy['Date'].astype(str)
covid_fl_tidy_sort = covid_fl_tidy.sort_values(by=["FIPS","Date"])

covid_fl_tidy_sort.rename(columns = {'POPESTIMATE2019':'Population','CTYNAME':'County'}, inplace = True)

covid_fl_tidy_sort["Cases per 100,000"] = (covid_fl_tidy_sort["Cases"]/covid_fl_tidy_sort["Population"])*100000

res_fl = [i for i in counties['features'] if not (i['properties']['STATE'] != '12')] 

fl = {'type': 'FeatureCollection', 'features': res_fl}
fig = px.choropleth_mapbox(covid_fl_tidy_sort, geojson=fl, locations='FIPS', color='Cases per 100,000', animation_frame="Date",
                           color_continuous_scale="Viridis",
                           range_color=(0, covid_fl_tidy_sort['Cases per 100,000'].max()),
                           mapbox_style="carto-positron",
                           center = {'lat':27.664827,'lon':-81.515754},
                           zoom = 5,
                           opacity=0.5,
                           hover_name = 'County',
                           hover_data = ['Population'],
                           title = "COVID-19 Cases Per 100,000 People in Florida from the Date of the First Case"
                          )
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 250
fig.show()
fig = px.line(covid_fl_tidy_sort,x = "Date", y='Cases per 100,000', line_group='County', color='County')

fig.update_layout(title = "COVID-19 Cases Per 100,000 People in Florida from the Date of the First Case")
fig.update_xaxes(rangeslider_visible=True)
fig.show()
deaths = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv")
deaths.head()
deaths = deaths.dropna()
deaths["FIPS"] = deaths["FIPS"].astype(int)
deaths["FIPS"] = deaths["FIPS"].astype(str)

for i in range(len(deaths)):
    if len(deaths["FIPS"].iloc[i]) < 5:
        deaths["FIPS"].iloc[i] = "0"+deaths["FIPS"].iloc[i]

deaths_new = pd.merge(deaths, census_pop, how="left", on=['FIPS'])

deaths_new = deaths_new.dropna()
deaths_data = deaths_new.loc[:,"1/22/20":]

deaths_data['State'] = deaths_new['Province_State']

deaths_data['FIPS'] = deaths_new['FIPS']

deaths_data.head()
deaths_fl = deaths_data.loc[deaths_data["State"]=="Florida"]

deaths_fl = deaths_fl.loc[:, (deaths_fl != 0).any(axis=0)]

deaths_fl = deaths_fl.drop(["POPESTIMATE2019", "State","CTYNAME"],axis=1)

col_deaths_fl_list = deaths_fl.columns.values.tolist()

del col_deaths_fl_list[-1]

deaths_fl_tidy = pd.melt(deaths_fl,id_vars = ['FIPS'], value_vars = col_deaths_fl_list)

deaths_fl_tidy.rename(columns = {'variable':'Date','value':'Deaths'}, inplace = True)

deaths_fl_tidy = pd.merge(deaths_fl_tidy, census_pop, how="left", on=['FIPS'])

deaths_fl_tidy['Date'] =  pd.to_datetime(deaths_fl_tidy['Date'], format='%m/%d/%y')
deaths_fl_tidy['Date'] = deaths_fl_tidy['Date'].astype(str)
deaths_fl_tidy_sort = deaths_fl_tidy.sort_values(by=["FIPS","Date"])

deaths_fl_tidy_sort.rename(columns = {'POPESTIMATE2019':'Population','CTYNAME':'County'}, inplace = True)

deaths_fl_tidy_sort["Deaths per 100,000"] = (deaths_fl_tidy_sort["Deaths"]/deaths_fl_tidy_sort["Population"])*100000

deaths_fl_tidy_sort

res_fl = [i for i in counties['features'] if not (i['properties']['STATE'] != '12')] 

fl = {'type': 'FeatureCollection', 'features': res_fl}
fig = px.choropleth_mapbox(deaths_fl_tidy_sort, geojson=fl, locations='FIPS', color='Deaths per 100,000', animation_frame="Date",
                           color_continuous_scale="Viridis",
                           range_color=(0, deaths_fl_tidy_sort['Deaths per 100,000'].max()),
                           mapbox_style="carto-positron",
                           center = {'lat':27.664827,'lon':-81.515754},
                           zoom = 5,
                           opacity=0.5,
                           hover_name = 'County',
                           hover_data = ['Population'],
                           title = "COVID-19 Deaths Per 100,000 People in Florida from the Date of the First Case"
                          )
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 250
fig.show()
fig = px.line(deaths_fl_tidy_sort,x = "Date", y='Deaths per 100,000', line_group='County', color='County')

fig.update_layout(title = "COVID-19 Deaths Per 100,000 People in Florida from the Date of the First Case")
fig.update_xaxes(rangeslider_visible=True)
fig.show()
deaths_fl_tidy_sort.head()
all_data_fl = pd.merge(covid_fl_tidy_sort, deaths_fl_tidy_sort, how="left", on=['FIPS','Date'])
all_data_fl = all_data_fl.dropna()
all_data_fl = all_data_fl.drop(['Population_y','County_y'],axis=1)
all_data_fl.rename(columns = {'Population_x':'Population','County_x':'County'}, inplace = True)

all_data_fl.head()
all_data_fl['Deaths per Case'] = all_data_fl['Deaths']/all_data_fl['Cases']

all_data_fl['Deaths per Case'] = all_data_fl['Deaths per Case'].fillna(0)
all_data_fl['Deaths per Case'] = all_data_fl['Deaths per Case'].replace([np.inf],0)
all_data_fl.head()
fig = px.choropleth_mapbox(all_data_fl, geojson=fl, locations='FIPS', color='Deaths per Case', animation_frame="Date",
                           color_continuous_scale="Viridis",
                           #range_color=(0, all_data_fl['Deaths per Case'].max()),
                           mapbox_style="carto-positron",
                           center = {'lat':27.664827,'lon':-81.515754},
                           zoom = 5,
                           opacity=0.5,
                           hover_name = 'County',
                           hover_data = ['Population','Cases','Deaths'],
                           title = "COVID-19 Deaths Per Confirmed Case in Florida from the Date of the First Case"
                          )
fig.update_layout(margin={"r":20,"t":30,"l":20,"b":20})
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 250
fig.show()
#fig.write_html("florida_deaths_per_case_choropleth.html")
fig = px.line(all_data_fl,x = "Date", y='Deaths per Case', line_group='County', color='County')

fig.update_layout(title = "COVID-19 Deaths Per Confirmed Case in Florida from the Date of the First Case")
fig.update_xaxes(rangeslider_visible=True)
fig.show()
all_data_fl["Percent Change Cases"] = all_data_fl["Cases"].pct_change()
all_data_fl['Percent Change Cases'] = all_data_fl['Percent Change Cases'].fillna(0)
all_data_fl['Percent Change Cases'] = all_data_fl['Percent Change Cases'].replace([np.inf,-np.inf],0)

all_data_fl.head()
all_data_fl.loc[all_data_fl['Date'] == '2020-03-08', 'Percent Change Cases'] = 0
all_data_fl.loc[all_data_fl['Date'] == '2020-03-08']
fig = px.line(all_data_fl,x = "Date", y='Percent Change Cases', line_group='County', color='County')

fig.update_layout(title = "COVID-19 Percent Change in Cases in Florida from the Date of the First Case")
fig.update_xaxes(rangeslider_visible=True)
fig.show()
all_data_fl["Percent Change Deaths"] = all_data_fl["Deaths"].pct_change()
all_data_fl['Percent Change Deaths'] = all_data_fl['Percent Change Deaths'].fillna(0)
all_data_fl['Percent Change Deaths'] = all_data_fl['Percent Change Deaths'].replace([np.inf,-np.inf],0)

all_data_fl.loc[all_data_fl['Date'] == '2020-03-08', 'Percent Change Cases'] = 0
all_data_fl.sample(5)
fig = px.choropleth_mapbox(all_data_fl, geojson=fl, locations='FIPS', color='Percent Change Deaths', animation_frame="Date",
                           color_continuous_scale="Viridis",
                           #range_color=(all_data_fl['Percent Change Deaths'].min(), all_data_fl['Percent Change Deaths'].max()),
                           mapbox_style="carto-positron",
                           center = {'lat':27.664827,'lon':-81.515754},
                           zoom = 5,
                           opacity=0.5,
                           hover_name = 'County',
                           hover_data = ['Population','Cases','Deaths'],
                           title = "COVID-19 Percent Change in Deaths in Florida from the Date of the First Case"
                          )
fig.update_layout(margin={"r":20,"t":30,"l":20,"b":20})
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 250
fig.show()
covid_data["Cases per 100,000 for 7/24/2020"] = covid_data["7/24/20"]/covid_data["POPESTIMATE2019"]


#The New York City Data is not fixed in this and the data is just removed
fig = px.choropleth_mapbox(covid_data, geojson=counties, locations='FIPS', color='Cases per 100,000 for 7/24/2020',
                           color_continuous_scale="Viridis",
                           mapbox_style="carto-positron",
                           zoom=3, center = {"lat": 37.0902, "lon": -95.7129},
                           opacity=0.5,
                           labels={'Cases per 100,000 for 7/24/2020':'Cases per 100,000'},
                           title = "COVID-19 Cases Per 100,000 People on 7/24/2020",
                           hover_name = "CTYNAME",
                           hover_data = ["State","7/24/20"]
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
