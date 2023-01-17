import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# curDir = os.getcwd()

# print(curDir)
# # IF KAGGLE

kaggle_path = '../input/covid19pluspopulations/'

# ELSE IF LOCALHOST

# kaggle_path = 'input/covid19pluspopulations/'
import pandas as pd

# manual rework OR download the full result set

# dfi = pd.read_csv(kaggle_path+"covid19cases-covid-19-case-counts-QueryResult.csv", sep=',')

# dfi = pd.read_csv(kaggle_path+"covid19cases-covid-19-case-counts-QueryResult.csv", sep=',')

dfi = pd.read_csv('https://query.data.world/s/xhgtdkxwe4c4ys4bwejjycwbwrnnkk')

dfi.columns = map(str.lower, dfi.columns)



dfi['Confirmed_sum'] = 0

dfi['Deaths_sum'] = 0

dfi.sample(3)
# set last update

DDD = dfi.date.max()

DDD = DDD.replace("/", "-")

print(DDD)
dfi['cases'].fillna(0, inplace=True)

dfi['difference'].fillna(0, inplace=True)

dfi
def get_confirmed(row):

    if row['case_type'] == 'Confirmed':

        return row['cases']

    else:

        return 0

    

def get_deaths(row):

    if row['case_type'] == 'Deaths':

        return row['cases']

    else:

        return 0



dfi['Confirmed_max'] = dfi.apply(get_confirmed, axis=1)

dfi['Deaths_max'] = dfi.apply(get_deaths, axis=1)

dfi
# select US only

dfi = dfi[dfi.country_region == 'US']

dfi
dfig = dfi.groupby(['province_state'])['Confirmed_max', 'Deaths_max'].max().reset_index()

dfig
# combined steps of 'prep wiki pop' section later on



# get population data

dfp = pd.read_csv(kaggle_path+"us_states_population.csv", sep=';')



# work cols

# dfp.rename(columns={ dfp.columns[3]: "population" }, inplace = True)

# dfp = dfp.rename(columns={'Country_cln': 'country_region'})

# dfp.drop(dfp.columns[[0, 1, 4, 5, 6]], axis = 1, inplace = True)



# merge

dfip = dfig.merge(dfp, how="left", on='province_state')



# add cols

dfip['Confirmed_max_%'] = (100.0 * dfip['Confirmed_max'])/dfip['population']

dfip['Deaths_max_%'] = (100.0 * dfip['Deaths_max'])/dfip['population']



# rename max cols

dfip = dfip.rename(columns={'Confirmed_max': 'Confirmed'})

dfip = dfip.rename(columns={'Deaths_max': 'Deaths'})

dfip = dfip.rename(columns={'Confirmed_max_%': 'Confirmed_%'})

dfip = dfip.rename(columns={'Deaths_max_%': 'Deaths_%'})

dfip.head(10)



# ignore lat lon previous version
tmp = dfip[dfip.province_state.isin(['New York'])]

tmp
dfi = dfip.copy()
dfi['population'] = dfi['population'].astype('float64')

dfi['Confirmed_%'] = dfi['Confirmed_%'].astype('float64')

dfi['Deaths_%'] = dfi['Deaths_%'].astype('float64')
dfi
# drop rows with NaN

dfi.dropna()
import numpy as np

import math

# dfi['pop_sq_rt'] = math.sqrt(dfi['population'])

dfi['pop_sq_rt'] = dfi['population'].apply(np.sqrt)

dfi
dfi_Ap_25 = dfi.sort_values(by=['Confirmed'], ascending=False).head(25)

dfi_Ap_25
dfi_Dp_25 = dfi.sort_values(by=['Deaths'], ascending=False).head(25)

dfi_Dp_25
import seaborn as sns; sns.set()

import matplotlib.pyplot as plt
sns_plot = sns.pairplot(dfi, height=2.5)

# fig = sns_plot.get_figure()
sns_plot.savefig("IMG_US_dfi_Pairplot_"+DDD+".png")
plt.figure(figsize=(16, 6))

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

ax = sns.scatterplot(x="Confirmed", y="Confirmed_%", 

                     size="population", 

                     sizes=(20, 200), 

                     alpha = 0.8,

                     hue="population", hue_norm=(1, 100), palette=cmap,

#                      legend="full", 

                     data=dfi)



plt.figure(figsize=(10,10))
import matplotlib

import matplotlib.pyplot as plt

# ax.figure.savefig("IMG_Active_vs_Active_%_20200321_22h54_NL.png")

ax.figure.savefig('IMG_Confirmed_vs_Confirmed_%_'+DDD+'.pdf')

ax.figure.savefig('IMG_Active_vs_Active_%__'+DDD+'.png')
import plotly.express as px



fig = px.scatter(dfi_Ap_25, x="Confirmed", y="Confirmed_%", log_x=False, log_y=False,

                size="pop_sq_rt", size_max=40,

                color="Confirmed_%",

                opacity = 0.6, 

#                      legend="full",  

                 hover_name="province_state", 

                 hover_data=["population"])



fig.show()
import plotly.express as px



fig = px.scatter(dfi_Dp_25, x="Deaths", y="Deaths_%", log_x=False, log_y=True,

                size="pop_sq_rt", size_max=40,

                color="Deaths_%",

                opacity = 0.6, 

#                      legend="full",  

                 hover_name="province_state", 

                 hover_data=["population"])



fig.show()
import plotly.express as px



fig = px.scatter(dfi_Dp_25, x="Confirmed_%", y="Deaths_%", log_x=False, log_y=True,

                size="pop_sq_rt", size_max=40,

                color="Deaths_%",

                opacity = 0.6, 

#                      legend="full",  

                 hover_name="province_state", 

                 hover_data=["population"])



fig.show()
import pandas as pd

# df = pd.read_csv(kaggle_path+"covid19cases-covid-19-case-counts-QueryResult.csv", sep=',')

df = pd.read_csv('https://query.data.world/s/xhgtdkxwe4c4ys4bwejjycwbwrnnkk')

df.columns = map(str.lower, df.columns)



# @covid-19-data-resource-hub



#     County-level data has been added for the US and is available from Mar 23.

#     Recoveries are no longer provided due to a lack of confidence in the data. Additionally, active cases cannot be calculated without recoveries.

#     The difference field has been nulled out due to a change in the level of detail provided. We are working through this issue.



df.head()
df.shape
import pandas as pd

dfp = pd.read_csv(kaggle_path+"us_states_population.csv", sep=';')



dfp.sample(3)
tmp = dfp[dfp['province_state'].isin(["New York"])]

tmp
dfp.dtypes
dfp.index
# dfp = dfp.rename(columns={'Population': 'population'})

# dfp.rename(columns={ dfp.columns[3]: "population" }, inplace = True)



# dfp = dfp.rename(columns={'Country_cln': 'country_region'})

# dfp = dfp.rename(columns={'% of world population': 'cr_pop_as %_of_world_pop'})



# dfp.drop(dfp.columns[[0, 1, 5, 6]], axis = 1, inplace = True)



dfp
dfp.dtypes
dfip = df.merge(dfp, how="left", on='province_state')

dfip.head()
dfip = dfip[dfip.country_region.isin(['US'])]

dfip
dfip['difference'].fillna(0, inplace=True)

dfip
dfip.drop(dfip.columns[[6,7,8,9,10,11,12]], axis = 1, inplace = True)

dfip
tmp = dfip[dfip['province_state'].isin(['New York'])]

tmp
# tmp = dfip[~dfip['cases'].isin([0])]

# tmp
# dfp['%_of_world_pop'] = dfp['%_of_world_pop'].map(lambda x: x.rstrip('%'))

dfip['cases_as_%_of_pop'] = (100.0 * dfip['cases'])/dfip['population']



# dfip[dfip['cases_as_%_of_pop'].notnull() & (df['case_type'] == "Active")]

dfip[(dfip['cases_as_%_of_pop'] >= 0.001) & (df['case_type'] == "Confirmed")]
# https://plot.ly/python/v3/gapminder-example/

# copy an convert to javascript
# https://medium.com/plotly/introducing-plotly-express-808df010143d

# https://plotly.com/python-api-reference/plotly.express.html
# dfip = df.merge(dfp, how="left", on='country_region')

dfip.sample(3)
dfip[dfip.p_s.isin(['NY'])]
# dfip_g = dfip.drop(['table_names', 'prep_flow_runtime', 'admin2', 'combined_key', 'fips', 'lat', 'location', 'long', 'prep_flow_runtime'], axis=1)

# dfip_g = dfip.copy()
# import pandas as pd

# dfc = pd.read_csv(kaggle_path+"ggl_latlon.csv", sep=';')

# dfc = dfc.rename(columns={'name_corr': 'country_region'})



# dfc.head()
# dfipc = dfip_g.merge(dfc, how="left", on='country_region')

# # dfipc = dfipc.drop(['lat', 'long', 'name'], axis=1)

# dfipc = dfipc.drop(['name'], axis=1)

# dfipc = dfipc.rename(columns={'latitude': 'lat'})

# dfipc = dfipc.rename(columns={'longitude': 'lon'})

# dfipc.tail(5)
# # tmp = dfipc[dfipc['province_state'].isin(['Netherlands'])]

# tmp = dfipc[dfipc['province_state'].isin(['N/A'])]

# tmp.sort_values(by=['date']).tail(3)
# import pandas as pd

# dfw = pd.read_csv(kaggle_path+"country-and-continent-codes-list.csv", sep=',')



# dfw = dfw.drop(['Continent_Code', 'Country_Name', 'Three_Letter_Country_Code', 'Country_Number'], axis=1)

# dfw = dfw.rename(columns={'Two_Letter_Country_Code': 'country'})

# # df = pd.read_csv()

# dfw.head()
# dfipc = dfipc.merge(dfw, how="left", on='country')

# dfipc = dfipc.rename(columns={'Continent_Name': 'continent'})

# dfipc.sample(3)
dfipc = dfip.copy()
dfipc_ct_a = dfipc.copy()

# sum Active of all province_states for each country_region

dfipc_ct_a = dfipc_ct_a[dfipc_ct_a['case_type'].isin(['Confirmed'])].groupby(['date','province_state'], as_index=False)[['cases']].agg('sum')

dfipc_ct_a = dfipc_ct_a.rename(columns={'cases': 'sum_cr_confirmed'})



dfipc_ct_d = dfipc.copy()

# sum Deaths of all privonce_states for each country_region

dfipc_ct_d = dfipc_ct_d[dfipc_ct_d['case_type'].isin(['Deaths'])].groupby(['date','province_state'], as_index=False)[['cases']].agg('sum')

dfipc_ct_d = dfipc_ct_d.rename(columns={'cases': 'sum_cr_deaths'})



dfipc_ct_d.tail(3)
tmp = dfipc_ct_a[dfipc_ct_a.province_state.isin(['New York'])]

tmp = tmp.sort_values(by=['date'], ascending=False)

tmp
# dfip_g.columns = list(map(''.join, dfip_g.columns.values))

# dfip_g

# tmp = dfipc_ct_a[dfipc_ct_a['province_state'].isin(['US'])]

# tmp.tail(3)
# dfip_g.columns = list(map(''.join, dfip_g.columns.values))

# dfip_g

tmp = dfipc_ct_d[dfipc_ct_d['province_state'].isin(['New York'])]

tmp.tail(3)
# pd.merge(a, b, on=['A', 'B'])

# dfip = df.merge(dfp, how="left", on='country_region')

dfg = dfipc.merge(dfipc_ct_a, how="left", on=['date', 'province_state'])

dfg = dfg.merge(dfipc_ct_d, how="left", on=['date', 'province_state'])

dfg
# tmp = dfg[dfg['country_region'].isin(['US'])]

# tmp.sort_values(by=['date']).tail(3)
tmp = dfg[dfg['province_state'].isin(['New York'])]

tmp.sort_values(by=['date']).tail(10)
dfg['cases'] = dfg['cases'].fillna(0)

dfg['difference'] = dfg['difference'].fillna(0)
def get_cases_confirmed(row):

    if row['case_type'] == 'Confirmed':

        return row['cases']

    else:

        return 0

    

def get_cases_deaths(row):

    if row['case_type'] == 'Deaths':

        return row['cases']

    else:

        return 0



def get_difference_confirmed(row):

    if row['case_type'] == 'Confirmed':

        return row['difference']

    else:

        return 0

    

def get_difference_deaths(row):

    if row['case_type'] == 'Deaths':

        return row['difference']

    else:

        return 0

    

dfg['cases_confirmed'] = dfg.apply(get_cases_confirmed, axis=1)

dfg['cases_deaths'] = dfg.apply(get_cases_deaths, axis=1)

dfg['difference_confirmed'] = dfg.apply(get_difference_confirmed, axis=1)

dfg['difference_deaths'] = dfg.apply(get_difference_deaths, axis=1)

dfg.sample(3)
# dfg = dfg.drop(['province_state', 'case_type', 'cases', 'difference'], axis=1)

dfg = dfg.drop(['country_region'], axis=1)

dfg.tail(3)
# dfg = dfg.drop_duplicates(subset=['date', 'province_state'], keep='first')

dfg = dfg.sort_values(by=['province_state','date'], ascending=False)

# dfg.sample(10)

dfg
tmp = dfg[dfg['province_state'].isin(['New York'])]

tmp.sort_values(by=['date']).tail(3)
# import numpy as np

# # dfg['Continent_Name'] = dfg['Continent_Name'].replace(np.nan, 'X', regex=True)

# dfg['continent'] = dfg['continent'].fillna('X')



# list_CN = dfg.continent.unique()

# print(list_CN)
import numpy as np

dfg['population'] = dfg['population'].fillna(0)
dfg.population = dfg.population.astype(int)

# dfg.date = dfg.date.astype('datetime64')
dfg.dtypes
# dfg = dfg.drop(['lat','long'], axis=1)
# # insightful

# sns_plot = sns.pairplot(dfg, height=2.5)
# sns_plot.savefig("IMG_US_dfg__Pairplot_'+DDD'.png")
dfg.tail(3)
dfg = dfg.sort_values(by=['date'], ascending=True)
# dfp['%_of_world_pop'] = dfp['%_of_world_pop'].map(lambda x: x.rstrip('%'))

dfg['sum_cr_confirmed_as_%_of_pop'] = (100.0 * dfg['sum_cr_confirmed'])/dfg['population']

# dfp['%_of_world_pop'] = dfp['%_of_world_pop'].map(lambda x: x.rstrip('%'))

dfg['sum_cr_deaths_as_%_of_pop'] = (100.0 * dfg['sum_cr_deaths'])/dfg['population']
# adapted from: https://towardsdatascience.com/coronavirus-data-visualizations-using-plotly-cfbdb8fcfc3d

# locations=["CA", "TX", "NY"], locationmode="USA-states", color=[1,2,3], scope="usa"



fig = px.choropleth(dfg, 

#                     locations="country_region", 

#                     locationmode = "country names",

#                     color="cases", 

#                     hover_name="country_region", 

#                     animation_frame="date"

                    locations="p_s", 

                    locationmode = "USA-states",

                    scope="usa",

                    color="sum_cr_confirmed", 

                    hover_name="province_state", 

                    animation_frame="date"

                   )

                

fig.update_layout(

    title_text = 'Spread of Coronavirus, Confirmed cases per State',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

    

fig.show()
dfg.sample(3)
dfg.dtypes
dfip_a_25 = dfg.copy()



# find max date

max_date = dfip_a_25.date.max()



# list countries within t25 on date_max

tmp = dfip_a_25[dfip_a_25['date']==max_date]

# sum percetages of all province_states for each country_region

tmp = tmp.groupby(['province_state'])[['cases_confirmed']].agg('sum')



# sort & get top 25

tmp = tmp.sort_values(by=['cases_confirmed'], ascending=False).head(25)

# save as list for filtering

list_cr_a_t25 = tmp.index.values.tolist()



# select rows of top25 countries

dfip_a_25 = dfip_a_25[dfip_a_25['province_state'].isin(list_cr_a_t25)]

dfip_a_25
# adapted from: https://towardsdatascience.com/coronavirus-data-visualizations-using-plotly-cfbdb8fcfc3d

fig = px.choropleth(dfip_a_25, 

#                     locations="country_region", 

#                     locationmode = "country names",

#                     color="cases", 

#                     hover_name="country_region", 

#                     animation_frame="date"

                    locations="p_s", 

                    locationmode = "USA-states",

                    scope="usa",

                    color="sum_cr_confirmed", 

                    hover_name="province_state", 

                    animation_frame="date"

                   )



fig.update_layout(

    title_text = 'Spread of Coronavirus, Confirmed cases of current Top 25 States',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

    

fig.show()
dfip_ap = dfg.copy()

dfip_ap = dfip_ap[dfip_ap['case_type'].isin(["Confirmed"])]

dfip_ap = dfip_ap.sort_values(by=['date'])

tmp = dfip_ap[~dfip_ap['cases_as_%_of_pop'].isin([0.000])]

tmp
# dfap.to_csv(r'covid19casesall-covid-19-case-counts-QueryResult_int_ap.csv', index = False)
# adapted from: https://towardsdatascience.com/coronavirus-data-visualizations-using-plotly-cfbdb8fcfc3d

fig = px.choropleth(dfip_ap, 

#                     locations="country_region", 

#                     locationmode = "country names",

#                     color="cases_as_%_of_pop", 

#                     hover_name="country_region", 

#                     animation_frame="date"

                    locations="p_s", 

                    locationmode = "USA-states",

                    scope="usa",

                    color="sum_cr_confirmed_as_%_of_pop", 

                    hover_name="province_state", 

                    animation_frame="date"

                   )



fig.update_layout(

    title_text = 'Spread of Coronavirus, Confirmed cases as a percentage of the State Population',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

    

fig.show()
# adapted from: https://towardsdatascience.com/coronavirus-data-visualizations-using-plotly-cfbdb8fcfc3d

# locations=["CA", "TX", "NY"], locationmode="USA-states", color=[1,2,3], scope="usa"



fig = px.choropleth(dfg, 

#                     locations="country_region", 

#                     locationmode = "country names",

#                     color="cases", 

#                     hover_name="country_region", 

#                     animation_frame="date"

                    locations="p_s", 

                    locationmode = "USA-states",

                    scope="usa",

                    color="sum_cr_deaths", 

                    hover_name="province_state", 

                    animation_frame="date"

                   )

                

fig.update_layout(

    title_text = 'Spread of Coronavirus, Deaths cases per State',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

    

fig.show()
dfip_d_25 = dfg.copy()



# find max date

max_date = dfip_d_25.date.max()



# list countries within t25 on date_max

tmp = dfip_d_25[dfip_d_25['date']==max_date]

# sum percetages of all privonce_states for each country_region

tmp = tmp.groupby(['province_state'])[['cases_deaths']].agg('sum')

# sort & get top 25

tmp = tmp.sort_values(by=['cases_deaths'], ascending=False).head(25)

# save as list for filtering

list_cr_d_t25 = tmp.index.values.tolist()



# select rows of top25 countries

dfip_d_25 = dfip_d_25[dfip_d_25['province_state'].isin(list_cr_d_t25)]

dfip_d_25
# adapted from: https://towardsdatascience.com/coronavirus-data-visualizations-using-plotly-cfbdb8fcfc3d

fig = px.choropleth(dfip_a_25, 

#                     locations="country_region", 

#                     locationmode = "country names",

#                     color="cases", 

#                     hover_name="country_region", 

#                     animation_frame="date"

                    locations="p_s", 

                    locationmode = "USA-states",

                    scope="usa",

                    color="sum_cr_confirmed", 

                    hover_name="province_state", 

                    animation_frame="date"

                   )



fig.update_layout(

    title_text = 'Spread of Coronavirus, Deaths cases of current Top 25 States',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

    

fig.show()
dfip_dp = dfg.copy()

dfip_dp = dfip_dp[dfip_dp['case_type'].isin(["Deaths"])]

dfip_dp = dfip_dp.sort_values(by=['date'])

tmp = dfip_dp[~dfip_dp['cases_as_%_of_pop'].isin([0.000])]

tmp
# dfap.to_csv(r'covid19casesall-covid-19-case-counts-QueryResult_int_ap.csv', index = False)
# adapted from: https://towardsdatascience.com/coronavirus-data-visualizations-using-plotly-cfbdb8fcfc3d

fig = px.choropleth(dfip_dp, 

#                     locations="country_region", 

#                     locationmode = "country names",

#                     color="cases_as_%_of_pop", 

#                     hover_name="country_region", 

#                     animation_frame="date"

                    locations="p_s", 

                    locationmode = "USA-states",

                    scope="usa",

                    color="sum_cr_deaths_as_%_of_pop", 

                    hover_name="province_state", 

                    animation_frame="date"

                   )



fig.update_layout(

    title_text = 'Spread of Coronavirus, Confirmed cases as a percentage of the State Population',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

    

fig.show()
import plotly.express as px

import numpy as np



# find max

max_s_s_c = dfip_d_25.sum_cr_confirmed.max()



fig = px.treemap(dfip_a_25, path=['province_state'], values='population',

                  color='sum_cr_confirmed', hover_data=['province_state'],

#                   color_continuous_scale='balance',

                  color_continuous_midpoint=np.average(dfi_Ap_25['Confirmed'], weights=dfi_Ap_25['population']),

                range_color=[1,max_s_s_c]

                )



fig.update_layout(

    title_text = 'Spread of Coronavirus, Confirmed Cases in Top 25 States',

    title_x = 0.5)



fig.show()
import plotly.express as px

import numpy as np



# find max

max_c_p = dfi_Ap_25['Confirmed_%'].max()



fig = px.treemap(dfi_Ap_25, path=['province_state'], values='population',

                  color='Confirmed_%', hover_data=['province_state'],

#                   color_continuous_scale='balance',

                  color_continuous_midpoint=np.average(dfi_Ap_25['Confirmed_%'], weights=dfi_Ap_25['population'])

                ,range_color = [0, max_c_p])



fig.update_layout(

    title_text = 'Spread of Coronavirus, Confirmed Cases as a percentage of population of Top 25 States',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))



fig.show()
import plotly.express as px

import numpy as np



# find max

max_d = dfip_d_25.sum_cr_deaths.max()



fig = px.treemap(dfi_Dp_25, path=['province_state'], values='population',

                  color='Deaths', hover_data=['province_state'],

#                   color_continuous_scale='balance',

                 range_color=[0.0, max_d],

                  color_continuous_midpoint=np.average(dfi_Dp_25['Deaths'], weights=dfi_Dp_25['population']))



fig.update_layout(

    title_text = 'Spread of Coronavirus, Deaths Cases in Top 25 States',

    title_x = 0.5)



fig.show()
import plotly.express as px

import numpy as np



# find max

max_d_p = dfi_Ap_25['Deaths_%'].max()



fig = px.treemap(dfi_Dp_25, path=['province_state'], values='population',

                  color='Deaths_%', hover_data=['province_state'],

#                   color_continuous_scale='balance',

                 range_color=[0.0, max_d_p],

                  color_continuous_midpoint=np.average(dfi_Dp_25['Deaths_%'], weights=dfi_Dp_25['population']))



fig.update_layout(

    title_text = 'Spread of Coronavirus, Deaths Cases as a percentage of population of Top 25 States',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))



fig.show()
# Get minimum value of a single column 'y'

print(dfi_Dp_25['Deaths_%'].min())

print(dfi_Dp_25['Deaths_%'].max())

# print(dfi_Dp_25['Deaths_%'])
dfgc = dfg.copy()
dfgc = dfgc.sort_values(by=['date'], ascending=True)

dfgc.sample(3)
dfgc_US = dfgc



# # SELECT T25 States (deaths per 2020 04 01)

dfgc_US = dfgc[dfgc.p_s.isin(['NY','NJ','WA','MI','LA','IL','CA','CT','GA','NV','IN','MA','VA','AZ','TX','FL','CO','WI','MD','TN','PA','VT','PA','OH','RI'])]



# ADD SQRT POPULATION SIZE (FOR BETTER VISUALIZATION)

dfgc_US['pop_sq_rt'] = dfgc_US['population'].apply(np.sqrt)



# ADD CASES/DEATHS PER 100.000 INHABITANTS

dfgc_US['confirmed_per_100.000'] = (100000 * dfgc_US['sum_cr_confirmed'])/dfgc_US['population']

dfgc_US['deaths_per_100.000'] = (100000 * dfgc_US['sum_cr_deaths'])/dfgc_US['population']



# SET READABLE TITLES & LABELS

# dfgc_US = dfgc_US.rename(columns={'confirmed_per_100.000': 'aantal besmet per 100.000'})

# dfgc_US = dfgc_US.rename(columns={'deaths_per_100.000': 'aantal overleden per 100.000'})

dfgc_US = dfgc_US.sort_values(by=['date'], ascending=True)



dfgc_US.tail(6)
is_NaN = dfgc_US.isnull()

row_has_NaN = is_NaN.any(axis=1)

rows_with_NaN = dfgc_US[row_has_NaN]

print(rows_with_NaN)
rows_with_NaN.to_csv(r'US_debug_NaN.csv', index = False)
# dropping Virgin Islands for now (something wrong: empty p_s)

dfgc_US = dfgc_US[dfgc_US.province_state != 'Virgin Islands']
rows_with_NaN.to_csv(r'US_debug_NaN2.csv', index = False)
#  adapted from: https://nbviewer.jupyter.org/github/plotly/plotly_express/blob/gh-pages/walkthrough.ipynb

import plotly.express as px



g = px.scatter(dfgc_US, 

               x="confirmed_per_100.000", y="deaths_per_100.000",

               title='<b>Developments COVID19 US</b> per '+DDD,

               color="p_s", size="pop_sq_rt", 

               hover_name="province_state", hover_data=["population"],

#                facet_col="continent",

               log_x=True, log_y=True,

               width=900, height=500,

               template='plotly_dark',

          )



# g.add_trace(

#     go.Scatter(

#         x=[0, 100],

#         y=[0, 100],

#         mode="lines",

#         line=go.scatter.Line(color="gray", dash="dash"),

#         showlegend=False)

# )



g.show()
#https://towardsdatascience.com/interactive-controls-for-jupyter-notebooks-f5c94829aee6

import ipywidgets as widgets

from ipywidgets import interact, interact_manual
# https://ipywidgets.readthedocs.io/en/stable/examples/Using%20Interact.html

# https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html



@interact

def show_articles_more_than(column=['sum_cr_confirmed', 'sum_cr_deaths'], 

                            x=(0, 100000, 10)): #the format is (start, stop, step)

    return dfgc.loc[dfgc[column] > x].sample(3)
# dfgc.dtypes
# ADD CASES/DEATHS PER 100.000 INHABITANTS

dfgc_US['confirmed_per_100.000'] = (100000 * dfgc_US['sum_cr_confirmed'])/dfgc_US['population']

dfgc_US['deaths_per_100.000'] = (100000 * dfgc_US['sum_cr_deaths'])/dfgc_US['population']
dfgc_US = dfgc_US.sort_values(by=['date'], ascending=False)

dfgc_US = dfgc_US.sort_values(by=['province_state'], ascending=True)
from ipywidgets import interact, interactive, fixed

import pandas as pd

import ipywidgets as widgets

from IPython.display import display



# https://plotly.com/python/cufflinks/#scatter-plots

import cufflinks as cf



@interact

def scatter_plot(

#                  x=list(dfgc.select_dtypes('number').columns), 

                 x=list(dfgc_US[['confirmed_per_100.000', 'deaths_per_100.000', 'cases_confirmed', 'cases_deaths', 'difference_confirmed', 'difference_deaths']].select_dtypes('number').columns), 

#                  y=list(dfgc.select_dtypes('number').columns)[1:],

                 y=list(dfgc_US[['confirmed_per_100.000', 'deaths_per_100.000', 'cases_confirmed', 'cases_deaths', 'difference_confirmed', 'difference_deaths']].select_dtypes('number').columns),

#                  theme=list(cf.themes.THEMES.keys()), 

#                  colorscale=list(cf.colors._scales_names.keys())

                 log_x=list([True,False]),

                 log_y=list([True,False])

                ):

    

#     widgets.SelectMultiple(

#     options=continents,

#     description='Groups',

#     disabled=False,

# )

    

    g = px.scatter(dfgc_US, 

               x=x, y=y,

               title='<b>Developments COVID19 US</b>\nT25 deaths per '+DDD,

               color="province_state", size="pop_sq_rt", 

               hover_name="province_state", hover_data=["population", "date"],

#                facet_col="province_state",

#                log_x=True, log_y=True,

               log_x=log_x, log_y=log_y,

               width=900, height=500,

               template='plotly_dark',

          )

    

    g.update_layout(

#     title="Plot Title",

    xaxis_title=f'{x.title()}',

    yaxis_title=f'{y.title()}',

    font=dict(

        family="Roboto, monospace",

        size=12,

        color="#eeeeee"

    )

)

    

#     g.update_traces(mode='lines+markers')

    g.show()