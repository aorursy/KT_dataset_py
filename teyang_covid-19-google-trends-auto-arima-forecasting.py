import numpy as np 

import pandas as pd 



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')



# train = pd.read_csv(r'C:\Users\TeYan\OneDrive\Work\Kaggle\COVID19\Data\train.csv')

# test = pd.read_csv(r'C:\Users\TeYan\OneDrive\Work\Kaggle\COVID19\Data\test.csv')

# train = pd.read_csv('/Users/teyang/OneDrive/Work/Kaggle/COVID19/Data/train.csv')

# test = pd.read_csv('/Users/teyang/OneDrive/Work/Kaggle/COVID19/Data/test.csv')
# rename columns

train = train.rename(columns={'Province/State': 'Province_State', 'Country/Region': 'Country_Region'})

test = test.rename(columns={'Province/State': 'Province_State', 'Country/Region': 'Country_Region'})
train.head()
test.head()
train['Date'].max(), test['Date'].min()
# Remove the overlapping train and test data



valid = train[train['Date'] >= test['Date'].min()] # set as validation data

train = train[train['Date'] < test['Date'].min()]

train.shape, valid.shape
# Standard plotly imports

#import chart_studio.plotly as py

import plotly.graph_objs as go

import plotly.express as px

import plotly.io as pio

from plotly.subplots import make_subplots

from plotly.offline import iplot, init_notebook_mode, plot

# Using plotly + cufflinks in offline mode

import cufflinks

cufflinks.go_offline(connected=True)

init_notebook_mode(connected=True)
train_total = train[['Country_Region','Province_State','ConfirmedCases','Fatalities']]

train_total['Province_State'] = train_total['Province_State'].fillna(train_total['Country_Region']) # replace NaN States with country name

train_total = train_total.groupby(['Country_Region','Province_State'],as_index=False).agg({'ConfirmedCases': 'max', 'Fatalities': 'max'})
# pio.renderers.default = 'vscode'

pio.renderers.default = 'kaggle'



fig = px.treemap(train_total.sort_values(by='ConfirmedCases', ascending=False).reset_index(drop=True), 

                 path=["Country_Region", "Province_State"], values="ConfirmedCases", height=600, width=800,

                 title='Number of Confirmed Cases',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value'

fig.show()



fig = px.treemap(train_total.sort_values(by='Fatalities', ascending=False).reset_index(drop=True), 

                 path=["Country_Region", "Province_State"], values="Fatalities", height=600, width=800,

                 title='Number of Deaths',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value'

fig.show()
# Sum countries with states, not dealing with states for now

train_agg= train[['Country_Region','Date','ConfirmedCases','Fatalities']].groupby(['Country_Region','Date'],as_index=False).agg({'ConfirmedCases': 'sum', 'Fatalities': 'sum'})



# change to datetime format

train_agg['Date'] = pd.to_datetime(train_agg['Date'])

! pip install pycountry_convert

import pycountry_convert as pc

import pycountry

# function for getting the iso code through fuzzy search

def do_fuzzy_search(country):

    try:

        result = pycountry.countries.search_fuzzy(country)

    except Exception:

        return np.nan

    else:

        return result[0].alpha_2



train_continent = train_agg

# manually change name of some countries

train_continent.loc[train_continent['Country_Region'] == 'Korea, South', 'Country_Region'] = 'Korea, Republic of'

train_continent.loc[train_continent['Country_Region'] == 'Taiwan*', 'Country_Region'] = 'Taiwan'

# create iso mapping for countries in df

iso_map = {country: do_fuzzy_search(country) for country in train_continent['Country_Region'].unique()}

# apply the mapping to df

train_continent['iso'] = train_continent['Country_Region'].map(iso_map)

#train_continent['Continent'] = [pc.country_alpha2_to_continent_code(iso) for iso in train_continent['iso']]

def alpha2_to_continent(iso):

    try: cont = pc.country_alpha2_to_continent_code(iso)

    except: cont = float('NaN')

    return cont



train_continent['Continent'] = train_continent['iso'].apply(alpha2_to_continent) # get continent code

train_continent.loc[train_continent['iso'] == 'CN', 'Continent'] = 'CN' # Replace China's continent value as we want to keep it separate



train_continent = train_continent[['Continent','Date','ConfirmedCases','Fatalities']].groupby(['Continent','Date'],as_index=False).agg({'ConfirmedCases':'sum','Fatalities':'sum'})

train_continent['Continent'] = train_continent['Continent'].map({'AF':'Africa','AS':'Asia','CN':'China','EU':'Europe','NA':'North America','OC':'Oceania','SA':'South America'})
long = pd.melt(train_continent, id_vars=['Continent','Date'], value_vars=['ConfirmedCases','Fatalities'], var_name='Case', value_name='Count').sort_values(['Date','Count'])

long['Date'] = long['Date'].astype('str')
pio.renderers.default = 'kaggle' # does not work on vscode



# color palette

cnf = '#393e46' # confirmed - grey

dth = '#ff2e63' # death - red

# rec = '#21bf73' # recovered - cyan

# act = '#fe9801' # active case - yellow



fig = px.bar(long, y='Continent', x='Count', color='Case', barmode='group', orientation='h', text='Count', title='Counts by Continent', animation_frame='Date',

             color_discrete_sequence= [dth,cnf], range_x=[0, 100000])

fig.update_traces(textposition='outside')
# Interactive time series plot of confirmed cases

fig = px.line(train_agg, x='Date', y='ConfirmedCases', color="Country_Region", hover_name="Country_Region")

fig.update_layout(autosize=False,width=1000,height=500,title='Confirmed Cases Over Time for Each Country')

fig.show()
# Interactive time series plot of fatalities

fig = px.line(train_agg, x='Date', y='Fatalities', color="Country_Region", hover_name="Country_Region")

fig.update_layout(autosize=False,width=1000,height=500,title='Fatalities Over Time for Each Country')

fig.show()
## Load Natural Earth Map Data



import geopandas as gpd # for reading vector-based spatial data format

shapefile = '/kaggle/input/natural-earth-maps/ne_110m_admin_0_countries.shp'

#shapefile = r'C:\Users\TeYan\OneDrive\Work\Kaggle\COVID19\110m_cultural\ne_110m_admin_0_countries.shp'



# Read shapefile using Geopandas

#gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]

gdf = gpd.read_file(shapefile)



# Drop row corresponding to 'Antarctica'

gdf = gdf.drop(gdf.index[159])
## Get the ISO 3166-1 alpha-3 Country Codes



import pycountry

# function for getting the iso code through fuzzy search

def do_fuzzy_search(country):

    try:

        result = pycountry.countries.search_fuzzy(country)

    except Exception:

        return np.nan

    else:

        return result[0].alpha_3



# manually change name of some countries

train_agg.loc[train_agg['Country_Region'] == 'Korea, South', 'Country_Region'] = 'Korea, Republic of'

train_agg.loc[train_agg['Country_Region'] == 'Taiwan*', 'Country_Region'] = 'Taiwan'

# create iso mapping for countries in df

iso_map = {country: do_fuzzy_search(country) for country in train_agg['Country_Region'].unique()}

# apply the mapping to df

train_agg['iso'] = train_agg['Country_Region'].map(iso_map)

# # function for getting the better country name through fuzzy search

# def do_fuzzy_search_country(country):

#     try:

#         result = pycountry.countries.search_fuzzy(country)

#     except Exception:

#         return np.nan

#     else:

#         return result[0].name



# country_map = {country: do_fuzzy_search_country(country) for country in train_agg['Country_Region'].unique()}

# # apply the mapping to df

# train_agg['Country_Region'] = train_agg['Country_Region'].map(country_map)
# countries with no iso

noiso = train_agg[train_agg['iso'].isna()]['Country_Region'].unique()

# get other iso from natural earth data, create the mapping and add to our old mapping

otheriso = gdf[gdf['SOVEREIGNT'].isin(noiso)][['SOVEREIGNT','SOV_A3']]

otheriso = dict(zip(otheriso.SOVEREIGNT, otheriso.SOV_A3))

iso_map.update(otheriso)
# apply mapping and find countries with no iso again

train_agg['iso'] = train_agg['Country_Region'].map(iso_map)

train_agg[train_agg['iso'].isna()]['Country_Region'].unique()
# change date to string, not sure why plotly cannot accept datetime format

train_agg['Date'] = train_agg['Date'].dt.strftime('%Y-%m-%d')

# apply log10 so that color changes are more prominent

import numpy as np

train_agg['ConfirmedCases_log10'] = np.log10(train_agg['ConfirmedCases']).replace(-np.inf, 0) # log10 changes 0 to -inf so change back

# Interactive Map of Confirmed Cases Over Time



#pio.renderers.default = 'browser' # does not work on vscode

pio.renderers.default = 'kaggle'

fig = px.choropleth(train_agg, locations='iso', color='ConfirmedCases_log10', hover_name='Country_Region', animation_frame='Date', color_continuous_scale='reds')

fig.show()

# load google trends data

#cv = pd.read_csv(r'C:\Users\TeYan\OneDrive\Work\Kaggle\COVID19\GoogleTrends\coronavirus.csv', encoding = 'ISO-8859-1')

#covid = pd.read_csv(r'C:\Users\TeYan\OneDrive\Work\Kaggle\COVID19\GoogleTrends\covid19.csv', encoding = 'ISO-8859-1')



cv = pd.read_csv('/kaggle/input/covid19-googletrends/coronavirus.csv', encoding = 'ISO-8859-1')

covid = pd.read_csv('/kaggle/input/covid19-googletrends/covid19.csv', encoding = 'ISO-8859-1')
cv = cv.merge(covid, left_on=['Country','iso','date'],right_on=['Country','iso','date'],suffixes=('_cv', '_covid')) # merging removes some small countries but that's alright

cv['hits'] = cv[['hits_cv','hits_covid']].max(axis=1) # get whichever has a higher proportion between the 2 keywords



cv['iso'] = [pycountry.countries.get(alpha_2=a).alpha_3 for a in cv['iso']] # get the alpha_3 codes from the alpha_2 to merge with confirmed cases df

cc_google = train_agg.merge(cv, left_on=['iso','Date'], right_on=['iso','date']) # merge confirmed cases df with google trend df
import seaborn as sns



sns.regplot(x='hits',y='ConfirmedCases_log10',data=cc_google,scatter_kws={'s':25},fit_reg=True, line_kws={"color": "black"})



# does not work on Kaggle

# p = sns.jointplot(x="hits", y="ConfirmedCases_log10", data=cc_google, kind='reg',

#                   joint_kws={'line_kws':{'color':'black'}}, bw=0.1)

# p.fig.set_figwidth(10)                    
popCountries = cc_google[cc_google['Country_Region'].isin(['Singapore','US','Italy','Iran'])] # select the countries



# separate confirmed cases (cc) and hits (h) columns to normalize them by group, then merge back the columns

pc_cc = popCountries[['Country_Region','Date','ConfirmedCases']] # popular countries confirmed cases

pc_f = popCountries[['Country_Region','Date','Fatalities']] # popular countries fatalities

pc_h = popCountries[['Country_Region','Date','hits']] # popular countries hits

# min-max normalization

pc_cc=pc_cc.assign(ConfirmedCases=pc_cc.groupby('Country_Region').transform(lambda x: (x - x.min()) / (x.max() - x.min()))) 

pc_f=pc_f.assign(Fatalities=pc_f.groupby('Country_Region').transform(lambda x: (x - x.min()) / (x.max() - x.min()))) 

pc_h=pc_h.assign(hits=pc_h.groupby('Country_Region').transform(lambda x: (x - x.min()) / (x.max() - x.min())))

# merge back the columns

popCountries = pc_cc.merge(pc_h, left_on=['Country_Region','Date'], right_on=['Country_Region','Date'])

popCountries = popCountries.merge(pc_f, left_on=['Country_Region','Date'], right_on=['Country_Region','Date'])

popCountries = popCountries[['Country_Region','Date','ConfirmedCases','Fatalities','hits']]

popCountries = popCountries.rename(columns={'ConfirmedCases':'val1','Fatalities':'val2','hits':'val3'})

# convert to long format for plotting

long = pd.wide_to_long(popCountries, stubnames='val', i=['Country_Region','Date'], j='CC_F_Hits').reset_index()



# Replace values with labels

def replaceVal(x):

    if x == 1: val = 'Confirmed Cases'

    elif x == 2: val = 'Fatalities'

    else: val = 'Hits'

    return val



long['CC_F_Hits'] = long['CC_F_Hits'].apply(replaceVal)
# plot facet line plots for each country

import matplotlib.pyplot as plt



g = sns.relplot(x="Date", y="val",

            hue="CC_F_Hits", col="Country_Region", col_wrap=2,

            height=4, aspect=1.45, facet_kws=dict(sharex=False),

            kind="line", legend="full", data=long)

g.set_xticklabels(rotation=45,fontsize=5,horizontalalignment='right')

g.fig.subplots_adjust(top=0.9)

g.fig.suptitle('Confirmed Cases & Google Search Trajectories for 4 Countries', fontsize=16)

g.fig.set_figheight(10)

g.set_axis_labels(y_var="Normalized Confirmed Cases & Google Search Hits")
# get Iran

ir = cc_google[cc_google['Country_Region'] == 'Iran'].reset_index()

ir = ir[['Date','ConfirmedCases','Fatalities','hits']]

ir.Date = pd.to_datetime(ir.Date)

ir.index = ir.Date  # reassign the index.
# correlation of confirmed cases and google trend

import scipy.stats

sns.regplot(x='hits',y='ConfirmedCases',data=ir,scatter_kws={'s':25},fit_reg=True, line_kws={"color": "black"})



scipy.stats.pearsonr(ir.ConfirmedCases, ir.hits)
! pip install pmdarima
# auto-arima

import pmdarima as pm



model = pm.auto_arima(ir[['ConfirmedCases']], exogenous=ir[['hits']], # include google trends data as external regressor

                        start_p=1, start_q=1,

                        test='adf',       # use adftest to find optimal 'd'

                        max_p=3, max_q=3, # maximum p and q

                        m=1,              # frequency of series

                        d=None,           # let model determine 'd'

                        seasonal=False,   # No Seasonality

                        start_P=0, D=0, trace=True, 

                        error_action='ignore', suppress_warnings=True, 

                        stepwise=True)

print(model.summary())
model.plot_diagnostics(figsize=(7,5))

plt.show()
# get the exogeneous regressor values (google trend) for the forecasting period

exo = cv

exo = cv[(pd.to_datetime(cv['date']) > ir.Date.max()) & (cv['Country'] == 'Iran')]

exo.index = pd.to_datetime(exo['date'])



# get the validation confirmed cases for the forecasting period

ir_val = valid[valid.Country_Region == 'Iran']

ir_val.Date = pd.to_datetime(ir_val.Date)

ir_val.index = ir_val.Date
from datetime import timedelta



# Forecast

n_periods = 7 # validation only has 7 days now

fitted, confint = model.predict(n_periods=n_periods, 

                                  exogenous=np.tile(exo.hits, 1).reshape(-1,1), # predict using google trend

                                  return_conf_int=True)



index_of_fc = pd.date_range(ir.index[-1] + timedelta(days=1), periods = n_periods, freq='D') # get the date index range of the forecasting period



# make series for plotting purpose

fitted_series = pd.Series(fitted, index=index_of_fc)

lower_series = pd.Series(confint[:, 0], index=index_of_fc)

upper_series = pd.Series(confint[:, 1], index=index_of_fc)



# Plot

plt.plot(ir.ConfirmedCases, label='Fitting')

plt.plot(fitted_series, color='darkgreen', label='Predicted')

plt.plot(ir_val.ConfirmedCases, color='red', label='Actual')

plt.fill_between(lower_series.index, 

                 lower_series, 

                 upper_series, 

                 color='k', alpha=.15)



plt.title("ARIMA Forecast of Confirmed Cases for Iran")

plt.xticks(rotation=45, horizontalalignment='right')

plt.legend(loc="upper left")

plt.show()