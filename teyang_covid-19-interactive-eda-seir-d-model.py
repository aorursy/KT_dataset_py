import numpy as np 

import pandas as pd 



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))





train = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')



country_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-locations-population/locations_population.csv')
# rename columns

train = train.rename(columns={'Province/State': 'Province_State', 'Country/Region': 'Country_Region', 'ConfirmedCases':'Confirmed', 'Fatalities':'Deaths'}).sort_values(['Country_Region','Province_State']).reset_index().drop('index',axis=1)

train['Date'] = pd.to_datetime(train['Date']).astype('str')

test = test.rename(columns={'Province/State': 'Province_State', 'Country/Region': 'Country_Region', 'ConfirmedCases':'Confirmed', 'Fatalities':'Deaths'})

country_data = country_data.rename(columns={'Province.State': 'Province_State', 'Country.Region': 'Country_Region'}).drop('Provenance',axis=1)
train.head()
test.head()
train['Date'].min(), train['Date'].max(), test['Date'].min()
# Remove the overlapping train and test data



valid = train[train['Date'] >= test['Date'].min()] # set as validation data

train = train[train['Date'] < test['Date'].min()]

train.shape, valid.shape
train['Date'].min(), train['Date'].max(), test['Date'].min(), test['Date'].max()
country_data.head()
train = train.merge(country_data, on=['Country_Region','Province_State'], how = 'left')

valid = valid.merge(country_data, on=['Country_Region','Province_State'], how = 'left')

test = test.merge(country_data, on=['Country_Region','Province_State'], how = 'left')
train['Province_State'] = train['Province_State'].fillna(train['Country_Region']) # replace NaN States with country name

valid['Province_State'] = valid['Province_State'].fillna(valid['Country_Region']) # replace NaN States with country name

test['Province_State'] = test['Province_State'].fillna(test['Country_Region']) # replace NaN States with country name
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
train.head()
train_total = train.groupby(['Country_Region','Province_State'],as_index=False).agg({'Confirmed': 'max', 'Deaths': 'max', 'Recovered': 'max'})
# pio.renderers.default = 'vscode'

pio.renderers.default = 'kaggle'



fig = px.treemap(train_total.sort_values(by='Confirmed', ascending=False).reset_index(drop=True), 

                 path=["Country_Region", "Province_State"], values="Confirmed", height=700, width=900,

                 title='Number of Confirmed Cases',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value'

fig.show()
fig = px.treemap(train_total.sort_values(by='Deaths', ascending=False).reset_index(drop=True), 

                 path=["Country_Region", "Province_State"], values="Deaths", height=700, width=900,

                 title='Number of Deaths',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value'

fig.show()
fig = px.treemap(train_total.sort_values(by='Recovered', ascending=False).reset_index(drop=True), 

                 path=["Country_Region", "Province_State"], values="Recovered", height=700, width=900,

                 title='Number of Recovered',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value'

fig.show()
# Sum countries with states, not dealing with states for now

train_agg= train[['Country_Region','Date','Confirmed','Deaths', 'Recovered']].groupby(['Country_Region','Date'],as_index=False).agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'})

# France will sum all its colonies, so it will be higher



# change to datetime format

train_agg['Date'] = pd.to_datetime(train_agg['Date']) 

! pip install pycountry

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



train_continent = train_continent[['Continent','Date','Confirmed','Deaths', 'Recovered']].groupby(['Continent','Date'],as_index=False).agg({'Confirmed':'sum','Deaths':'sum','Recovered':'sum'})

train_continent['Continent'] = train_continent['Continent'].map({'AF':'Africa','AS':'Asia','CN':'China','EU':'Europe','NA':'North America','OC':'Oceania','SA':'South America'})
long = pd.melt(train_continent, id_vars=['Continent','Date'], value_vars=['Confirmed','Deaths','Recovered'], var_name='Case', value_name='Count').sort_values(['Date','Count'])

long['Date'] = long['Date'].astype('str')
long.head()
#pio.renderers.default = 'browser' # does not work on vscode



# color palette

cnf = '#393e46' # confirmed - grey

dth = '#ff2e63' # death - red

rec = '#21bf73' # recovered - cyan

# act = '#fe9801' # active case - yellow



fig = px.bar(long, y='Continent', x='Count', color='Case', barmode='group', orientation='h', text='Count', title='Counts by Continent', animation_frame='Date',

             color_discrete_sequence= [dth,cnf,rec], range_x=[0, 100000])

fig.update_traces(textposition='outside')
# Interactive time series plot of confirmed cases

fig = px.line(train_agg, x='Date', y='Confirmed', color="Country_Region", hover_name="Country_Region")

fig.update_layout(autosize=False,width=1000,height=500,title='Confirmed Cases Over Time for Each Country')

fig.show()
# Interactive time series plot of fatalities

fig = px.line(train_agg, x='Date', y='Deaths', color="Country_Region", hover_name="Country_Region")

fig.update_layout(autosize=False,width=1000,height=500,title='Deaths Over Time for Each Country')

fig.show()
# Interactive time series plot of recovered

fig = px.line(train_agg, x='Date', y='Recovered', color="Country_Region", hover_name="Country_Region")

fig.update_layout(autosize=False,width=1000,height=500,title='Deaths Over Time for Each Country')

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



# function for getting the iso code through fuzzy search

def do_fuzzy_search(country):

    try:

        result = pycountry.countries.search_fuzzy(country)

    except Exception:

        return np.nan

    else:

        return result[0].alpha_3



# manually change name of some countries

train_agg.loc[train_agg['Country_Region'] == 'South Korea', 'Country_Region'] = 'Korea, Republic of'

train_agg.loc[train_agg['Country_Region'] == 'Taiwan*', 'Country_Region'] = 'Taiwan'

train_agg.loc[train_agg['Country_Region'] == 'Burma', 'Country_Region'] = 'Myanmar'

train_agg.loc[train_agg['Country_Region'] == 'Congo (Kinshasa)', 'Country_Region'] = 'Congo, The Democratic Republic of the'

train_agg.loc[train_agg['Country_Region'] == 'Congo (Brazzaville)', 'Country_Region'] = 'Congo'

train_agg.loc[train_agg['Country_Region'] == 'Laos', 'Country_Region'] = "Lao People's Democratic Republic"



# create iso mapping for countries in df

iso_map = {country: do_fuzzy_search(country) for country in train_agg['Country_Region'].unique()}

# apply the mapping to df

train_agg['iso'] = train_agg['Country_Region'].map(iso_map)

noiso = train_agg[train_agg['iso'].isna()]['Country_Region'].unique()

noiso
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
# change date to string, not sure why plotly cannot accept datetime format

train_agg['Date'] = train_agg['Date'].dt.strftime('%Y-%m-%d')

# apply log10 so that color changes are more prominent

import numpy as np

train_agg['Confirmed_log10'] = np.log10(train_agg['Confirmed']).replace(-np.inf, 0) # log10 changes 0 to -inf so change back

# Interactive Map of Confirmed Cases Over Time



#pio.renderers.default = 'browser' # does not work on vscode

# pio.renderers.default = 'kaggle'

fig = px.choropleth(train_agg, locations='iso', color='Confirmed_log10', hover_name='Country_Region', animation_frame='Date', color_continuous_scale='reds')

fig.show()

#train.Province_State.fillna(train.Country_Region, inplace=True) # replace Nan Province_State to Country Name

train['idx'] = train.groupby(['Country_Region','Province_State']).cumcount() # add days from start of series
from scipy.integrate import solve_ivp

from scipy.optimize import minimize

from datetime import timedelta, datetime



# function to get the initial SIR values and their time series values

def getSEIRD(df,country,state):

    SIRD_TS = df[(df['Country_Region']==country) & (df['Province_State']==state)][['idx','Population','Confirmed','Recovered','Deaths']]

    start_idx = next((i for i, x in enumerate(SIRD_TS.Confirmed) if x), None) # start index of 1st infected

    I_ts = SIRD_TS.Confirmed # infected timeseries

    R_ts = SIRD_TS['Recovered'] # recovered time series

    D_ts = SIRD_TS.Deaths # deceased time series

    S_ts = SIRD_TS['Population'] - I_ts - R_ts - D_ts # susceptible = total population - infected - recovered - deceased (time series)

    N = SIRD_TS['Population'].iloc[0] # population size 

    try:

        mortality = D_ts.iloc[-1] / I_ts.iloc[-1] # use country's last day to estimate mortality rate

    except:

        mortality = 0 # if divide by 0 will cause error, set is as 0 for countries with no confirmed cases to date

    



    # SEIRD = [S_ts.iloc[start_idx], I_ts.iloc[start_idx]*20, I_ts.iloc[start_idx], R_ts.iloc[start_idx], D_ts.iloc[start_idx]] # initial values for SIR; exposed is set to 20x infected 

    SEIRD = [S_ts.iloc[start_idx], 0, I_ts.iloc[start_idx], R_ts.iloc[start_idx], D_ts.iloc[start_idx]] # initial values for SIR; exposed is set to 20x infected 

    timespan = len(I_ts) - start_idx # length of time series

    return SIRD_TS, SEIRD, start_idx, N, timespan, mortality



# function to calculate the derivative or rate of changes for the SEIRD components

def deriv(t,SEIRD,N,R_0,gamma,T_inc,mortality):

    

    if callable(R_0): R_t = R_0(t)

    else: R_t = R_0



    beta = gamma*R_t # transmission rate

    S=SEIRD[0]; E = SEIRD[1]; I=SEIRD[2]; R=SEIRD[3]; D=SEIRD[4] # get initial values

    dS = -beta * S * I / N # change in susceptible = transmission rate * how many susceptible there are as well as infected there are (this is -ve change)

    dE = (beta * S * I / N) - ((T_inc**-1) * E) # change in exposed = transmission rate * how many susceptible there are as well as infected there are, minus incubation rate * how many exposed there are

    dI = (T_inc**-1) * E - (gamma + mortality) * I # change in infected = those who were exposed becoming infected (plus as added to infected) minus change in recovered/deceased (minus as moved to recovered)

    dR = gamma * I # change in recovered = recovery rate * how many infected there are (this is +ve change)

    dD = mortality * I # change in death = mortality rate * how many infected there are

    return dS, dE, dI, dR, dD



# function for plotting out SEIRD model prediction and fit to actual data

def plot_predict_model(solution, df, timespan, start_idx, N, state):

    # first plot - SEIRD model predictions

    sus,exp,inf,rec,dea = solution.y # get the predicted values for components

    f = plt.figure(figsize=(16,5))

    ax = f.add_subplot(1,2,1)

    #ax.plot(sus, 'b', label='Susceptible');

    ax.plot(exp/N, 'orange', label='Exposed')

    ax.plot(inf/N, 'r', label='Infected');

    #ax.plot(rec/N, 'c', label='Recovered');

    ax.plot(dea/N, 'black', label='Deceased');

    plt.title('SEIRD Model')

    plt.xlabel("Days", fontsize=10);

    plt.ylabel("Fraction of population", fontsize=10);

    plt.legend(loc='best');



    # second plot - SEIRD model fit to actual data

    # plot only confirmed and deaths

    ax2 = f.add_subplot(1,2,2)

    preds = inf + rec + dea # get back actual counts from proportion

    predD = dea

    # ax2.plot(range(timespan),preds[:len(SIRD_TS)],label = 'Predicted Confirmed Cases')

    ax2.plot(range(timespan),preds[:timespan],label = 'Predicted Confirmed Cases')

    ax2.plot(range(timespan),df['Confirmed'][start_idx:],label = 'Actual Confirmed Cases')

    ax2.plot(range(timespan),predD[:timespan],label = 'Predicted Deaths')

    ax2.plot(range(timespan),df['Deaths'][start_idx:],label = 'Actual Deaths')

    plt.title('Model predict and data')

    plt.ylabel("Population", fontsize=10);

    plt.xlabel("Days", fontsize=10);

    plt.legend(loc='best');

    plt.suptitle(state,size=30)
import math

# function for applying SEIRD to individual country

def SEIRD_by_country(df,country,forecast_days,params,decay=True,state=''):

    state = country if state == '' else state # account for some countries having same state names

    SEIRD_TS,SEIRD,start_idx,N,timespan,mortality=getSEIRD(df,country,state)

    forecast_days = forecast_days # days to forecast ahead



    if decay: # apply decay

        L = params[3]; k = params[4]

        def decaying_reproduction(t):       

            return params[0] / (1 + (t/L)**k)

    else: decaying_reproduction = params[0] # don't apply decay





    # solve for ordinary differential equations

    solution = solve_ivp(deriv, [0,timespan+forecast_days], [SEIRD[0],SEIRD[1],SEIRD[2],SEIRD[3],SEIRD[4]], 

        args=(N,decaying_reproduction,params[1],params[2],mortality), t_eval=np.arange(0, timespan+forecast_days, 1))

    #np.add(solution.y[2], solution.y[3], solution.y[4]) # Confirmed = Infected + Recovered + Deceased, have to do so because original data is cumulative counts

    plot_predict_model(solution, SEIRD_TS, timespan, start_idx, N, state)

    return solution
import matplotlib.pyplot as plt



R_0 = 2.7 # reproduction number

beta = 2.75 # transmission rate

T_inc = 3.3 # incubation duration

L = 10 # time of inflection point 

k = 2 # steepness of decaying curve

params = [R_0, beta, T_inc, L, k]



solution = SEIRD_by_country(train,'China',100,params,decay=True,state='Hubei')
R_0 = 2 # reproduction number

beta = 1.6 # transmission rate

T_inc = 3.6 # incubation duration

L = 5 # time of inflection point 

k = 2 # steepness of decaying curve

params = [R_0, beta, T_inc, L, k]



solution = SEIRD_by_country(train,'Italy',100,params,decay=False)
R_0 = 1.9 # reproduction number

beta = 2 # transmission rate

#gamma = 4.6 # recovery rate

T_inc = 2 # incubation duration

L = 10 # time of inflection point 

k = 3 # steepness of decaying curve

params = [R_0, beta, T_inc, L, k]



solution = SEIRD_by_country(train,'Iran',100,params,decay=False)
R_0 = 2.75 # reproduction number

beta = 2.8 # transmission rate

#gamma = 4.6 # recovery rate

T_inc = 6 # incubation duration

L = 20 # time of inflection point 

k = 1 # steepness of decaying curve

params = [R_0, beta, T_inc, L, k]



solution = SEIRD_by_country(train,'Singapore',100,params,decay=True)