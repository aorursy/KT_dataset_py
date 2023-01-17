# This Python 3 environment comes with many helpful analytics libraries installed

# Loading datasets required for analysis



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))
full_table = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', 

                         parse_dates=['Date'])

full_table.head()
# Defining COVID-19 cases as per classifications 

cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']



# Defining Active Case: Active Case = confirmed - deaths - recovered

full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']



# Renaming Mainland china as China in the data table

full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')



# filling missing values 

full_table[['Province/State']] = full_table[['Province/State']].fillna('')

full_table[cases] = full_table[cases].fillna(0)



# cases in the ships

ship = full_table[full_table['Province/State'].str.contains('Grand Princess')|full_table['Country/Region'].str.contains('Cruise Ship')]



# china and the row

china = full_table[full_table['Country/Region']=='China']

row = full_table[full_table['Country/Region']!='China']



# latest

full_latest = full_table[full_table['Date'] == max(full_table['Date'])].reset_index()

china_latest = full_latest[full_latest['Country/Region']=='China']

row_latest = full_latest[full_latest['Country/Region']!='China']



# latest condensed

full_latest_grouped = full_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

china_latest_grouped = china_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

row_latest_grouped = row_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
temp = full_table.groupby(['Country/Region', 'Province/State'])['Confirmed', 'Deaths', 'Recovered', 'Active'].max()
temp = full_table.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)

temp.style.background_gradient(cmap='Pastel1')
temp_f = full_latest_grouped.sort_values(by='Confirmed', ascending=False)

temp_f = temp_f.reset_index(drop=True)

temp_f.style.background_gradient(cmap='Reds')
temp_f.head(10)
import plotly as py

import plotly.graph_objects as go

import pandas as pd

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)    #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON 

#NOTEBOOK WHILE KERNEL IS RUNNING



#Time Series plot for knwoing the spread



fig = go.Figure()

fig.add_trace(go.Scatter(

                x=full_table.Date,

                y=full_table['Confirmed'],

                name="Confirmed",

                line_color='deepskyblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=full_table.Date,

                y=full_table['Recovered'],

                name="Recovered",

                line_color='dimgray',

                opacity=0.8))

fig.update_layout(title_text='Time Series with Rangeslider',

                  xaxis_rangeslider_visible=True)

py.offline.iplot(fig)
import plotly.offline as py

py.init_notebook_mode(connected=True)



# Calculating the count of confirmed cases by country



countries = np.unique(temp_f['Country/Region'])

mean_conf = []

for country in countries:

    mean_conf.append(temp_f[temp_f['Country/Region'] == country]['Confirmed'].sum())

    

# Building the dataframe



    data = [ dict(

        type = 'choropleth',

        locations = countries,

        z = mean_conf,

        locationmode = 'country names',

        text = countries,

        marker = dict(

            line = dict(color = 'rgb(0,0,0)', width = 1)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = 'Count')

            )

       ]

    

# Building the visual



    layout = dict(

    title = 'COVID-19 Confirmed Cases',

    geo = dict(

        showframe = False,

        showocean = True,

        oceancolor = 'rgb(0,255,255)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )



fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap')
import pandas as pd

global_temp_country = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv")
global_temp_country.head()
import plotly.offline as py

py.init_notebook_mode(connected=True)



## Removing the duplicates



global_temp_country_clear = global_temp_country[~global_temp_country['Country'].isin(

    ['Denmark', 'Antarctica', 'France', 'Europe', 'Netherlands',

     'United Kingdom', 'Africa', 'South America'])]



global_temp_country_clear = global_temp_country_clear.replace(

   ['Denmark (Europe)', 'France (Europe)', 'Netherlands (Europe)', 'United Kingdom (Europe)'],

   ['Denmark', 'France', 'Netherlands', 'United Kingdom'])



#Calculating average temperature by country



countries = np.unique(global_temp_country_clear['Country'])

mean_temp = []

for country in countries:

    mean_temp.append(global_temp_country_clear[global_temp_country_clear['Country'] == 

                                               country]['AverageTemperature'].mean())



# Building the data frame

    

data = [ dict(

        type = 'choropleth',

        locations = countries,

        z = mean_temp,

        locationmode = 'country names',

        text = countries,

        marker = dict(

            line = dict(color = 'rgb(0,0,0)', width = 1)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = '# Average\nTemperature,\n°C')

            )

       ]



# Building the visual



layout = dict(

    title = 'GLOBAL AVERAGE LAND TEMPERATURES',

    geo = dict(

        showframe = False,

        showocean = True,

        oceancolor = 'rgb(0,255,255)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )



fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap')
import plotly.express as px

import plotly.offline as py

py.init_notebook_mode(connected=True)

formated_gdf = full_table.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])

formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.3)



fig = px.scatter_geo(formated_gdf, locations="Country/Region", locationmode='country names', 

                     color="Confirmed", size='size', hover_name="Country/Region", 

                     range_color= [0, max(formated_gdf['Confirmed'])+2], 

                     projection="natural earth", animation_frame="Date", 

                     title='Progression of spread of COVID-19')

fig.update(layout_coloraxis_showscale=False)

py.offline.iplot(fig)
import warnings



from scipy.optimize import curve_fit

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from tqdm.notebook import tqdm
plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [20, 8]



warnings.filterwarnings('ignore')
train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv',

    parse_dates=['Date']).drop(['Lat', 'Long'], axis=1)

test = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv',

    parse_dates=['Date']).drop(['Lat', 'Long'], axis=1)

submission = pd.read_csv('../input/covid19-global-forecasting-week-1/submission.csv',

    index_col='ForecastId')
train.iloc[6425,4] = 0
# Susceptible equation

def fa(N, a, b, beta):

    fa = -beta*a*b

    return fa



# Infected equation

def fb(N, a, b, beta, gamma):

    fb = beta*a*b - gamma*b

    return fb



# Recovered/deceased equation

def fc(N, b, gamma):

    fc = gamma*b

    return fc
# Runge-Kutta method of 4rth order for 3 dimensions (susceptible a, infected b and recovered r)

def rK4(N, a, b, c, fa, fb, fc, beta, gamma, hs):

    a1 = fa(N, a, b, beta)*hs

    b1 = fb(N, a, b, beta, gamma)*hs

    c1 = fc(N, b, gamma)*hs

    ak = a + a1*0.5

    bk = b + b1*0.5

    ck = c + c1*0.5

    a2 = fa(N, ak, bk, beta)*hs

    b2 = fb(N, ak, bk, beta, gamma)*hs

    c2 = fc(N, bk, gamma)*hs

    ak = a + a2*0.5

    bk = b + b2*0.5

    ck = c + c2*0.5

    a3 = fa(N, ak, bk, beta)*hs

    b3 = fb(N, ak, bk, beta, gamma)*hs

    c3 = fc(N, bk, gamma)*hs

    ak = a + a3

    bk = b + b3

    ck = c + c3

    a4 = fa(N, ak, bk, beta)*hs

    b4 = fb(N, ak, bk, beta, gamma)*hs

    c4 = fc(N, bk, gamma)*hs

    a = a + (a1 + 2*(a2 + a3) + a4)/6

    b = b + (b1 + 2*(b2 + b3) + b4)/6

    c = c + (c1 + 2*(c2 + c3) + c4)/6

    return a, b, c
def SIR(N, b0, beta, gamma, hs):

    

    """

    N = total number of population

    beta = transition rate S->I

    gamma = transition rate I->R

    k =  denotes the constant degree distribution of the network (average value for networks in which 

    the probability of finding a node with a different connectivity decays exponentially fast

    hs = jump step of the numerical integration

    """

    

    # Initial condition

    a = float(N-1)/N -b0

    b = float(1)/N +b0

    c = 0.



    sus, inf, rec= [],[],[]

    for i in range(10000): # Run for a certain number of time-steps

        sus.append(a)

        inf.append(b)

        rec.append(c)

        a,b,c = rK4(N, a, b, c, fa, fb, fc, beta, gamma, hs)



    return sus, inf, rec
 # Parameters of the model

N = 7800*(10**6)

b0 = 0

beta = 0.6 # Tranmission rate reduced considering factor

gamma = 0.3

hs = 0.1



sus, inf, rec = SIR(N, b0, beta, gamma, hs)



f = plt.figure(figsize=(8,5)) 

plt.plot(sus, 'b.', label='susceptible');

plt.plot(inf, 'r.', label='infected');

plt.plot(rec, 'c.', label='recovered/deceased');

plt.title("SIR model")

plt.xlabel("time", fontsize=10);

plt.ylabel("Fraction of population", fontsize=10);

plt.legend(loc='best')

plt.xlim(0,1000)

plt.savefig('SIR_example.png')

plt.show()
def sigmoid(x, m, alpha, beta):

    return m / ( 1 + np.exp(-beta * (x - alpha)))
def get_curve(covid, which):

    covid['DaysPassed'] = covid['Date'].dt.dayofyear

    curve = covid[covid[which] > 0].set_index('DaysPassed')[which]

    if curve.index.size > 4:

        return curve

    



def plot_curve(curve, test, name, plot_n, popt, ax):

    if curve is not None:

        _ = curve.plot(ax=ax[plot_n % 5, plot_n // 5], title=name)

        _.set_xlabel('')

        x = np.append(curve[:-12].index.values, test['Date'].dt.dayofyear.values)

        y = sigmoid(x, popt[0], popt[1], popt[2])

        pd.Series(y, x).plot(ax=ax[plot_n % 5, plot_n // 5], style=':')

    else:

        pd.Series(0).plot(ax=ax[plot_n % 5, plot_n // 5], title=name)



    

def predict_curve(covid, test, popt, which):

    train_curve = get_curve(covid, which)

    if train_curve is not None:

        x_train = train_curve.index.values

        y_train = train_curve.values

        popt, _ = curve_fit(sigmoid, x_train, y_train, p0=popt, maxfev=1000000)

        x_test = test['Date'].dt.dayofyear.values

        y_test = sigmoid(x_test, popt[0], popt[1], popt[2])

        test[which] = y_test

        return test.set_index('ForecastId')[which].astype('int'), popt

    return None, None





def append_predictions(train, test, popts):

    cases_popt, fatalities_popt = popts

    cases, cases_popt = predict_curve(train, test, cases_popt, 'ConfirmedCases')

    if cases is not None:

        CASES_ALL.append(cases)

    fatalities, fatalities_popt = predict_curve(train, test, fatalities_popt, 'Fatalities')

    if fatalities is not None:

        FATALITIES_ALL.append(fatalities)

    return cases_popt, fatalities_popt

   

    

def known_popt(country, region):

    known = {}

    known['cases'] = {

        'Hubei': [67625, 18.7, 0.24],

        'China': [680, 13.3, 0.265],

    }

    known['fatalities'] = {

        'Hubei': [3007, 23.6, 0.17]

    }

    if region in known['cases']:

        cases_popt = known['cases'][region]

    elif country in known['cases']:

        cases_popt = known['cases'][country]

    else:

        cases_popt = [5000, 100, 0.2]

        

    if region in known['fatalities']:

        fatalities_popt = known['fatalities'][region]

    if country in known['fatalities']:

        fatalities_popt = known['fatalities'][country]

    else:

        fatalities_popt = [100, 150, 0.25]

    

    return cases_popt, fatalities_popt

    

    

def main():

    n = -1

    for country in tqdm(train['Country/Region'].unique()):

        country_train = train[train['Country/Region'] == country].copy()

        country_test = test[test['Country/Region'] == country].copy()

        if not country_train['Province/State'].isna().all():

            for region in country_train['Province/State'].unique():

                region_train = country_train[country_train['Province/State'] == region].copy()

                region_test = country_test[country_test['Province/State'] == region].copy()

                cases_popt, fatalities_popt = append_predictions(region_train, region_test, known_popt(country, region))

                if region in ['Hubei', 'Guangdong', 'Hunan', 'California', 'France', 'Netherlands']:

                    n += 1

                    plot_curve(get_curve(region_train, 'ConfirmedCases'), region_test, region, n, cases_popt, AX)

                    plot_curve(get_curve(region_train, 'Fatalities'), region_test, region, n, fatalities_popt, AXX)

        else:

            cases_popt, fatalities_popt = append_predictions(country_train, country_test, known_popt(country, None))

            if country in ['Italy', 'Spain', 'Mexico', 'India']:

                n += 1

                plot_curve(get_curve(country_train, 'ConfirmedCases'), country_test, country, n, cases_popt, AX)

                plot_curve(get_curve(country_train, 'Fatalities'), country_test, country, n, fatalities_popt, AXX)
CASES_ALL = []

FIG, AX = plt.subplots(5, 2)

FIG.suptitle('Confirmed Cases')



FATALITIES_ALL = []

FIGG, AXX = plt.subplots(5, 2)

FIGG.suptitle('Fatalities')



main()



FIG.subplots_adjust(hspace=0.5)

FIGG.subplots_adjust(hspace=0.5)
final = pd.DataFrame(pd.concat(CASES_ALL).reindex(index=submission.index, fill_value=1))

final = final.join(pd.DataFrame(pd.concat(FATALITIES_ALL).reindex(index=submission.index, fill_value=0)))

final = final.where(final['Fatalities'] <= final['ConfirmedCases'], final['ConfirmedCases'] * 0.06, axis=0)

final.to_csv('submission.csv')