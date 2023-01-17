import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from itertools import cycle, islice

import seaborn as sb

import matplotlib.dates as dates

import datetime as dt



import plotly.offline as py

py.init_notebook_mode(connected=True)

from plotly import tools, subplots

import plotly.figure_factory as ff

import plotly.express as px

import plotly.graph_objects as go



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")#index_col=0

display(train_data.head())

test_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")#index_col=0

display(test_data.head())
sum_df = pd.pivot_table(train_data, values=['ConfirmedCases','Fatalities'], index=['Date'],aggfunc=np.sum)

display(sum_df.max())
train_data['NewConfirmedCases'] = train_data['ConfirmedCases'] - train_data['ConfirmedCases'].shift(1)

train_data['NewConfirmedCases'] = train_data['NewConfirmedCases'].fillna(0.0)

train_data['NewFatalities']     = train_data['Fatalities'] - train_data['Fatalities'].shift(1)

train_data['NewFatalities']     = train_data['NewFatalities'].fillna(0.0)#.astype(int)

train_data['MortalityRate']     = train_data['Fatalities'] / train_data['ConfirmedCases']

train_data['MortalityRate']     = train_data['MortalityRate'].fillna(0.0)

train_data['GrowthRate']        = train_data['NewConfirmedCases']/train_data['NewConfirmedCases'].shift(1)

train_data['GrowthRate']        = train_data['GrowthRate'].replace([-np.inf, np.inf],  0.0)

train_data['GrowthRate']        = train_data['GrowthRate'].fillna(0.0) 

display(train_data.head())
def getColumnInfo(df):

    n_province =  df['Province_State'].nunique()

    n_country  =  df['Country_Region'].nunique()

    n_days     =  df['Date'].nunique()

    start_date =  df['Date'].unique()[0]

    end_date   =  df['Date'].unique()[-1]

    return n_province, n_country, n_days, start_date, end_date



n_train = train_data.shape[0]

n_test = test_data.shape[0]



n_prov_train, n_count_train, n_train_days, start_date_train, end_date_train = getColumnInfo(train_data)

n_prov_test,  n_count_test,  n_test_days,  start_date_test,  end_date_test  = getColumnInfo(test_data)



print ('<==Train data==> \n # of Province_State: '+str(n_prov_train),', # of Country_Region:'+str(n_count_train), 

       ', Time Period: '+str(start_date_train)+' to '+str(end_date_train), '==> days:',str(n_train_days))

print("\n Countries with Province/State information:  ", train_data[train_data['Province_State'].isna()==False]['Country_Region'].unique())

print ('\n <==Test  data==> \n # of Province_State: '+str(n_prov_test),', # of Country_Region:'+str(n_count_test),

       ', Time Period: '+start_date_test+' to '+end_date_test, '==> days:',n_test_days)



df_test = test_data.loc[test_data.Date > '2020-04-14']

overlap_days = n_test_days - df_test.Date.nunique()

print('\n overlap days with training data: ', overlap_days, ', total days: ', n_train_days+n_test_days-overlap_days)
prob_confirm_check_train = train_data.ConfirmedCases.value_counts(normalize=True)

prob_fatal_check_train = train_data.Fatalities.value_counts(normalize=True)



n_confirm_train = train_data.ConfirmedCases.value_counts()[1:].sum()

n_fatal_train = train_data.Fatalities.value_counts()[1:].sum()



print('Percentage of confirmed case records = {0:<2.0f}/{1:<2.0f} = {2:<2.1f}%'.format(n_confirm_train, n_train, prob_confirm_check_train[1:].sum()*100))

print('Percentage of fatality records = {0:<2.0f}/{1:<2.0f} = {2:<2.1f}%'.format(n_fatal_train, n_train, prob_fatal_check_train[1:].sum()*100))
train_data_by_country = train_data.groupby(['Date','Country_Region'],as_index=False).agg({'ConfirmedCases': 'sum', 'Fatalities': 'sum',

                                                                                         'GrowthRate':'last' })

#display(train_data_by_country.tail(10))

max_train_date = train_data['Date'].max()

train_data_by_country_confirm = train_data_by_country.query('(Date == @max_train_date) & (ConfirmedCases > 100)').sort_values('ConfirmedCases', ascending=False)

train_data_by_country_confirm.set_index('Country_Region', inplace=True)



train_data_by_country_confirm.style.background_gradient(cmap='Reds').format({'ConfirmedCases': "{:.0f}", 'GrowthRate': "{:.2f}"})

from itertools import cycle, islice

discrete_col = list(islice(cycle(['orange', 'r', 'g', 'k', 'b', 'c', 'm']), None, len(train_data_by_country_confirm.head(30))))

plt.rcParams.update({'font.size': 22})

train_data_by_country_confirm.head(20).plot(figsize=(20,15), kind='barh', color=discrete_col)

plt.legend(["Confirmed Cases", "Fatalities"]);

plt.xlabel("Number of Covid-19 Affectees")

plt.title("First 20 Countries with Highest Confirmed Cases")

ylocs, ylabs = plt.yticks()

for i, v in enumerate(train_data_by_country_confirm.head(20)["ConfirmedCases"][:]):

    plt.text(v+0.01, ylocs[i]-0.25, str(int(v)), fontsize=12)

for i, v in enumerate(train_data_by_country_confirm.head(20)["Fatalities"][:]):

    if v > 0: #disply for only >300 fatalities

        plt.text(v+0.01,ylocs[i]+0.1,str(int(v)),fontsize=12)    
def reformat_time(reformat, ax):

    ax.xaxis.set_major_locator(dates.WeekdayLocator())

    ax.xaxis.set_major_formatter(dates.DateFormatter('%b %d'))    

    if reformat: #reformat again if you wish

        date_list = train_data_by_date.reset_index()["Date"].tolist()

        x_ticks = [dt.datetime.strftime(t,'%Y-%m-%d') for t in date_list]

        x_ticks = [tick for i,tick in enumerate(x_ticks) if i%8==0 ]# split labels into same number of ticks as by pandas

        ax.set_xticklabels(x_ticks, rotation=90)

    # cosmetics

    ax.yaxis.grid(linestyle='dotted')

    ax.spines['right'].set_color('none')

    ax.spines['top'].set_color('none')

    ax.spines['left'].set_color('none')

    ax.spines['bottom'].set_color('none')



train_data['Date'] = pd.to_datetime(train_data['Date'])

train_data_by_date = train_data.groupby(['Date'],as_index=True).agg({'ConfirmedCases': 'sum','Fatalities': 'sum', 

                                                                     'NewConfirmedCases':'sum', 'NewFatalities':'sum', 'MortalityRate':'mean'})

num0 = train_data_by_date._get_numeric_data() 

num0[num0 < 0.0] = 0.0

#display(train_data_by_date.head())



## ======= Sort by countries with fatalities > 600 ========



train_data_by_country_max = train_data.groupby(['Country_Region'],as_index=True).agg({'ConfirmedCases': 'max', 'Fatalities': 'max'})

train_data_by_country_fatal = train_data_by_country_max[train_data_by_country_max['Fatalities']>600]

train_data_by_country_fatal = train_data_by_country_fatal.sort_values(by=['Fatalities'],ascending=False).reset_index()

#display(train_data_by_country_fatal.head(20))



df_merge_by_country = pd.merge(train_data,train_data_by_country_fatal['Country_Region'],on=['Country_Region'],how='inner')

df_max_fatality_country = df_merge_by_country.groupby(['Date','Country_Region'],as_index=False).agg({'ConfirmedCases': 'sum',

                                                                                                     'Fatalities': 'sum',

                                                                                                     'NewConfirmedCases':'sum',

                                                                                                     'NewFatalities':'sum',

                                                                                                     'MortalityRate':'mean'})



num1 = df_max_fatality_country._get_numeric_data() 

num1[num1 < 0.0] = 0.0

df_max_fatality_country.set_index('Date',inplace=True)

#display(df_max_fatality_country.head(20))



countries = train_data_by_country_fatal['Country_Region'].unique()



plt.rcParams.update({'font.size': 16})



fig,(ax0,ax1) = plt.subplots(1,2,figsize=(15, 8))

fig,(ax2,ax3) = plt.subplots(1,2,figsize=(15, 8))#,sharey=True)



train_data_by_date.ConfirmedCases.plot(ax=ax0, x_compat=True, title='Confirmed Cases Globally', legend='Confirmed Cases',

                                       color=discrete_col)#, logy=True)

reformat_time(0,ax0)

train_data_by_date.NewConfirmedCases.plot(ax=ax0, x_compat=True, linestyle='dotted', legend='New Confirmed Cases',

                                          color=discrete_col)#, logy=True)

reformat_time(0,ax0)



train_data_by_date.Fatalities.plot(ax=ax2, x_compat=True, title='Fatalities Globally', legend='Fatalities', color='r')

reformat_time(0,ax2)

train_data_by_date.NewFatalities.plot(ax=ax2, x_compat=True, linestyle='dotted', legend='Daily Deaths',color='r')#tell pandas not to use its own datetime format

reformat_time(0,ax2)



for country in countries:

    match = df_max_fatality_country.Country_Region==country

    df_fatality_by_country = df_max_fatality_country[match] 

    df_fatality_by_country.ConfirmedCases.plot(ax=ax1, x_compat=True, title='Confirmed Cases Nationally')

    reformat_time(0,ax1)

    df_fatality_by_country.Fatalities.plot(ax=ax3, x_compat=True, title='Fatalities Nationally')

    reformat_time(0,ax3)

    

#ax1.legend(countries)

#ax3.legend(countries)

ax1.legend(countries, loc='center left',bbox_to_anchor=(1.0, 0.5))

ax3.legend(countries, loc='center left',bbox_to_anchor=(1.0, 0.5))

fig = plt.figure()

fig,(ax4,ax5) = plt.subplots(1,2,figsize=(20, 8))

#train_data_by_date.loc[(train_data_by_date.ConfirmedCases > 200)]#useless, its already summed.

train_data_by_date.MortalityRate.plot(ax=ax4, x_compat=True, legend='Mortality Rate',color='r')#tell pandas not to use its own datetime format

reformat_time(0,ax4)



for num, country in enumerate(countries):

    match = df_max_fatality_country.Country_Region==country 

    df_fatality_by_country = df_max_fatality_country[match] 

    df_fatality_by_country.MortalityRate.plot(ax=ax5, x_compat=True, title='Average Mortality Rate Nationally')    

    reformat_time(0,ax5)



ax5.legend(countries, loc='center left',bbox_to_anchor=(1.0, 0.5))
train_data_by_max_date = train_data_by_country.query('(Date == @max_train_date) & (ConfirmedCases > 100)')

train_data_by_max_date.loc[:, 'MortalityRate'] = train_data_by_max_date.loc[:,'Fatalities']/train_data_by_max_date.loc[:,'ConfirmedCases']

train_data_by_mortality = train_data_by_max_date.sort_values('MortalityRate', ascending=False)

train_data_by_mortality.set_index('Country_Region', inplace=True)

#display(train_data_by_mortality.head())



palette = plt.get_cmap('OrRd_r')

rainbow_col = [palette(1.*i/20.0) for i in range(20)]



train_data_by_mortality.MortalityRate.head(20).plot(figsize=(15,10), kind='barh', color=rainbow_col)

plt.xlabel("Mortality Rate")

plt.title("First 20 Countries with Highest Mortality Rate")

ylocs, ylabs = plt.yticks()



#import plotly.io as pio              # to set shahin plot layout



world_df = train_data_by_country.query('Date == @max_train_date')

world_df.loc[:,'Date']           = world_df.loc[:,'Date'].apply(str)

world_df.loc[:,'Confirmed_log']  = round(np.log10(world_df.loc[:,'ConfirmedCases'] + 1), 3)

world_df.loc[:,'Fatalities_log'] = np.log10(world_df.loc[:,'Fatalities'] + 1)

world_df.loc[:,'MortalityRate']  = round(world_df.loc[:, 'Fatalities'] / world_df.loc[:,'ConfirmedCases'], 3)

world_df.loc[:,'GrowthFactor']  = round(world_df.loc[:,'GrowthRate'], 3)

#display(world_df.head())



fig1 = px.choropleth(world_df, locations="Country_Region", 

                    locationmode="country names",  

                    color="Confirmed_log",                     

                    hover_name="Country_Region",

                    hover_data=['ConfirmedCases', 'Fatalities', 'MortalityRate', 'GrowthFactor'],

                    range_color=[world_df['Confirmed_log'].min(), world_df['Confirmed_log'].max()], 

                    color_continuous_scale = px.colors.sequential.Plasma,

                    title='COVID-19: Confirmed Cases')

fig1.show()

fig2 = px.scatter_geo(world_df, 

                     locations="Country_Region", 

                     locationmode="country names", 

                     color="ConfirmedCases", size='ConfirmedCases', 

                     hover_name="Country_Region", 

                     hover_data=['ConfirmedCases', 'Fatalities', 'MortalityRate', 'GrowthFactor'],

                     range_color= [world_df['Confirmed_log'].min(), world_df['ConfirmedCases'].max()], 

                     projection="natural earth", 

                     animation_frame="Date",

                     animation_group="Country_Region",

                     color_continuous_scale="portland",

                     title='COVID-19: Spread Over Time')



#fig2.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 10

#fig2.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 10

fig2.layout.coloraxis.showscale = False

#fig2.layout.sliders[0].pad.t = 10

#fig2.layout.updatemenus[0].pad.t= 10

fig2.show()
#world_population = pd.read_csv("/kaggle/input/population-by-country-2020/population_by_country_2020.csv")

#display(world_population.head()) #for next round
from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures 

from sklearn.pipeline import make_pipeline

from tqdm import tqdm



plt.rcParams.update({'font.size': 12})

fig,(ax0,ax1) = plt.subplots(1,2,figsize=(20, 8))

countries_europe = ['Italy', 'France', 'Spain', 'Germany', 'United Kingdom']



# Take the 1st day as 2020-02-23

df = train_data.loc[train_data.Date >= '2020-02-23']

n_days_europe = df.Date.nunique()

rainbow_col= plt.cm.jet(np.linspace(0,1,len(countries)))



for country, c in tqdm(zip(countries,rainbow_col)): 

    df_country_train = df_max_fatality_country[df_max_fatality_country['Country_Region']==country] 

    df_country_test = test_data[test_data['Country_Region']==country]  

    df_country_train = df_country_train.reset_index()[df_country_train.reset_index().Date > '2020-02-22']

    n_days_sans_China = df.Date.nunique() - df_country_train.Date.nunique() 

    

    x_train = np.arange(1, n_days_europe+1).reshape((-1,1))

    x_test  = (np.arange(1,n_days_europe+n_test_days+1-overlap_days)).reshape((-1,1)) 

    y_train_f = df_country_train['Fatalities']

    #print (x_train, y_train_f)

    model_f = make_pipeline(PolynomialFeatures(degree=3), Ridge(fit_intercept=False)) 

    model_f = model_f.fit(x_train, y_train_f)

    y_predict_f = model_f.predict(x_test) 

    #print (x_test[-n_test_days:], y_predict_f[-n_test_days:])

    y_train_c = df_country_train['ConfirmedCases'] 

    model_c = make_pipeline(PolynomialFeatures(degree=3), Ridge(fit_intercept=False)) 

    model_c = model_c.fit(x_train, y_train_c)

    y_predict_c = model_c.predict(x_test)

    

    extend_days_test = [i+len(x_test) for i in range(n_days_sans_China)]

    x_test      = np.append(x_test, extend_days_test) 

    y_predict_c = np.pad(y_predict_c, (n_days_sans_China, 0), 'constant')

    y_predict_f = np.pad(y_predict_f, (n_days_sans_China, 0), 'constant')

    

    ax0.plot(x_test[-n_test_days:], y_predict_c[-n_test_days:],linewidth=2, label='predict_'+country, color=c)

    ax0.plot(x_train, y_train_c, linewidth=2, color=c, linestyle='dotted', label='train_'+country)

    ax0.set_title("Prediction vs Training for Confirmed Cases")

    ax0.set_xlabel("Number of days")

    ax0.set_ylabel("Confirmed Cases")

    #ax0.legend(loc='center left',bbox_to_anchor=(1.0, 0.5))

    #ax0.set_yscale('log')

    

    ax1.plot(x_test[-(n_test_days):], y_predict_f[-(n_test_days):],linewidth=2, label='predict_'+country, color=c)

    ax1.plot(x_train, y_train_f, linewidth=2, color=c, linestyle='dotted', label='train_'+country)

    ax1.set_title("Prediction vs Training for Fatalities")

    ax1.set_xlabel("Number of days")

    ax1.set_ylabel("Fatalities")

    ax1.legend(loc='center left',bbox_to_anchor=(1.0, 0.5))

    #ax1.set_yscale('log')
from scipy.optimize.minpack import curve_fit

from sklearn.metrics import r2_score

from scipy.special import expit



def Gompertz(a, c, t, t0):    

    Q = a * np.exp(-np.exp(-c*(t-t0)))

    return Q

def Boltzman(a, c, t, t0):

    Q = a / (1 + np.exp(-c*(t-t0)))

    return Q

emerging_countries = ['Albania', 'Andorra', 'Argentina', 'Armenia', 'Azerbaijan', 'Bahrain', 

                      'Barbados', 'Bhtan', 'Bulgaria', 'Burkina Faso', 'Cambodia', 'Chile', 

                      'Colombia', 'Congo (Kinshasa)', 'Costa Rica', 'Cote d’Ivoire', 'Croatia', 

                      'Cuba', 'Cyprus', 'Czechia', 'Dominican Republic', 'Egypt', 'Estonia', 

                      'Georgia', 'Greece', 'Honduras', 'Iceland', 'Iraq', 'Israel', 'Jamaica', 

                      'Japan', 'Jordan', 'Kuwait', 'Latvia', 'Lebanon', 'Lithuania', 

                      'Luxembourg', 'Malaysia', 'Maldives', 'Malta', 'Mauritania', 'Mauritius', 'Monaco',

                      'Mongolia', 'Montenegro', 'Morocco', 'Namibia', 'Nigeria', 'North Macedonia', 

                      'Norway', 'Oman', 'Panama','Paraguay', 'Rawanda', 'Saint Lucia', 'San Marino', 

                      'Senegal', 'Seychelles', 'Singapore','Slovakia', 'Slovenia', 'Sri Lanka', 'Thailand', 

                      'Tunisia', 'Uganda', 'Uruguay', 'Venezuela']

def get_bounds_fatal (country, isState, y_train):

    x = ''

    for c in emerging_countries:

        if country == c: 

            x = c; break

    maximum = max(y_train)

    if maximum == 0.0: maximum = 1.0         

    if country == 'China':

        lower = [0, 0.02, 0]

        upper = [2.0*maximum,0.16, 40]

    elif country == 'Iran':

        lower = [0, 0.00, 0]

        upper = [3.0*maximum,0.11, 68]

    elif country == 'Italy':

        lower = [0, 0.00, 0]

        upper = [3.0*maximum,0.13, 72]       

    elif country == 'US':

        lower = [0, 0.02, 0]

        if maximum <=10:upper = [4.0*maximum, 0.30, 85] 

        else:           upper = [3.5*maximum, 0.20, 90] 

    elif country == 'France':

        lower = [0, 0.02, 0]

        if maximum <=10:upper = [4.0*maximum,0.18, 80]

        else:           upper = [4.0*maximum,0.15, 90] 

    elif country == 'Spain':

        lower = [0, 0.02, 0]

        upper = [3.0*maximum,0.15, 78]

    elif country == 'Germany':

        lower = [0.0, 0.02, 0]

        upper = [3.0*maximum,0.20, 85] 

    elif country == 'Belgium':

        lower = [0.0, 0.02, 0]

        upper = [3.0*maximum,0.25, 88] 

    elif country == 'Turkey':

        lower = [0.0, 0.02, 0]

        upper = [3.5*maximum,0.22, 90]

    elif country == 'Netherlands':

        lower = [0.0, 0.02, 0]

        upper = [4.0*maximum,0.14, 88] 

    elif country == 'Switzerland':

        lower = [0.0, 0.02, 0]

        upper = [4.0*maximum,0.12, 90] 

    elif country == 'United Kingdom':

        lower = [0.0, 0.02, 0]

        upper = [4.5*maximum,0.16, 95]

    elif country == 'Portugal':

        lower = [100, 0.02, 0]

        upper = [4.5*maximum,0.12, 95]  

    elif country == 'Sweden':

        lower = [100, 0.02, 0]

        upper = [4.0*maximum,0.18, 90] 

    elif country == 'Brazil':

        lower = [100, 0.02, 0]

        upper = [3.5*maximum,0.20, 90] 

    elif country == 'Indonesia':

        lower = [100, 0.02, 0]

        upper = [4.5*maximum,0.10, 95]  

    elif country == 'Austria':

        lower = [0, 0.02, 0]

        upper = [4.5*maximum,0.10, 95]  

    elif country == 'Ireland':

        lower = [0, 0.02, 0]

        upper = [4.5*maximum,0.15, 95]          

    elif country == 'Canada':

        lower = [0, 0.02, 0]

        if maximum <=10: upper = [2.0*maximum, 0.20, 65] 

        else:            upper = [4.5*maximum, 0.16, 95]     

    elif country == 'India':

        lower = [0, 0.02, 0]

        upper = [4.5*maximum,0.20, 95]  

    elif country == 'Ecuador':

        lower = [0, 0.02, 0]

        upper = [4.5*maximum,0.16, 96]  

    elif country == 'Romania':

        lower = [0, 0.02, 0]

        upper = [4.5*maximum,0.15, 95]  

    elif country == 'Philippines':

        lower = [0, 0.02, 0]

        upper = [4.5*maximum,0.12, 95]    

    elif country == 'Algeria':

        lower = [0, 0.02, 0]

        upper = [4.5*maximum,0.12, 95]     

    elif country == 'Mexico':

        lower = [0, 0.02, 0]

        upper = [4.5*maximum,0.20, 95]       

    elif country == 'Denmark':

        lower = [0, 0.02, 0]

        if maximum <=10:upper = [4.0*maximum, 0.30, 80] 

        else:           upper = [4.5*maximum,0.12, 94]      

    elif country == 'Poland':

        lower = [0, 0.02, 0]

        upper = [4.0*maximum,0.20, 94]  

    elif country == 'Korea, South':

        lower = [0, 0.02, 0]

        upper = [2.5*maximum,0.10, 52] 

    elif country == 'Peru':

        lower = [0.0, 0.02, 0]

        upper = [4.5*maximum,0.18, 95] 

    elif country == 'Australia':

        lower = [0, 0.02, 0]

        if maximum <=10: upper = [2.0*maximum, 0.20, 45] 

        else:            upper = [2.5*maximum,0.20, 70]

    elif country == 'Pakistan':

        lower = [0.0, 0.02, 0] 

        upper = [4.5*maximum,0.12,95]

    elif country == 'Saudi Arabia':

        lower = [0.0, 0.02, 0] 

        upper = [4.5*maximum,0.15,95]     

    elif country == 'Afghanistan':

        lower = [0.0, 0.02, 0] 

        upper = [4.5*maximum,0.12,95]

    elif country == 'Diamond Princess':

        lower = [0.0, 0.02, 0] 

        upper = [1.0*maximum,0.50,2] 

    elif country == 'Hungary':

        lower = [0.0, 0.02, 0] 

        upper = [4.5*maximum,0.14,94]

    elif country == 'New Zealand':

        lower = [0.0, 0.02, 0] 

        upper = [4.0*maximum,0.14,90]

    elif country == 'Somalia':

        lower = [0.0, 0.02, 0] 

        upper = [4.5*maximum,0.10,94] 

    elif country == x:

        lower = [0.0, 0.02, 0] 

        upper = [3.5*maximum,0.15,85]  

    else:

        lower = [0.0, 0.02, 0] 

        if isState:

            if maximum <=10:upper = [4.0*maximum,0.30,80] 

            else:           upper = [4.5*maximum,0.15,80]

        else: 

            if maximum <=10:upper = [4.0*maximum,0.60,85] 

            else:           upper = [4.5*maximum,0.18,95]  

                

    return lower, upper



def get_bounds_confirm (country, isState, y_train):

    x = ''

    for c in emerging_countries:

        if country == c: 

            x = c; break

    maximum = max(y_train)

    if maximum == 0.0: maximum = 1.0        

    if country == 'China':

        lower = [0, 0.02, 0]

        upper = [2.0*maximum,0.20,30]

    elif country == 'Iran':

        lower = [0, 0.00, 0]

        upper = [3.0*maximum,0.12,70]

    elif country == 'Italy':

        lower = [0, 0.00, 0]

        upper = [3.0*maximum,0.12, 70]

    elif country == 'US':

        lower = [0, 0.02, 0]

        if maximum <=10:upper = [4.0*maximum, 0.30, 80] 

        else:           upper = [3.0*maximum, 0.18, 85]     

    elif country == 'France':

        lower = [0, 0.02, 0]

        if maximum <=10:upper = [4.0*maximum, 0.15, 80] 

        else:           upper = [4.5*maximum, 0.10, 90]             

    elif country == 'Spain':

        lower = [0, 0.02, 0]

        upper = [3.0*maximum,0.13, 75] 

    elif country == 'Germany':

        lower = [0, 0.02, 0]

        upper = [3.0*maximum,0.13, 75] 

    elif country == 'Belgium':

        lower = [0, 0.02, 0]

        upper = [3.0*maximum,0.15, 78]

    elif country == 'Turkey':

        lower = [0, 0.02, 0]

        upper = [3.5*maximum,0.20, 90] 

    elif country == 'Netherlands':

        lower = [0, 0.02, 0]

        upper = [4.0*maximum,0.10, 88] 

    elif country == 'Switzerland':

        lower = [0, 0.02, 0]

        upper = [3.5*maximum,0.10, 75]  

    elif country == 'United Kingdom':

        lower = [0, 0.02, 0]

        upper = [4.5*maximum,0.12, 95] 

    elif country == 'Portugal':

        lower = [0, 0.02, 0]

        upper = [4.0*maximum,0.11, 88]   

    elif country == 'Sweden':

        lower = [0, 0.02, 0]

        upper = [4.0*maximum,0.10, 88]    

    elif country == 'Brazil':

        lower = [0, 0.02, 0]

        upper = [3.5*maximum,0.18, 88]  

    elif country == 'Indonesia':

        lower = [0, 0.02, 0]

        upper = [5.5*maximum,0.09, 100] 

    elif country == 'Austria':

        lower = [0, 0.02, 0]

        upper = [3.5*maximum,0.12, 75] 

    elif country == 'Ireland':

        lower = [0, 0.02, 0]

        upper = [4.5*maximum,0.12, 95]          

    elif country == 'Canada':

        lower = [0, 0.02, 0]

        if maximum <=10: upper = [3.0*maximum, 0.28, 75] 

        else:            upper = [4.5*maximum, 0.12, 93]            

    elif country == 'India':

        lower = [0, 0.02, 0]

        upper = [4.5*maximum,0.16, 96] 

    elif country == 'Ecuador':

        lower = [0, 0.02, 0]

        upper = [4.5*maximum,0.20, 95] 

    elif country == 'Romania':

        lower = [0, 0.02, 0]

        upper = [4.5*maximum,0.11, 93]  

    elif country == 'Philippines':

        lower = [0, 0.02, 0]

        upper = [5.5*maximum,0.12, 95]  

    elif country == 'Algeria':

        lower = [0, 0.02, 0]

        upper = [5.5*maximum,0.10, 98] 

    elif country == 'Mexico':

        lower = [100, 0.02, 0]

        upper = [4.5*maximum,0.15, 95]        

    elif country == 'Denmark':

        lower = [0, 0.02, 0]

        if isState:

            if maximum <= 10: upper = [2.0*maximum,0.20,80] 

            else:             upper = [2.5*maximum,0.25, 55]    

        else:

            if maximum <=10: upper = [2.0*maximum,0.30, 40] 

            else:            upper = [5.5*maximum,0.06, 100]       

    elif country == 'Poland':

        lower = [0, 0.02, 0]

        upper = [4.5*maximum,0.11, 94]

    elif country == 'Korea, South':

        lower = [0, 0.02, 0]

        upper = [2.0*maximum,0.25, 18] 

    elif country == 'Peru':

        lower = [0, 0.02, 0]

        upper = [4.5*maximum,0.20, 96] 

    elif country == 'Australia':

        lower = [0, 0.02, 0]

        if maximum <=10: upper = [2.0*maximum, 0.25, 45] 

        else:            upper = [2.5*maximum,0.18, 65] 

    elif country == 'Pakistan':

        lower = [0.0, 0.02, 0] 

        upper = [4.5*maximum,0.10,94]

    elif country == 'Saudi Arabia':

        lower = [0.0, 0.02, 0] 

        upper = [4.5*maximum,0.10,94]    

    elif country == 'Afghanistan':

        lower = [0.0, 0.02, 0] 

        upper = [4.5*maximum,0.12,94]

    elif country == 'Diamond Princess':

        lower = [0.0, 0.02, 0] 

        upper = [1.0*maximum,1.0,1.0]

    elif country == 'Hungary':

        lower = [0.0, 0.02, 0] 

        upper = [4.5*maximum,0.10,94]  

    elif country == 'New Zealand':

        lower = [0.0, 0.02, 0] 

        upper = [4.5*maximum,0.15,85] 

    elif country == 'Somalia':

        lower = [0.0, 0.02, 0] 

        upper = [1.0*maximum,0.08,50] 

    elif country == x:

        lower = [0.0, 0.02, 0] 

        upper = [3.5*maximum,0.10,80]  

    else:

        lower = [0.0, 0.02, 0] 

        if isState:

            if maximum <= 200: upper = [2.0*maximum,0.20,80] 

            else:              upper = [4.5*maximum,0.20,80]

        else:  

            if maximum <= 200: upper = [3.0*maximum,0.20,85]  

            else:              upper = [4.5*maximum,0.20,96]    

                

    return lower, upper 



plt.rcParams.update({'font.size': 12})

fig,(ax0,ax1) = plt.subplots(1,2,figsize=(20, 8))

fig,(ax2,ax3) = plt.subplots(1,2,figsize=(20, 8))



rainbow_col= plt.cm.jet(np.linspace(0,1,len(countries)))



for country, c in tqdm(zip(countries,rainbow_col)): 

    #print('\n\n\n\n country ==>', country)

    df_country_train = df_max_fatality_country[df_max_fatality_country['Country_Region']==country] 

    df_country_test = test_data[test_data['Country_Region']==country]  

    if country != 'China':

        df_country_train = df_country_train.reset_index().loc[df_country_train.reset_index().Date>'2020-02-22'] #17

        n_days_sans_China =train_data.Date.nunique() - df_country_train.Date.nunique()        

    else:

        df_country_train = df_country_train.reset_index()

        n_days_sans_China = 0

        

    n_train_days =df_country_train.Date.nunique()    

    x_train = range(n_train_days)

    x_test  = range(n_train_days+n_test_days-overlap_days)#n_test_days+overlap_days)

    y_train_f = df_country_train['Fatalities']

    y_train_c = df_country_train['ConfirmedCases'] 

    y_train_cn = (df_country_train['ConfirmedCases'] - df_country_train['ConfirmedCases'].shift(1)).fillna(0.0).replace([-np.inf, np.inf],  0.0)

    y_train_fn = (df_country_train['Fatalities'] - df_country_train['Fatalities'].shift(1)).fillna(0.0).replace([-np.inf, np.inf],  0.0)

    

    ###### Fatalities:   

    lower, upper = get_bounds_fatal (country, 0, y_train_f)

    popt_f, pcov_f = curve_fit(Gompertz, x_train, y_train_f, method='trf', bounds=(lower,upper))

    a_max, estimated_c, estimated_t0 = popt_f

    y_predict_f = Gompertz(a_max, estimated_c, x_test, estimated_t0)

    y_predict_f_at_t0 =  Gompertz(a_max, estimated_c, estimated_t0, estimated_t0)

    #print('\nfatalities ==>, max: ',a_max, ', slope: %.2f'% estimated_c, ', inflection point: ', 

    #      estimated_t0, ', r2 score: %.2f'% r2_score(y_train_f[:], y_predict_f[0:n_train_days]))

    y_fn = np.array([])

    fn = [y_predict_f[i]-y_predict_f[i-1] if i!=0 else y_predict_f[i] for i in range(len(y_predict_f))]    

    y_predict_fn = np.append(y_fn, fn)

   

    ###### Confirmed cases:    

    lower_c,upper_c = get_bounds_confirm (country, 0, y_train_c)

    popt_c, pcov_c = curve_fit(Gompertz, x_train, y_train_c, method='trf', bounds=(lower_c,upper_c))

    a_max_c, estimated_c_c, estimated_t0_c = popt_c

    y_predict_c = Gompertz(a_max_c, estimated_c_c, x_test, estimated_t0_c)

    y_predict_c_at_t0 =  Gompertz(a_max_c, estimated_c_c, estimated_t0_c, estimated_t0_c)

    #print('confirmed ==> max: ',a_max_c, ', slope: %.2f'% estimated_c_c, ', inflection point: ', 

    #      estimated_t0_c, ', r2 score: %.2f'% r2_score(y_train_c[:], y_predict_c[0:n_train_days]))

    y_cn = np.array([])

    cn = [y_predict_c[i]-y_predict_c[i-1] if i!=0 else y_predict_c[i] for i in range(len(y_predict_c))]    

    y_predict_cn = np.append(y_cn, cn)

       

    ## ===== Move the x-axis of trained and test datasets to allign with dates in China ======

    extend_days_test = [i+len(x_test) for i in range(n_days_sans_China)]

    x_test       = np.append(x_test, extend_days_test) 

    y_predict_c  = np.pad(y_predict_c, (n_days_sans_China, 0), 'constant')

    y_predict_cn = np.pad(y_predict_cn,(n_days_sans_China, 0), 'constant')

    y_predict_f  = np.pad(y_predict_f, (n_days_sans_China, 0), 'constant')

    y_predict_fn = np.pad(y_predict_fn, (n_days_sans_China, 0), 'constant')

    inflection_c = estimated_t0_c+n_days_sans_China



    extend_days_train = [i+len(x_train) for i in range(n_days_sans_China)]

    x_train      = np.append(x_train, extend_days_train)

    y_train_c    = np.pad(y_train_c, (n_days_sans_China, 0), 'constant')

    y_train_cn   = np.pad(y_train_cn, (n_days_sans_China, 0), 'constant')

    y_train_f    = np.pad(y_train_f, (n_days_sans_China, 0), 'constant')

    y_train_fn  = np.pad(y_train_fn, (n_days_sans_China, 0), 'constant')

    inflection_f = estimated_t0+n_days_sans_China

    

    ## ===== Plot =======

    ax0.plot(x_test, y_predict_c, linewidth=2, label=country, color=c) 

    ax0.plot(inflection_c, y_predict_c_at_t0, marker='o', markersize=6, color='green')#, label='inflection')

    ax0.plot(x_train, y_train_c, linewidth=2, color=c,linestyle='dotted')#, label='train_'+country)   

    ax0.set_title("Total Confirmed Cases")

    ax0.set_xlabel("Number of days")

    ax0.set_ylabel("Confirmed Cases")

    ax0.legend()#loc='center left',bbox_to_anchor=(1.0, 0.5))

    

    ax1.plot(x_test, y_predict_f, linewidth=2, label=country,color=c) 

    ax1.plot(inflection_f, y_predict_f_at_t0, marker='o', markersize=6, color='green')

    ax1.plot(x_train, y_train_f, linewidth=2,color=c, linestyle='dotted')#, label='train_'+country)    

    ax1.set_title("Total Fatalities")

    ax1.set_xlabel("Number of days")

    ax1.set_ylabel("Fatalities")

    ax1.legend()#loc='center left',bbox_to_anchor=(1.0, 0.5))

    

    ax2.plot(x_test, y_predict_cn, linewidth=2, label=country, color=c) 

    ax2.scatter(x_train, y_train_cn, linewidth=2, color=c, linestyle='dotted')#, label='train_'+country)   

    ax2.set_title("New Confirmed Cases")

    ax2.set_xlabel("Number of days")

    ax2.set_ylabel("New Confirmed Cases")

    ax2.legend()#loc='center left',bbox_to_anchor=(1.0, 0.5))

    

    ax3.plot(x_test, y_predict_fn, linewidth=2, label=country, color=c) 

    ax3.scatter(x_train, y_train_fn, linewidth=2, color=c, linestyle='dotted')#, label='train_'+country)   

    ax3.set_title("New Fatalities")

    ax3.set_xlabel("Number of days")

    ax3.set_ylabel("New Fatalities")

    ax3.legend()#loc='center left',bbox_to_anchor=(1.0, 0.5))
nCountries= train_data['Country_Region'].unique() 

isState = bool

x_train = range(n_train_days)

x_test  = range(n_train_days+n_test_days-overlap_days)



for country in tqdm(nCountries): 

    fig,(ax0,ax1) = plt.subplots(1,2,figsize=(20,8))

    fig,(ax2,ax3) = plt.subplots(1,2,figsize=(20,8))

    #print('\n\n\n\n country ==>', country) 

    

    df_country_train = train_data[train_data['Country_Region']==country] 

    df_country_test = test_data[test_data['Country_Region']==country]  

    

    if country != 'China':

        df_country_train = df_country_train.reset_index().loc[df_country_train.reset_index().Date>'2020-02-22'] #17

        n_days_sans_China =train_data.Date.nunique() - df_country_train.Date.nunique()        

    else:

        df_country_train = df_country_train.reset_index()

        n_days_sans_China = 0

        

    n_train_days =df_country_train.Date.nunique()    

    x_train = range(n_train_days)

    x_test  = range(n_train_days+n_test_days-overlap_days)   

    nvalues = df_country_train['Province_State'].isna().nunique() #fix for problem with Denmark data

    

    if (df_country_train['Province_State'].isna().unique()==True).any() and nvalues<2: 

        isState = False        

        y_train_f = df_country_train['Fatalities']

        y_train_c = df_country_train['ConfirmedCases']  

        y_train_cn = (df_country_train['ConfirmedCases'] - df_country_train['ConfirmedCases'].shift(1)).fillna(0.0)

        y_train_fn = (df_country_train['Fatalities'] - df_country_train['Fatalities'].shift(1)).fillna(0.0)

        

        if y_train_f.empty == False:

            lower, upper = get_bounds_fatal (country, isState, y_train_f)

            #print(lower, upper)

            popt_f, pcov_f = curve_fit(Gompertz, x_train, y_train_f, method='trf', bounds=(lower,upper))

            a_max, estimated_c, estimated_t0 = popt_f

            y_predict_f = Gompertz(a_max, estimated_c, x_test, estimated_t0)            

            #print('\nfatalities ==>, max: ',a_max, ', slope: %.2f'% estimated_c, ', inflection point: ', 

             #     estimated_t0, ', r2 score: %.2f'% r2_score(y_train_f[:], y_predict_f[0:n_train_days]))

            y_fn = np.array([])

            fn = [y_predict_f[i]-y_predict_f[i-1] if i!=0 else y_predict_f[i] for i in range(len(y_predict_f))]    

            y_predict_fn = np.append(y_fn, fn)

   

            

        if y_train_c.empty == False:  

            lower_c, upper_c = get_bounds_confirm (country, isState, y_train_c)

            #print(lower_c, upper_c)

            popt_c, pcov_c = curve_fit(Gompertz, x_train, y_train_c, method='trf', bounds=(lower_c,upper_c))

            a_max_c, estimated_c_c, estimated_t0_c = popt_c

            y_predict_c = Gompertz(a_max_c, estimated_c_c, x_test, estimated_t0_c)

            #print('\nconfirmed ==> max: ',a_max_c, ', slope: %.2f'% estimated_c_c, ', inflection point: ', 

             #     estimated_t0_c, ', r2 score: %.2f'% r2_score(y_train_c[:], y_predict_c[0:n_train_days]))

            y_cn = np.array([])

            cn = [y_predict_c[i]-y_predict_c[i-1] if i!=0 else y_predict_c[i] for i in range(len(y_predict_c))]    

            y_predict_cn = np.append(y_cn, cn)

            

        ## ===== Move the x-axis of trained and test datasets to allign with dates in China ======

        extend_days_test = [i+len(x_test) for i in range(n_days_sans_China)]

        x_test       = np.append(x_test, extend_days_test)                         

        y_predict_c  = np.pad(y_predict_c, (n_days_sans_China, 0), 'constant')

        y_predict_cn = np.pad(y_predict_cn,(n_days_sans_China, 0), 'constant')

        y_predict_f  = np.pad(y_predict_f, (n_days_sans_China, 0), 'constant')

        inflection_f = estimated_t0+n_days_sans_China

        y_predict_fn = np.pad(y_predict_fn, (n_days_sans_China, 0), 'constant')

            

        extend_days_train = [i+len(x_train) for i in range(n_days_sans_China)]

        x_train      = np.append(x_train, extend_days_train)           

        y_train_c    = np.pad(y_train_c, (n_days_sans_China, 0), 'constant')

        y_train_cn   = np.pad(y_train_cn, (n_days_sans_China, 0), 'constant')

        y_train_f    = np.pad(y_train_f, (n_days_sans_China, 0), 'constant')

        y_train_fn   = np.pad(y_train_fn, (n_days_sans_China, 0), 'constant')

        inflection_c = estimated_t0_c+n_days_sans_China           

        

        ax0.plot(x_test, y_predict_c, linewidth=2, label='predict_'+country) 

        ax0.plot(x_train, y_train_c, linewidth=2, color='r', linestyle='dotted', label='train_'+country)

        ax0.set_title("Prediction vs Training for Confirmed Cases")

        ax0.set_xlabel("Number of days")

        ax0.set_ylabel("Confirmed Cases")

        ax0.legend()

        test_data.loc[test_data['Country_Region']==country,'ConfirmedCases'] = y_predict_c[-n_test_days:]

        

        ax1.plot(x_test, y_predict_f, linewidth=2, label='predict_'+country) 

        ax1.plot(x_train, y_train_f, linewidth=2, color='r', linestyle='dotted', label='train_'+country)    

        ax1.set_title("Prediction vs Training for Fatalities")

        ax1.set_xlabel("Number of days")

        ax1.set_ylabel("Fatalities")

        ax1.legend()

        test_data.loc[test_data['Country_Region']==country,'Fatalities'] = y_predict_f[-n_test_days:] 

                

        ax2.plot(x_test, y_predict_cn, linewidth=2, label='predict_'+country) 

        ax2.scatter(x_train, y_train_cn, linewidth=2, color='r', linestyle='dotted', label='train_'+country)   

        ax2.set_title("New Confirmed Cases")

        ax2.set_xlabel("Number of days")

        ax2.set_ylabel("New Confirmed Cases")

        ax2.legend()#loc='center left',bbox_to_anchor=(1.0, 0.5))

    

        ax3.plot(x_test, y_predict_fn, linewidth=2, label='predict_'+country) 

        ax3.scatter(x_train, y_train_fn, linewidth=2, color='r', linestyle='dotted', label='train_'+country)   

        ax3.set_title("New Fatalities")

        ax3.set_xlabel("Number of days")

        ax3.set_ylabel("New Fatalities")

        ax3.legend()#loc='center left',bbox_to_anchor=(1.0, 0.5))

    

    else: # use Province/State data when available

        isState = True

        state_list = []

        y_predict_c_dict = {}; y_train_c_dict = {}

        y_predict_cn_dict = {}; y_train_cn_dict = {}

        y_predict_f_dict = {}; y_train_f_dict = {}

        y_predict_fn_dict = {}; y_train_fn_dict = {}

        for state in df_country_train['Province_State'].unique():

            df_state_train = df_country_train[df_country_train['Province_State']==state] #state

            df_state_test = df_country_test[df_country_test['Province_State']==state]   

            state_list.append(state)

            y_train_f = df_state_train['Fatalities']

            y_train_c = df_state_train['ConfirmedCases']  

            y_train_cn = (df_state_train['ConfirmedCases'] - df_state_train['ConfirmedCases'].shift(1)).fillna(0.0)

            y_train_fn = (df_state_train['Fatalities'] - df_state_train['Fatalities'].shift(1)).fillna(0.0)

            

            if y_train_f.empty== False:                 

                lower, upper = get_bounds_fatal (country, isState, y_train_f)

                popt_f, pcov_f = curve_fit(Gompertz, x_train, y_train_f, method='trf', bounds=(lower,upper))

                a_max, estimated_c, estimated_t0 = popt_f

                y_predict_f = Gompertz(a_max, estimated_c, x_test, estimated_t0) 

                y_predict_f_dict[state] =  y_predict_f

                y_train_f_dict[state]   =  y_train_f                

                #print('\nfatalities state ==>, max: ',a_max, ', slope: %.2f'% estimated_c, ', inflection point: ', 

                #    estimated_t0, ', r2 score: %.2f'% r2_score(y_train_f[:], y_predict_f[0:70]))

                y_fn = np.array([])

                fn = [y_predict_f[i]-y_predict_f[i-1] if i!=0 else y_predict_f[i] for i in range(len(y_predict_f))]    

                y_predict_fn = np.append(y_fn, fn)

                y_predict_fn_dict[state] = y_predict_fn

                y_train_fn_dict[state]   = y_train_fn

                                

            if y_train_c.empty == False:  

                lower_c, upper_c = get_bounds_confirm (country, isState, y_train_c)

                popt_c, pcov_c = curve_fit(Gompertz, x_train, y_train_c, method='trf', bounds=(lower_c,upper_c))

                a_max_c, estimated_c_c, estimated_t0_c = popt_c

                y_predict_c = Gompertz(a_max_c, estimated_c_c, x_test, estimated_t0_c)

                y_predict_c_dict[state] =  y_predict_c

                y_train_c_dict[state]   =  y_train_c

                #print('\nconfirmed state ==> max: ',a_max_c, ', slope: %.2f'% estimated_c_c, ', inflection point: ', 

                #  estimated_t0_c, ', r2 score: %.2f'% r2_score(y_train_c[:], y_predict_c[0:70]))                

                y_cn = np.array([])

                cn = [y_predict_c[i]-y_predict_c[i-1] if i!=0 else y_predict_c[i] for i in range(len(y_predict_c))]    

                y_predict_cn = np.append(y_cn, cn)

                y_predict_cn_dict[state] = y_predict_cn

                y_train_cn_dict[state]   = y_train_cn

                            

        ## ====== Plot and Store the Results: ======

        ## ====== Move the x-axis of trained and test datasets to allign with dates in China ======       

        extend_days_test = [i+len(x_test) for i in range(n_days_sans_China)]

        x_test      = np.append(x_test, extend_days_test) 

        extend_days_train = [i+len(x_train) for i in range(n_days_sans_China)]

        x_train     = np.append(x_train, extend_days_train)           

            

        for state, y_predict in y_predict_f_dict.items():

            y_predict = np.pad(y_predict, (n_days_sans_China, 0), 'constant') 

            ax1.plot(x_test, y_predict, linewidth=2, label=country+'_'+state) 

            ax1.legend(loc='center left',bbox_to_anchor=(1.0, 0.5)) 

            test_data.loc[(test_data['Country_Region']==country)&(test_data['Province_State']==state),'Fatalities'] = y_predict[-n_test_days:]

        for state, y_train in y_train_f_dict.items():

            y_train   = np.pad(y_train, (n_days_sans_China, 0), 'constant')

            ax1.plot(x_train, y_train, linewidth=2, color='r', linestyle='dotted', label='train_'+state)             

        ax1.set_title("Prediction vs Training for Fatalities")

        ax1.set_xlabel("Number of days")

        ax1.set_ylabel("Fatalities")   

        

        

        for state, y_predict in y_predict_c_dict.items():

            y_predict = np.pad(y_predict, (n_days_sans_China, 0), 'constant') 

            ax0.plot(x_test, y_predict, linewidth=2, label=country+'_'+state) 

            #ax0.legend(loc='center left',bbox_to_anchor=(1.0, 0.5)) 

            test_data.loc[(test_data['Country_Region']==country)&(test_data['Province_State']==state),'ConfirmedCases'] = y_predict[-n_test_days:]

        for state, y_train in y_train_c_dict.items():

            y_train   = np.pad(y_train, (n_days_sans_China, 0), 'constant')

            ax0.plot(x_train, y_train, linewidth=2, color='r', linestyle='dotted', label='train_'+country+'_'+state)             

        ax0.set_title("Prediction vs Training for ConfirmedCases")

        ax0.set_xlabel("Number of days")

        ax0.set_ylabel("Confirmed Cases") 

        

        for state, y_predict in y_predict_fn_dict.items():

            y_predict = np.pad(y_predict, (n_days_sans_China, 0), 'constant') 

            ax3.plot(x_test, y_predict, linewidth=2, label=country+'_'+state) 

            ax3.legend(loc='center left',bbox_to_anchor=(1.0, 0.5)) 

        for state, y_train in y_train_fn_dict.items():

            y_train   = np.pad(y_train, (n_days_sans_China, 0), 'constant')

            ax3.scatter(x_train, y_train, linewidth=2, color='r', linestyle='dotted', label='train_'+state)    

        ax3.set_title("New Fatalities")

        ax3.set_xlabel("Number of days")

        ax3.set_ylabel("New Fatalities")

        

        

        for state, y_predict in y_predict_cn_dict.items():

            y_predict = np.pad(y_predict, (n_days_sans_China, 0), 'constant') 

            ax2.plot(x_test, y_predict, linewidth=2, label=country+'_'+state) 

            #ax2.legend(loc='center left',bbox_to_anchor=(1.0, 0.5)) 

            test_data.loc[(test_data['Country_Region']==country)&(test_data['Province_State']==state),'ConfirmedCases'] = y_predict[-n_test_days:]

        for state, y_train in y_train_cn_dict.items():

            y_train   = np.pad(y_train, (n_days_sans_China, 0), 'constant')

            ax2.scatter(x_train, y_train, linewidth=2, color='r', linestyle='dotted', label='train_'+country+'_'+state)

        ax2.set_title("New Confirmed Cases")

        ax2.set_xlabel("Number of days")

        ax2.set_ylabel("New Confirmed Cases")
submit_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/submission.csv")#, index_col=0)



test_data['Fatalities'] = test_data['Fatalities'].fillna(0.0).astype(int)

test_data['ConfirmedCases'] = test_data['ConfirmedCases'].fillna(0.0).astype(int)



submit_data['Fatalities'] = test_data['Fatalities'].astype('int')

submit_data['ConfirmedCases'] = test_data['ConfirmedCases'].astype('int')



submit_data.to_csv('submission.csv', index=False)

submit_data.head()
display(submit_data.describe())