from __future__ import print_function

%matplotlib inline

import pandas as pd

import json

import numpy as np

from datetime import date 

import matplotlib.pyplot as plt

import statsmodels.api as sm

from scipy import stats

from statsmodels.graphics.api import qqplot

import warnings

warnings.simplefilter(action='ignore', category=(FutureWarning,UserWarning,RuntimeWarning))

from ipywidgets import interact, interactive, fixed, interact_manual

import ipywidgets as widgets

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.stats.stattools import durbin_watson 

from scipy.integrate import odeint
#Highlight the min cells in a table row or column



def highlight_min(s):

    '''

    highlight the maximum in a Series red.

    '''

    is_min = s == s.min()

    return ['background-color: red' if v else '' for v in is_min]
#Source - covid19india.org

i=1

c_data=pd.DataFrame()

while True:

    try:

        raw_data=pd.read_json("https://api.covid19india.org/raw_data"+str(i)+".json")

        c_data_temp=pd.io.json.json_normalize(raw_data['raw_data'])

        c_data = c_data.append(c_data_temp)

        i=i+1

    except:

        print("All data received from source")

        break
#Remove Blank Rows

c_data_clean=c_data[c_data.detectedstate!=""].copy()



#Format announced date into datetime for index

c_data_clean['dateannounced']=pd.to_datetime(c_data_clean['dateannounced'],format='%d/%m/%Y')



#Filter Data upto last day

c_data_clean=c_data_clean[c_data_clean.dateannounced < pd.Timestamp('today').floor('D')].copy()

c_data_clean['detectedstate']=c_data_clean['detectedstate'].astype('category')



#Dummy variable for number of case for each record (to be used in aggregations)

c_data_clean['Total Cases']=np.where(c_data_clean.numcases.isna(),1,pd.to_numeric(c_data_clean.numcases))

#c_data_clean['Total Cases']=1



#Number of cases state wise

daily_cases_state_wise=c_data_clean[c_data_clean.currentstatus=='Hospitalized'].pivot_table(index=['dateannounced'],columns='detectedstate',values='Total Cases',aggfunc=np.sum).fillna(int(0)).sort_index()
#Total cases in India = Sum of all cases state wise

daily_cases_total=pd.DataFrame({'Total':daily_cases_state_wise.sum(axis=1)})

print("Total Cases Reported so far:", daily_cases_total['Total'].sum())
daily_cases_total.plot.line(y='Total',figsize=(16,9),rot=60,title='Daily Cases',grid=True)
rows=int(np.ceil(daily_cases_state_wise.columns.shape[0]/2))

fig, axes = plt.subplots(nrows=rows, ncols=2,figsize=[16,112],sharex=True, sharey=True)

j=0

k=0

for i in daily_cases_state_wise.columns:

    daily_cases_state_wise[i].plot.line(rot=60,title='Daily Cases in '+i,grid=True,ax=axes[j,k])

    j=(j+k)%rows

    k=(k+1)%2

    

plt.show()
rows=int(np.ceil(daily_cases_state_wise.columns.shape[0]/2))

fig, axes = plt.subplots(nrows=rows, ncols=2,figsize=[16,112],sharex=True, sharey=True)

j=0

k=0

for i in daily_cases_state_wise.columns:

    daily_cases_state_wise[i].cumsum().plot.line(rot=60,title='Daily Cases in '+i,grid=True,ax=axes[j,k])

    j=(j+k)%rows

    k=(k+1)%2

    

plt.show()
#Fetch death and recoveries data from covid19india.org (Till 26th April 2019)

deaths_n_recoveries=pd.read_json("https://api.covid19india.org/deaths_recoveries.json")

deaths_n_recoveries_data=pd.io.json.json_normalize(deaths_n_recoveries['deaths_recoveries'])



#Remove Blank Rows

deaths_n_recoveries_clean=deaths_n_recoveries_data[deaths_n_recoveries_data.state!=""].copy()



#Format date into datetime for index

deaths_n_recoveries_clean.date =pd.to_datetime(deaths_n_recoveries_clean['date'],format='%d/%m/%Y')



#Filter Data upto last day

deaths_n_recoveries_clean=deaths_n_recoveries_clean[deaths_n_recoveries_clean.date < pd.Timestamp('today').floor('D')].copy()



#dummy variable for number of cases for each record

deaths_n_recoveries_clean['Total Cases']=1



#Append data post 26th April 2019

dnc_data=c_data_clean[c_data_clean.dateannounced > deaths_n_recoveries_clean.date.max()][['agebracket', 'detectedcity', 'dateannounced', 'detecteddistrict', 'gender', 'nationality', 

                                                                                          'notes', 'patientnumber', 'currentstatus', 'source1', 'source2', 'source3', 'detectedstate', 

                                                                                          'statecode','Total Cases']].rename(columns={'detectedcity':'city', 'dateannounced':'date', 

                                                                                                                                      'detecteddistrict':'district', 

                                                                                                                                      'patientnumber':'patientnumbercouldbemappedlater', 

                                                                                                                                      'currentstatus':'patientstatus', 'detectedstate':'state'})

deaths_n_recoveries_clean=deaths_n_recoveries_clean.append(dnc_data)
import datetime



#State wise number of death and recovered patients

total_cases_sw=deaths_n_recoveries_clean.pivot_table(index=['state'],columns='patientstatus',values='Total Cases',aggfunc=np.sum).fillna(0).sort_values(by='Recovered', ascending=False)

total_cases_sw['Total Cases']=c_data_clean.pivot_table(index='detectedstate',values='Total Cases',aggfunc=np.sum)['Total Cases']

total_cases_sw['Hospitalized']=total_cases_sw['Total Cases']-(total_cases_sw['Recovered']+total_cases_sw['Deceased'])

total_cases_sw['%Deceased']=round(total_cases_sw['Deceased']/total_cases_sw['Total Cases']*100,2)

total_cases_sw['%Recovered']=round(total_cases_sw['Recovered']/total_cases_sw['Total Cases']*100,2)

total_cases_sw['%Hospitalized']=round(total_cases_sw['Hospitalized']/total_cases_sw['Total Cases']*100,2)

total_cases_sw.sort_values(by='%Hospitalized').plot.bar(y=['%Hospitalized','%Recovered','%Deceased'],figsize=[20,10], title='State Wise Recovery % and Death %',stacked=True)
#Dailyy number of death and recovered patients

total_drcases_daily=deaths_n_recoveries_clean.pivot_table(index=['date'],columns='patientstatus',values='Total Cases',aggfunc=np.sum).fillna(0).sort_index()

total_drcases_daily['Total Cases']=c_data_clean.pivot_table(index='dateannounced',values='Total Cases',aggfunc=np.sum)['Total Cases']

total_drcases_daily['%Deceased']=round(total_drcases_daily['Deceased']/total_drcases_daily['Total Cases']*100,2)

total_drcases_daily['%Recovered']=round(total_drcases_daily['Recovered']/total_drcases_daily['Total Cases']*100,2)

total_drcases_daily['Hospitalized']=total_drcases_daily['Total Cases']-total_drcases_daily['Recovered']-total_drcases_daily['Deceased']

total_drcases_daily['R/H']=total_drcases_daily['Recovered']/total_drcases_daily['Hospitalized']

total_drcases_daily.plot.area(y=['Hospitalized','Deceased','Recovered',],figsize=[20,10], title='Daily Hospitalization, Recovery and Death')

#total_drcases_daily[total_drcases_daily['R/H']<20].plot.line(y=['R/H',],figsize=[20,10], title='Daily Recovery % and Death %')
def run_arima_model(df=pd.DataFrame,maxp=0,maxd=0,maxq=0, save=False):

    daily_cases_total_sw=df

    daily_cases_total_sw.plot.line(figsize=(16,9),rot=60,title='Daily Trend',grid=True)

    plt.show()

    ci_l=0-1.96*np.sqrt(1/daily_cases_total_sw.shape[0])

    ci_m=0+1.96*np.sqrt(1/daily_cases_total_sw.shape[0])

    print("95% Confidence Interval for true correlation coefficient: ", ci_l," to ", ci_m)

    arima_models_s=pd.DataFrame(columns=['p','d','q','AIC','Residual Mean','Residual Variance'])

    for d in range (0,maxd):  

        for p in range (0,maxp):

            for q in range (0,maxq):

                try:

                    arima_mod_s = ARIMA(daily_cases_total_sw, (p,d,q)).fit()

                    arima_models_s=arima_models_s.append({'p':p,

                                                    'd':d,

                                                    'q':q,

                                                    'AIC':arima_mod_s.aic,

                                                    'Residual Mean':arima_mod_s.resid.mean(),

                                                    'Residual Variance':arima_mod_s.resid.var()},ignore_index=True)

                    #print("(p,d,q): (",p,d,q,") | AIC:",arima_mod.aic, " | Residual Mean:",arima_mod.resid.mean())

                except:

                    #print(p,d,q," : Not Fit")

                    k=1   

    arima_models_sw=arima_models_s.round(6)[(arima_models_s['Residual Mean']>-0.05) & (arima_models_s['Residual Mean']<0.05)].sort_values(by='Residual Mean')

    #display(arima_models_sw.style)

    #print("Lowest AIC: ",arima_models_sw.AIC.min())

    #print("Median Residual Mean: ",arima_models_sw['Residual Mean'].median())

    amodel_p=arima_models_sw[arima_models_sw.AIC==arima_models_sw.AIC.min()].p

    amodel_q=arima_models_sw[arima_models_sw.AIC==arima_models_sw.AIC.min()].q

    amodel_d=arima_models_sw[arima_models_sw.AIC==arima_models_sw.AIC.min()].d

    

    arima_mod_sw=""

    def a_m(model_p, model_d, model_q, Forecast_Duration):

        #Run Model with p,d,q provided in input

        arima_mod_sw = ARIMA(daily_cases_total_sw, (int(model_p), int(model_d), int(model_q))).fit()

        display(arima_mod_sw.summary())

    

        #Predict Model for the historic data

        print("Predictions from the model: ")

        arima_mod_sw.plot_predict()

        plt.show()

    

        #Show distribution of residuals. It should have 0 mean and distribution of white noise

        print("Distribution of Residuals: ")

        pd.DataFrame(arima_mod_sw.resid).plot(kind='kde', figsize=[9,6],grid=True)

        plt.show()

    

        #Forecast data for durtion provided in input

        sw_arima_model_forecasts = arima_mod_sw.forecast(Forecast_Duration)

        sw_daily_cases_forecast = pd.DataFrame({'Total':sw_arima_model_forecasts[0].transpose(),

                                    'Standard Error':sw_arima_model_forecasts[1].transpose(),

                                    'CI_95_L':sw_arima_model_forecasts[2].transpose()[0],

                                   'CI_95_U':sw_arima_model_forecasts[2].transpose()[1]})

        sw_daily_cases_forecast.index=pd.date_range(start=pd.Timestamp('today'), periods=Forecast_Duration,freq='D')

        sw_daily_cases_forecast[['Total','CI_95_U']].plot.bar(figsize=[16,7], title='Estimated Trend',grid=True)

        plt.show()

        sw_daily_cases_total_forecast=daily_cases_total_sw.append(sw_daily_cases_forecast)

        sw_daily_cases_total_forecast['Total'].plot(title='Daily Trend since beginning',figsize=[16,10],grid=True)

        plt.show()

        sw_daily_cases_total_forecast['Total'].cumsum().plot(title='Estimated Trend(Cummulative Sum) since beginning',figsize=[16,10],grid=True)

        plt.show()

        if save:

            sw_daily_cases_total_forecast.to_csv('Forecast-'+str(pd.Timestamp('today').floor('D'))+'.csv')

    

        return arima_mod_sw

    #display("Provide a suitable ARIMA Model (p,d,q) value from above table where residual mean is as close to 0 and AIC is least")

    #select_p_d_q=interactive(a_m,{'manual': True}, model_p=arima_models_sw.p.unique() ,model_d=arima_models_sw.d.unique(),model_q=arima_models_sw.q.unique(),Forecast_Duration=range(1,46))

    #display(select_p_d_q)

    a_m(amodel_p,amodel_d,amodel_q,45)

    return            
ci_l=0-1.96*np.sqrt(1/daily_cases_total.shape[0])

ci_m=0+1.96*np.sqrt(1/daily_cases_total.shape[0])

print("95% Confidence Interval for true correlation coefficient: ", ci_l," to ", ci_m)
display(fig = sm.graphics.tsa.plot_acf(daily_cases_total.diff().dropna().values.squeeze(), lags=40))
display(fig = sm.graphics.tsa.plot_pacf(daily_cases_total.diff(), lags=40))
#%pip install --upgrade statsmodels

#from statsmodels.tsa.ar_model import AutoReg

#ar_mod = sm.tsa.AutoReg(daily_cases_total,lags=10).fit()

#print(ar_mod.summary())
run_arima_model(pd.DataFrame({'Total':daily_cases_total['Total']}),5,3,5, save=True)
def f(State):

    run_arima_model(df=pd.DataFrame({'Total':daily_cases_state_wise[State]}),maxp=5,maxd=3,maxq=5)



select_state=interactive(f, {'manual': True}, State=c_data_clean.detectedstate.cat.categories)

select_state
run_arima_model(df=pd.DataFrame({'Total':total_drcases_daily['Hospitalized']}),maxp=5,maxd=3,maxq=5)
run_arima_model(df=pd.DataFrame({'Total':total_drcases_daily['Recovered']}),maxp=5,maxd=3,maxq=5)
mortality_rate = round(deaths_n_recoveries_clean[deaths_n_recoveries_clean.patientstatus=="Deceased"]['Total Cases'].sum()/c_data_clean['Total Cases'].sum()*100,2)

recovery_rate = round(deaths_n_recoveries_clean[deaths_n_recoveries_clean.patientstatus=="Recovered"]['Total Cases'].sum()/c_data_clean['Total Cases'].sum()*100,2)



print('Mortality Rate per 100 : ', mortality_rate)

print('Recovery Rate per 100 : ', recovery_rate)
def case_projections(Rate):

    daily_cases_status=c_data_clean.pivot_table(index=['dateannounced'],values='Total Cases',aggfunc=np.sum).fillna(int(0)).sort_index()

    daily_cases_status['Total Cases Projected']=daily_cases_status['Total Cases']*Rate

    daily_cases_status['Recovered']=daily_cases_status['Total Cases Projected'].apply(lambda x: np.ceil(x*recovery_rate/100))

    daily_cases_status['Deceased']=daily_cases_status['Total Cases Projected'].apply(lambda x: np.ceil(x*mortality_rate/100))

    print('Had the actual cases been ',Rate,' times the reported:')

    daily_cases_status.cumsum().plot(y=['Total Cases','Total Cases Projected'],title='Total (Cummulative Sum)',figsize=[16,9],grid=True)

    daily_cases_status.cumsum().plot.area(y=['Total Cases','Recovered','Deceased'],title='Total (Cummulative Sum)',figsize=[16,9],grid=True)

    return ""



select_projection_rate=interactive(case_projections, {'manual': True}, Rate=widgets.IntSlider(min=1, max=10, step=1, value=1))
display(select_projection_rate)
daily_cases_total['Grand Total']=daily_cases_total.cumsum()

daily_cases_total['Days to Double']=0

daily_cases_total['Day by Day Increase Rate']=0

daily_cases_total['Days to Increase by 5K']=0

daily_cases_total['Weekly Average Gowth Rate']=0

start_val=daily_cases_total['Grand Total'][0]

days=0

_5kdays=0

_5k=5000

wagr=0

for i in range(1,daily_cases_total.shape[0]):

  days=days+1

  _5kdays=_5kdays+1

  if(daily_cases_total['Grand Total'][i]>=2*start_val):

    daily_cases_total['Days to Double'][i]=days

    start_val=daily_cases_total['Grand Total'][i]

    days=0

  if i>0:

    num=daily_cases_total['Total'][i]

    denom=daily_cases_total['Total'][i-1]

    daily_cases_total['Day by Day Increase Rate'][i]=round(num/denom*100,2)

    wagr=wagr+daily_cases_total['Day by Day Increase Rate'][i]  

  if daily_cases_total['Grand Total'][i]>_5k:

    _5k=_5k+5000

    daily_cases_total['Days to Increase by 5K'][i]=_5kdays

    _5kdays=0

  if (i+1)%7==0:

    daily_cases_total['Weekly Average Gowth Rate'][i]=wagr/7

    wagr=0

print("Average number of days to double: ", np.ceil(daily_cases_total[daily_cases_total['Days to Double']!=0]['Days to Double'].mean()))
daily_cases_total.plot.bar(x='Grand Total',y='Days to Increase by 5K',figsize=[16,9],grid=True,title='Days taken for cases to increase by 5000')
daily_cases_total[daily_cases_total['Day by Day Increase Rate']!=2200].plot.line(y='Day by Day Increase Rate',figsize=[16,9],grid=True,title='Day by % Increase in Cases')
#run_arima_model(df=pd.DataFrame({'Total':daily_cases_total[daily_cases_total['Day by Day Increase Rate']!=2200]['Day by Day Increase Rate']}),maxp=5,maxd=3,maxq=5)
population_sw = pd.read_csv("https://api.data.gov.in/resource/463d908b-c03b-47d3-b5bb-33d584d27f65?api-key=579b464db66ec23bdd000001dad3517ef4b3400c553d59c13709a27a&format=csv&offset=0&limit=100")
population_sw['Population 2020']=np.ceil(population_sw['Population 2011']*(1+population_sw['Decadal Population Growth Rate - 2001-2011']/100)).astype(int)
population_sw['India/State/Union Territory']=population_sw['India/State/Union Territory'].apply(lambda x: "Andaman and Nicobar Islands" if x=='Andaman & Nicobar Island' else "Jammu and Kashmir" if x=="Jammu & Kashmir" else x)

population_sw['State']=population_sw['India/State/Union Territory'].str.upper()
total_cases_sw['Population']=population_sw.pivot_table(index='India/State/Union Territory',values='Population 2020')['Population 2020']

total_cases_sw['Susceptible']=total_cases_sw['Population']-total_cases_sw['Total Cases']

total_cases_sw['Removed']=total_cases_sw['Deceased']+total_cases_sw['Recovered']

total_cases_sw['Infected']=total_cases_sw['Total Cases']-total_cases_sw['Removed']
c_data_clean_s=c_data_clean[c_data_clean.statuschangedate!=""].copy()

c_data_clean_s['statuschangedate']=pd.to_datetime(c_data_clean_s['statuschangedate'],format='%d/%m/%Y')

c_data_clean_s['statuschangedays']=(c_data_clean_s.statuschangedate-c_data_clean_s.dateannounced).dt.days
total_cases_sw['Avg Recovery Time(Days)']=c_data_clean_s[(c_data_clean_s.currentstatus=='Recovered')|(c_data_clean_s.currentstatus=='Deceased')].pivot_table(index='detectedstate',values='statuschangedays',aggfunc=np.mean)

total_cases_sw['Mean Recovery Frequency(Gamma)']=1/total_cases_sw['Avg Recovery Time(Days)']

total_cases_sw['Date of 1st Case']=c_data_clean_s.pivot_table(index='detectedstate',values=['dateannounced'],aggfunc={np.min})

total_cases_sw['Date of Last Case']=c_data_clean_s.pivot_table(index='detectedstate',values=['dateannounced'],aggfunc={np.max})

total_cases_sw['Covid Reporting Days']=(total_cases_sw['Date of Last Case']-total_cases_sw['Date of 1st Case']).dt.days

#Assuming a uniform geometric progression of R0 ratio from 1st patient, R0 = Total Cases ** (1/Number of Days Elapsed) 

total_cases_sw['R0']=total_cases_sw['Total Cases']**(1/total_cases_sw['Covid Reporting Days'])

total_cases_sw['R0_Calculated']=total_cases_sw['R0'].copy()



#Substituting known RO 

#(Source: (May 4, 2020) - https://www.timesnownews.com/times-facts/article/the-r-naught-factor-why-keeping-r0-low-is-critical-and-what-the-projected-numbers-for-india-say/586947)

total_cases_sw.at['Maharashtra','R0']=2.1

total_cases_sw.at['Gujarat','R0']=1.98

total_cases_sw.at['Madhya Praesh','R0']=1.90

total_cases_sw.at['Rajasthan','R0']=1.88

total_cases_sw.at['Delhi','R0']=1.78

total_cases_sw.at['Tamil Nadu','R0']=1.59

total_cases_sw.at['Kerala','R0']=1.28

total_cases_sw.at['Haryana','R0']=1.25

#(Source: (May 5, 2020) - https://theprint.in/health/slowing-infection-better-recovery-but-mixed-bag-in-states-what-india-gained-from-lockdown/414166/)

total_cases_sw.at['Bihar','R0']=2.0

total_cases_sw.at['Jharkhand','R0']=1.87

total_cases_sw.at['Andhra Pradesh','R0']=1.27



#Considering R0 = Beta(Infection Rate) / Gamma (Mean Recovery Frequency)

total_cases_sw['Infection Rate(Beta)']=total_cases_sw['R0']*total_cases_sw['Mean Recovery Frequency(Gamma)']
total_cases_sw['Population Density']=population_sw.pivot_table(index='India/State/Union Territory',values='Population Density (per sq.km) - 2011')['Population Density (per sq.km) - 2011']
total_cases_sw
#Reference - https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/



def deriv(y, t, N, beta, gamma):

    S, I, R = y

    dSdt = -beta * S * I / N

    dIdt = beta * S * I / N - gamma * I

    dRdt = gamma * I

    return dSdt, dIdt, dRdt
states_with_no_model=""



for i in total_cases_sw.iterrows():

    if(i[1]['Avg Recovery Time(Days)'] >= 1):

        print("========================")

        print("State: ",i[0])

        print("========================")

        gamma=i[1]['Mean Recovery Frequency(Gamma)']

        N=i[1]['Population']

        #beta=i[1]['Infection Rate(Beta)']

        covid_days=int(i[1]['Covid Reporting Days'])

        #covid_days=0

        beta=2.1*gamma

        # A grid of time points (in days)

        t = np.linspace(covid_days, 365 , (365-covid_days))

        #t=(c_data_clean_s.dateannounced.max()-c_data_clean_s.dateannounced.min()).days

        #y0=i[1]['Population']-1, 1, 0

        y0=i[1]['Susceptible'], i[1]['Infected'], i[1]['Removed']



        print("\tPopulation (Estimated in 2020): ",N)

        print("\tAverage Recovery Time : ", i[1]['Avg Recovery Time(Days)'], " days")

        print("\tDays since 1st case : ",covid_days)

        print("\tSIR Parameters - R0 : ", i[1]['R0'], 

              " | Gamma : ", gamma, 

              " | Beta : ", beta)

        print("\tInitial: \n\t\tSusceptible Ppulation (S) : ", y0[0], "\n\t\tInfected : ", y0[1], 

              "\n\t\tRemoved(Recovered+Deceased) : ", y0[2])



        # Integrate the SIR equations over the time grid, t.

        ret = odeint(deriv, y0, t, args=(N, beta, gamma))

        S, I, R = ret.T

        print("\tProjected: \n\t\tSusceptible Populaton (Min) : ", np.ceil(S.min()), 

              "\n\t\tMaximum Infected Persons : ", np.ceil(I.max()), 

              "\n\t\tTime to reach Maximum Infection Case: ",I.argmax(),"days i.e. on",(pd.Timestamp('today').date()+pd.Timedelta(I.argmax(),unit='d')).strftime('%d/%m/%Y'),

              "\n\t\tTime to remove all Infection: ",I[I>1].shape[0],"days i.e. on",(pd.Timestamp('today').date()+pd.Timedelta(I[I>1].shape[0],unit='d')).strftime('%d/%m/%Y'),

              "\n\t\tTotal Recovered+Deceased : ", R.max(),

              "\n\t\tTime to reach Total Recovery: ",R.argmax(),"days i.e. on",(pd.Timestamp('today').date()+pd.Timedelta(R.argmax(),unit='d')).strftime('%d/%m/%Y'))



        # Plot the data on three separate curves for S(t), I(t) and R(t)

        fig = plt.figure(figsize=[16,9])

        ax = fig.add_subplot(111, axisbelow=True)

        ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')

        ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')

        ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered+Deceased')

        ax.set_xlabel('Time /days')

        ax.set_ylabel('Cases (in Scientific Notation)')

        legend = ax.legend()

        legend.get_frame().set_alpha(0.5)

        plt.show()

    else:

        states_with_no_model=states_with_no_model+i[0]+", "



print("SIR Model could not be generated for following States due to unknown Average Recovery Time - ", states_with_no_model)
N=population_sw.iloc[0]['Population 2020']

infected=c_data_clean[c_data_clean.currentstatus=='Hospitalized']['Total Cases'].sum()

removed=c_data_clean[(c_data_clean.currentstatus=='Recovered')|(c_data_clean.currentstatus=='Deceased')]['Total Cases'].sum()

susceptible=N-infected-removed

gamma=1/(c_data_clean_s[(c_data_clean_s.currentstatus=='Recovered')|(c_data_clean_s.currentstatus=='Deceased')|(c_data_clean_s.statuschangedays>0)].statuschangedays.mean())

covid_days=daily_cases_total.shape[0]

#R0=daily_cases_total['Total'].sum()**(1/covid_days)

R0=1.29 # Source (May 5, 2020) - https://theprint.in/health/slowing-infection-better-recovery-but-mixed-bag-in-states-what-india-gained-from-lockdown/414166/ 

beta= R0*gamma



# A grid of time points (in days)

t = np.linspace(covid_days, 500 , (500-covid_days))

#t = np.linspace(0, 1500 , 1500)

#t=(c_data_clean_s.dateannounced.max()-c_data_clean_s.dateannounced.min()).days



#y0=N-1, 1, 0

y0=susceptible, infected, removed



print("Population (Estimated in 2020) : ",N)

print("Average Recovery Time : ",np.ceil(1/gamma), " days")

print("Days cases has been reported : ",covid_days)

print("SIR Parameters - R0 : ", R0, 

              " | Gamma : ", gamma, 

              " | Beta : ", beta)

print("Initial: \n\t\tSusceptible Ppulation (S) : ", y0[0], "\n\t\tInfected : ", y0[1], 

              "\n\t\tRemoved(Recovered+Deceased) : ", y0[2])





# Integrate the SIR equations over the time grid, t.

ret = odeint(deriv, y0, t, args=(N, beta, gamma))

S, I, R = ret.T



print("Projected: \n\t\tSusceptible Populaton (Min) : ", np.ceil(S.min()), 

              "\n\t\tMaximum Infected Persons : ", np.ceil(I.max()), 

              "\n\t\tTime to reach Maximum Infection Case: ",I.argmax(),"days i.e. on",(pd.Timestamp('today').date()+pd.Timedelta(I.argmax(),unit='d')).strftime('%d/%m/%Y'),

              "\n\t\tTime to remove all Infection: ",I[I>1].shape[0],"days i.e. on",(pd.Timestamp('today').date()+pd.Timedelta(I[I>1].shape[0],unit='d')).strftime('%d/%m/%Y'),

              "\n\t\tTotal Recovered+Deceased : ", R.max(),

              "\n\t\tTime to reach Total Recovery: ",R.argmax(),"days i.e. on",(pd.Timestamp('today').date()+pd.Timedelta(R.argmax(),unit='d')).strftime('%d/%m/%Y'))



# Plot the data on three separate curves for S(t), I(t) and R(t)

fig = plt.figure(figsize=[16,9])

ax = fig.add_subplot(211, axisbelow=True,)

ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')

ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')

ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered+Deceased')

ax.set_xlabel('Time /days')

ax.set_ylabel('Cases (in Scientific Notation)')

legend = ax.legend()

legend.get_frame().set_alpha(0.5)

plt.show()
print("Projected Cases by Next 45 Days")

plt.figure(figsize=[16,9])

plt.plot(t[:45], I[:45], 'r', alpha=0.5, lw=2, label='Infected')

plt.show()

print("Estimated cases by 45th Day: ", I[45])
state_test_data_json=pd.read_json("https://api.covid19india.org/state_test_data.json")

state_test_data=pd.io.json.json_normalize(state_test_data_json['states_tested_data'])
state_test_data.columns
state_test_data.updatedon=np.where(state_test_data.updatedon=='02/05/2050',"02/05/2020",state_test_data.updatedon)

state_test_data.updatedon=pd.to_datetime(state_test_data['updatedon'],format='%d/%m/%Y')

state_test_data.totaltested=pd.to_numeric(state_test_data['totaltested']).fillna(0)

state_test_data.negative=pd.to_numeric(state_test_data['negative']).fillna(0)

state_test_data["numcallsstatehelpline"]=pd.to_numeric(state_test_data["numcallsstatehelpline"]).fillna(0)

state_test_data["numicubeds"]=pd.to_numeric(state_test_data["numicubeds"]).fillna(0)

state_test_data["numisolationbeds"]=pd.to_numeric(state_test_data["numisolationbeds"]).fillna(0)

state_test_data["numventilators"]=pd.to_numeric(state_test_data["numventilators"]).fillna(0)

state_test_data["populationsourcecovid19india"]=pd.to_numeric(state_test_data["populationncp2019projection"].str.replace(',','')).fillna(0)

state_test_data["positive"]=pd.to_numeric(state_test_data["positive"]).fillna(0)

state_test_data["testpositivityrate"]=pd.to_numeric(state_test_data["testpositivityrate"].str.replace("%","")).fillna(0)

state_test_data["testspermillion"]=pd.to_numeric(state_test_data["testspermillion"].apply(lambda x: 0 if x=="#REF!" else x)).fillna(0)

state_test_data["testsperthousand"]=pd.to_numeric(state_test_data["testsperthousand"].apply(lambda x: 0 if x=="#REF!" else x)).fillna(0)

state_test_data["totalpeopleinquarantine"]=pd.to_numeric(state_test_data["totalpeoplecurrentlyinquarantine"]).fillna(0)

state_test_data["totalpeoplereleasedfromquarantine"]=pd.to_numeric(state_test_data["totalpeoplereleasedfromquarantine"]).fillna(0)

state_test_data["totaltested"]=pd.to_numeric(state_test_data["totaltested"]).fillna(0)

state_test_data["unconfirmed"]=pd.to_numeric(state_test_data["unconfirmed"]).fillna(0)

state_test_data.sort_values(by=['state','updatedon'],inplace=True)
list_of_states=state_test_data.state.unique()
state_test_data['Daily Tests']=0

state_test_data['Daily Change in tests/million']=0.0

state_test_data['Daily +ve Cases']=0

state_test_data['Daily Calls On Helpline']=0

population=state_test_data['populationsourcecovid19india'][0]

for i in range(1,state_test_data.shape[0]):

    if(state_test_data['state'][i]==state_test_data['state'][i-1]):

        dailyTests=state_test_data['totaltested'][i]-state_test_data['totaltested'][i-1]

        state_test_data['Daily Tests'][i]=dailyTests

        population=state_test_data['populationsourcecovid19india'][i]+population

        if(population>0):

            state_test_data['Daily Change in tests/million'][i]=(dailyTests*1000000)/population

        state_test_data['Daily +ve Cases'][i]=state_test_data['positive'][i]-state_test_data['positive'][i-1]

        state_test_data['Daily Calls On Helpline'][i]=state_test_data['numcallsstatehelpline'][i]-state_test_data['numcallsstatehelpline'][i-1]

    else: 

        population=state_test_data['populationsourcecovid19india'][i]

state_test_data['Daily +ve Cases %']=np.round(state_test_data['Daily +ve Cases']/state_test_data['Daily Tests']*100,2)

state_test_data['Daily Tests']=state_test_data['Daily Tests'].apply(lambda x:0 if x<0 else x)

state_test_data['Daily +ve Cases']=state_test_data['Daily +ve Cases'].apply(lambda x:0 if x<0 else x)

state_test_data['Daily Calls On Helpline']=state_test_data['Daily Calls On Helpline'].apply(lambda x:0 if x<0 else x)
fig, axes = plt.subplots(nrows=int(list_of_states.shape[0]/2), ncols=2,figsize=[16,112])

j=0

k=0

for i in list_of_states:

    state_test_data[state_test_data.state==i].plot.scatter(x='Daily Change in tests/million', y='Daily +ve Cases',title='Daily Cases Vs Delta in Tests/million in '+i,grid=True,ax=axes[j,k])

    j=(j+k)%int(list_of_states.shape[0]/2)

    k=(k+1)%2

    

plt.show()
fig, axes = plt.subplots(nrows=int(list_of_states.shape[0]/2), ncols=2,figsize=[16,80], sharex=True,sharey=False)

j=0

k=0

for i in list_of_states:

    state_test_data[state_test_data.state==i].plot.bar(x='updatedon',y=['Daily Change in tests/million','Daily +ve Cases %'], title='Daily +ve Test % in '+i,grid=True,ax=axes[j,k])

    j=(j+k)%int(list_of_states.shape[0]/2)

    k=(k+1)%2

plt.show()
state_test_data_agg=state_test_data.pivot_table(index='updatedon',

                                                values=['Daily Change in tests/million',

                                                        'Daily Tests',

                                                        'Daily Calls On Helpline'],

                                                aggfunc={'Daily Change in tests/million':np.mean,

                                                         'Daily Tests':np.sum,

                                                        'Daily Calls On Helpline':np.sum})
daily_cases_total['Daily Tests']=state_test_data_agg['Daily Tests'].fillna(0)

daily_cases_total['Daily Change in tests/million']=state_test_data_agg['Daily Change in tests/million'].fillna(0)

daily_cases_total['Daily Calls On Helpline']=state_test_data_agg['Daily Calls On Helpline'].fillna(0)

daily_cases_total['Daily +vity Rate']=np.where(daily_cases_total['Daily Tests']>0,daily_cases_total['Total']/daily_cases_total['Daily Tests'],0)
print('Current +vity rate of tests per 100 = ',round(daily_cases_total['Total'].sum()/daily_cases_total['Daily Tests'].sum()*100,2),"\n",

     'Average of daily +vity rate',daily_cases_total['Daily +vity Rate'].mean(),

     'Median daily +vity rate',daily_cases_total['Daily +vity Rate'].median())

daily_cases_total['Daily +vity Rate'].plot.box(vert=False,figsize=[20,2])

plt.show()
def olsreg_mod_analysis(X,Y,alpha = 0.05,p_coeff = 0.05,p_model = 0.05,rs_threshold=0.1):

    

    mod = sm.OLS(Y, X,missing='drop',hasconst=False)

    res=mod.fit()

    sm.graphics.plot_fit(res, 0)

    plt.show()

    print(res.summary())

    

    print("\n\n-------------------------------Analysis of OLS Regression Results---------------------------------------------------\n")

    

    print("Assumptions: \n Significance (alpha) = ",alpha,

          "\n p-value of Coefficients (For Significance of each Coefficients) = ",p_coeff,

          "\n p-value of Regression Model (For Significance of All Estimators in Model)",p_model)

    

    isModelSignificant = True

    

    print("\n\n====================================")

    print("Are the error terms correlated? Is there autocorrelation in independent variables (Xs) ?")

    print("------------------------------------")

    d=durbin_watson(res.resid)

    print("\tThe Durbin Watson d statistic is: ",d,". d-statistic always lie in the closed intervale [0-4].")

    if((d>=0) & (d<1)):

        print("\tThere probably is greater evidence of positive autocorrelation")

        isModelSignificant = False

    elif((d>=1) & (d<2)):

        print("\tThere probably is slight evidence of positive autocorrelation")

    elif(d==2):

        print("\tThere is no evidence of positive or negtive 1st order autocorrelation")

    elif((d>2) & (d<=3)):

        print("\tThere probably is slight evidence of negtive autocorrelation")

    elif((d>3) & (d<=4)):

        print("\tThere probably is greater evidence of negtive autocorrelation")

        isModelSignificant = False

    

    if not(isModelSignificant):

        print("\n\n=======================================================")

        print("Conclusion: Model is not correct due to autocorrelation")

        print("=======================================================")

        return



    print("\n\n====================================")

    print("Are the coefficients Significant ? Is there a low probability of finding a sample in which the independent variable has no effect on the dependent one i.e. its p-value is less than level of significance assumed for the coefficients ", p_coeff)

    print("------------------------------------")

    total_insignificant_dvars=int(res.pvalues.shape[0])

    for i in range(0,res.pvalues.shape[0]):

        if(res.pvalues[i]<p_coeff):

            print("\t",i+1,". ",res.pvalues.index[i],": Yes. 1 unit change in this variable ", "increases" if res.params[i]<0 else "decreases"," dependent variable by ",round(res.params[i],4)," units.")

            total_insignificant_dvars=total_insignificant_dvars-1

        else:

            print("\t",i+1,". ",res.pvalues.index[i],": No")

    

    if (res.pvalues.shape[0]==total_insignificant_dvars):

        print("\n\n=======================================================")

        print("Note: None of the independent variables are individually significant")

        print("=======================================================")

    

    print("\n\n====================================")

    print("What is the overall significance of the regression model? Is there a low probability of finding a sample in which the independent variables altogether has no effect on the dependent one i.e. its p-value is less than level of significance assumed for the model ", p_model)

    print("------------------------------------")

    if(res.f_pvalue<p_model):

        print("\tYes")

    else:

        print("\tNo")

        isModelSignificant = False

   

    if not(isModelSignificant):

        print("\n\n=======================================================")

        print("Conclusion: Model is not correct as none of the independent variable together has any effect on the dependent variable")

        print("=======================================================")

        return

    

    print("\n\n====================================")    

    print("What is the goodness of fit of the regression model? How much % of total variation in dependent variable (Y) is explaied by all the independent variables (",res.pvalues.shape[0],")?")

    print("------------------------------------")

    print("\t",round(res.rsquared*100,2),"%")

    if res.rsquared<rs_threshold:

        print("\n\n=======================================================")

        print("Conclusion: Model is not fit as the ",res.pvalues.shape[0]," independent variable(s) can explain less than ",rs_threshold*100,"% variation in the dependent variable")

        print("=======================================================")

        return

    

    print("\n\n====================================")

    print("What is the standard error of regression i.e. root mean square error?")

    print("------------------------------------")

    print("\t",res.mse_resid**0.5)

    print("====================================")



    print("\n\n=======================================================")

    print("Conclusion: Model appears to be significant")

    print("=======================================================")

    return
olsreg_mod_analysis(Y=daily_cases_total['Total'], X=daily_cases_total['Daily Tests'])
#population_dw=pd.read_csv("../input/Health-fw-stat-2015_annexure4.csv").dropna()
#population_dw['State']=population_dw['State'].apply(lambda x: "DELHI" if x=='NCT OF DELHI' else 'JAMMU AND KASHMIR' if x=='JAMMU & KASHMIR' else 'ANDAMAN AND NICOBAR ISLANDS' if x=='ANDAMAN & NICOBAR ISLANDS' else x)

#population_dw=population_dw.merge(population_sw[['State','Decadal Population Growth Rate - 2001-2011']],on='State',how='left')
#population_dw['Population 2020']=population_dw['Population - Total - 2011']*(1+population_dw['Decadal Population Growth Rate - 2001-2011']/100.0)