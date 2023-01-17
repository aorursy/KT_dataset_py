import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime, timedelta, date

from scipy.optimize import minimize, root_scalar

from scipy.special import gammainc, gamma

        

    

#This is the 'training' data from the Kaggle project, which I use for all other countries

data_global = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

data_global.columns = ['Id','subregion','state','date','positive','death']

data_global['date'] = pd.to_datetime(data_global['date'],format='%Y-%m-%d')

tref = data_global['date'].iloc[0]

data_global['elapsed'] = (data_global['date'] - tref)/timedelta(days=1)

data_global = data_global.fillna(value='NaN')



#Can also take US data from the covidtracking.com website, which has daily updates.

data_live = pd.read_csv('http://covidtracking.com/api/states/daily.csv')

data_live['date'] = pd.to_datetime(data_live['date'],format='%Y%m%d')

data_live['elapsed'] = (data_live['date'] - tref)/timedelta(days=1)

#These are notes on data quality, and other relevant info for each state

info = pd.read_csv('https://covidtracking.com/api/states/info.csv',index_col=0)

abbreviations = pd.read_csv('/kaggle/input/state-abbreviations/state_list.csv',index_col=0).squeeze().to_dict()



IHME = pd.read_csv('/kaggle/input/ihme-covid19-predictions/Hospitalization_all_locs.csv',index_col=0)

IHME['date'] = pd.to_datetime(IHME['date'],format='%Y-%m-%d')
metric = 'positive'

start_cutoff = 2

start_shift=0

D = 6

TG = 3

Delta = 5

r = 0.05



region = 'US'

subregion = 'New York'



table = data_global.pivot_table(index='elapsed',values=metric,columns=['state','subregion'])

t0 = table.loc[table[region,subregion]>=start_cutoff].index.values[0]+start_shift

t = table.index.values

Nmax = table[region,subregion].values.max()

tmax = t[-1]-t0

logK = np.log(Nmax/gammainc(D,tmax/TG))

t = table.index.values

plt.plot(t-t0,table[region,subregion].values,'o')

if metric == 'positive':

    plt.plot(t-t0,np.exp(logK)*gammainc(D,(t-t0)/TG))

elif metric == 'death':

    plt.plot(t-t0,r*np.exp(logK)*gammainc(D,(t-t0-Delta)/TG))

plt.gca().set_xscale('log')

plt.gca().set_yscale('log')

plt.gca().set_xlabel('Elapsed Time (days)')

plt.gca().set_ylabel(metric)

plt.gca().set_title(region+', '+subregion)

plt.gca().set_ylim((500,Nmax))

plt.gca().set_xlim((5,tmax))

plt.show()
daymin = -20

daymax = 120

p0=5e2

TGlow = 5

TGhigh = 15





region = 'US'

table = data_global.pivot_table(index='elapsed',values='positive',columns=['state','subregion'])

params_table = pd.read_csv('/kaggle/input/covid-19-predictor-using-power-law-model/params.csv')

params_table['State_Province'] = params_table['State_Province'].fillna(value='NaN')

params_table = params_table.set_index(['Country_Region','State_Province'])

params_region = params_table.loc[region].copy()

params_region['t0_abs'] = pd.to_datetime(params_region['t0_abs'],format='%Y-%m-%d')

for item in params_region.index:

    params_region.loc[item,'Delta t'] = timedelta(days=params_region.loc[item,'Delta t'])



dates = np.asarray([datetime.today()+timedelta(days=k) for k in range(daymin,daymax)])

case_predictions_power = pd.DataFrame(index=dates,columns=params_region.reset_index()['State_Province'])

death_predictions_power = pd.DataFrame(index=dates,columns=params_region.reset_index()['State_Province'])

case_predictions_low = pd.DataFrame(index=dates,columns=params_region.reset_index()['State_Province'])

death_predictions_low = pd.DataFrame(index=dates,columns=params_region.reset_index()['State_Province'])

case_predictions_high = pd.DataFrame(index=dates,columns=params_region.reset_index()['State_Province'])

death_predictions_high = pd.DataFrame(index=dates,columns=params_region.reset_index()['State_Province'])

t = case_predictions_low.reset_index()['index']

for subregion in case_predictions_power.keys():

    old_params = params_region.loc[subregion][['t0','t0_abs','C','z','Delta t','r']].values

    t0,t0_abs,C,z,Delta,r = old_params

    

    case_predictions_power[subregion] = C*((t-t0_abs)/timedelta(days=1)).values**z

    death_predictions_power[subregion] = r*C*((t-t0_abs-Delta)/timedelta(days=1)).values**z

    

    tmax = table.index.values[-1]

    if (table[region,subregion]>=p0).sum() > 2:

        tdata = table.loc[table[region,subregion]>=p0].index.values[0]

        tobs = tdata+(tmax-tdata)/2

    else:

        tobs = tmax

    TG = TGlow

    D = z + ((tobs-t0)/TG)

    logK = np.log(C*((tobs-t0)**z)/gammainc(D,(tobs-t0)/TG))

    logR = logK/D

    params_region.loc[subregion]['TG_low'] = np.exp(TG)

    params_region.loc[subregion]['K_low'] = np.exp(logK)

    params_region.loc[subregion]['D_low'] = D

    

    tau = ((t-t0_abs)/timedelta(days=TG)).values

    case_predictions_low[subregion] = np.exp(logK)*gammainc(D,tau)

    tau = ((t-t0_abs-Delta)/timedelta(days=TG)).values

    death_predictions_low[subregion] = r*np.exp(logK)*gammainc(D,tau)

        

    TG = TGhigh

    D = z + ((tobs-t0)/TG)

    logK = np.log(C*((tobs-t0)**z)/gammainc(D,(tobs-t0)/TG))  

    params_region.loc[subregion]['TG_high'] = np.exp(TG)

    params_region.loc[subregion]['K_high'] = np.exp(logK)

    params_region.loc[subregion]['D_high'] = D

    

    tau = ((t-t0_abs)/timedelta(days=TG)).values

    case_predictions_high[subregion] = np.exp(logK)*gammainc(D,tau)

    tau = ((t-t0_abs-Delta)/timedelta(days=TG)).values

    death_predictions_high[subregion] = r*np.exp(logK)*gammainc(D,tau)

    

case_predictions_power = case_predictions_power.fillna(value=0)

death_predictions_power = death_predictions_power.fillna(value=0)

case_predictions_low = case_predictions_low.fillna(value=0)

death_predictions_low = death_predictions_low.fillna(value=0)

case_predictions_high = case_predictions_high.fillna(value=0)

death_predictions_high = death_predictions_high.fillna(value=0)



fig,ax=plt.subplots()

case_predictions_low.sum(axis=1).plot(label='TG = '+str(TGlow),ax=ax)

case_predictions_high.sum(axis=1).plot(label='TG = '+str(TGhigh),ax=ax)

case_predictions_power.sum(axis=1).plot(label='Power law',ax=ax)

table = data_global.pivot_table(index='date',values='positive',columns=['state','subregion'])

table['US'].sum(axis=1).loc[table['US'].sum(axis=1)>1].plot(marker='o',ax=ax,label='Case data')

death_predictions_low.sum(axis=1).plot(label='TG = '+str(TGlow),ax=ax)

death_predictions_high.sum(axis=1).plot(label='TG = '+str(TGhigh),ax=ax)

death_predictions_power.sum(axis=1).plot(label='Power law',ax=ax)

table = data_global.pivot_table(index='date',values='death',columns=['state','subregion'])

table['US'].sum(axis=1).loc[table['US'].sum(axis=1)>0].plot(marker='o',ax=ax,label='Fatality data')

ax.set_yscale('log')

ax.set_title('US Total Predictions')

plt.legend()

plt.show()
case_predictions_low[state].diff()
data_live
state = 'New York'

abbr = abbreviations[state]

IHME_state = IHME.set_index('location').loc[state].set_index('date')



metric='positive'

fig,ax=plt.subplots()

case_predictions_low[state].plot(label='TG = '+str(TGlow),ax=ax)

case_predictions_high[state].plot(label='TG = '+str(TGhigh),ax=ax)

case_predictions_power[state].plot(label='Power law',ax=ax)

table = data_live.pivot_table(index='date',values=metric,columns='state')

table[abbr].plot(marker='o',ax=ax)

ax.set_yscale('log')

ax.set_title(state+' Case Predictions')

plt.legend()

plt.show()



fig,ax=plt.subplots()

case_predictions_low[state].diff().plot(label='TG = '+str(TGlow),ax=ax)

case_predictions_high[state].diff().plot(label='TG = '+str(TGhigh),ax=ax)

case_predictions_power[state].diff().plot(label='Power law',ax=ax)

IHME_state['admis_mean'].plot(label='IHME Hospitalization',ax=ax)

#IHME_state['allbed_mean'].plot(label='IHME Hospitalization',ax=ax)

table = data_live.pivot_table(index='date',values=metric,columns='state')

table[abbr].diff().plot(marker='o',ax=ax)

ax.set_yscale('log')

ax.set_title(state+' New Cases')

ax.set_ylim((1,None))

plt.legend()

plt.show()



metric='death'

fig,ax=plt.subplots()

death_predictions_low[state].plot(label='TG = '+str(TGlow),ax=ax)

death_predictions_high[state].plot(label='TG = '+str(TGhigh),ax=ax)

death_predictions_power[state].plot(label='Power law',ax=ax)

IHME_state['totdea_mean'].plot(label='IHME',ax=ax)

table = data_live.pivot_table(index='date',values=metric,columns='state')

table[abbr].plot(marker='o',ax=ax)

ax.set_yscale('log')

ax.set_title(state+' Fatality Predictions')

ax.set_ylim((1,None))

plt.legend()

plt.show()