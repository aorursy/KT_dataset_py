#load the github repository into workspace

!pip install psypy

!cp -r ../input/foodwall/repository/footprintzero-foodwall-b7d8960/* ./

!cp -r ../input/foodwall-simulation/* ./



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import sqlite3

import os, sys

import math

import seaborn as sb

from design import climate as climate

from pyppfd import solar as light

from fryield import fvcb

from fryield import photosynthesis as ps

from fryield import model as plants

from nutrients import digester as digester

from hvac import model as hvac
hourly = climate.update()['hourly']
f, axes = plt.subplots(2,2,figsize=(20,20))

fields = list(hourly.keys())

for i in range(4):

    sb.lineplot(x=range(24),y=hourly[fields[i]],ax=axes.flat[i]).set_title(fields[i])

plt.tight_layout()
#fvcb

#show the dependency of electron transport limited assimulation on temperature and irradiance

tmin = 0 ; tmax = 50 ; npts = 50

T_C = [tmin+x/(npts-1)*(tmax-tmin) for x in range(npts)]

I = [0,100,250,500,1000]

pCO2 = [55,110,165,230,330]



#show the dependency of net assimilation on CO2 partial pressure at 700 umol_m2_s (Figure 8 FvCB 1980)

A_cd = [[fvcb.net_assimilation_rate(t,700,c) for t in T_C] for c in pCO2]



#show the dependency of net assimilation on photon flux umol_m2_s at 230 ubar CO2 (Figure 9 FvCB 1980)

A_i = [[fvcb.net_assimilation_rate(t,i,230) for t in T_C] for i in I]
f, axes = plt.subplots(1,2,figsize=(20,10))

i = 0

for a in A_cd:

    sb.lineplot(x=T_C,y=a,label=pCO2[i],ax=axes.flat[0]).set_title('CO2 umol/m2/s vs T, CO2 ubar')

    axes.flat[0].legend(loc='upper_right')

    i = i +1

i = 0

for a in A_i:

    sb.lineplot(x=T_C,y=a,label=I[i],ax=axes.flat[1]).set_title('CO2 umol/m2/s vs T, PPFD')

    axes.flat[1].legend(loc='upper_right')

    i = i +1

plt.tight_layout()
energy_photon = 2.1 #umol/J

RH = [0.55,0.65,0.75,0.85]

cases = [(t,rh,i) for t in T_C for rh in RH for i in I]

ps.setup()

A_RH = [[ps.net_assimilation_rate(t,rh,800,800*energy_photon) for t in T_C] for rh in RH]
i = 0

for a in A_RH:

    sb.lineplot(x=T_C,y=a,label=RH[i]).set_title('CO2 umol/m2/s vs T, RH%')

    plt.legend(loc='upper_right')

    i = i +1
hours = range(24)

T = [ps.plant_transpiration_rate(hourly['irradiance_W_m2'][h]) for h in hours]

A_hr = [ps.net_assimilation_rate(hourly['T_C'][h],hourly['RH'][h],

        hourly['irradiance_W_m2'][h],hourly['ppfd_umol_m2_s'][h]) for h in hours]
f, axes = plt.subplots(1,2,figsize=(20,10))

sb.lineplot(x=hours,y=T,ax=axes.flat[0]).set_title('transpiration mmol/m2/s')

sb.lineplot(x=hours,y=A_hr,ax=axes.flat[1]).set_title('assimilation CO2 umol/m2/s')

plt.tight_layout()
plant_spacing_cm = 65

A_m2 = math.pi*(plant_spacing_cm/100)**2

fm0_g = 10

A_daily = 0.3 # molCO2_m2_d

days = range(80)

fw = [ps.fw_at_t(d,fm0_g=fm0_g,ps_max_molCO2_m2_d=A_daily,A_m2=A_m2) for d in days]
sb.lineplot(x=days,y=fw).set_title('plant fresh weight g vs days after sow')
SQL_DB = 'simruns_03.db'

con = sqlite3.connect(SQL_DB)

cases = pd.read_sql('select * from cases',con)
parameters = pd.read_sql('select * from parameters',con)
s_j = 0 ; c_p = 0

seriesid = []



#create a new label 'seriesid'

for c in cases.caseid:

    if c<c_p:

        s_j = s_j + 1

    seriesid.append(s_j)

    c_p = c



#update the caseid field to combine the old caseid with the new series id

cases['seriesid'] = seriesid

cases['uniqueid'] = cases['seriesid']*10000 + cases['caseid']

del cases['caseid']

cases.rename(columns={'uniqueid':'caseid'},inplace=True)
cases.tail()
parameters[(parameters.group=='plants') & (parameters.confidence<0.8) ]
pldata = pd.pivot_table(cases[cases['seriesid']==1], index='caseid',values='value',

                        columns='parameter', aggfunc='mean')

pldata['tco'] = pldata['capex'] + pldata['opex']/0.03

pldata.head()
f, axes = plt.subplots(4,2,figsize=(20,20))

DOF = [x for x in pldata.columns if 'plants' in x]

for i in range(len(DOF)):

    sb.jointplot(x=DOF[i],y='revenue',data=pldata,kind='kde',ax=axes.flat[i])

    axes.flat[i].title.set_text(DOF[i])

plt.tight_layout()
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score



def run_ml(pvt, DOF,target='revenue'):

    std = pvt[[x for x in pvt.columns if x in (DOF + [target])]].copy()

    target_avg = sum(std[target])/len(std[target])

    std['target'] = std[target] > target_avg

    y = std['target'].astype(int)

    del std[target], std['target']

    xparams = std.copy()

    train_X, test_X, train_y, test_y = train_test_split(xparams, y, random_state=1)

    parameters = {

        'learning_rate': [.1, .5, 1],

        'n_estimators': [10, 200, 500]

    }

    clf = GridSearchCV(AdaBoostClassifier(), parameters, cv=10, n_jobs=-1, scoring='roc_auc')

    clf.fit(train_X, train_y)

    b_estimator = clf.best_estimator_

    predictions = clf.predict(test_X)

    score = roc_auc_score(test_y,predictions)

    importances = b_estimator.feature_importances_

    ftable = pd.DataFrame()

    ftable['params']=xparams.columns

    ftable['importance']=importances

    ftable = ftable.sort_values('importance',ascending=False)

    return ftable,score
ftable, score = run_ml(pldata,DOF,'revenue')

score
ftable
parameters[(parameters.group=='nutrients') & (parameters.confidence<0.8) ]
nudata = pd.pivot_table(cases[cases['seriesid']==3], index='caseid',values='value',

                        columns='parameter', aggfunc='mean')

nudata['kpi_tco'] = nudata['kpi_capex'] + (nudata['kpi_opex']-nudata['kpi_revenue'])/0.03

nudata.head()
DOF = [x for x in nudata.columns if 'nutrients' in x]

ftable, score = run_ml(nudata,DOF,'kpi_simple_return')

score
ftable
parameters[(parameters.group=='tower') & (parameters.confidence<0.8) ]
twdata = pd.pivot_table(cases[cases['seriesid']==5], index='caseid',values='value',

                        columns='parameter', aggfunc='mean')

twdata['kpi_tco'] = twdata['kpi_capex'] + (twdata['kpi_opex']-twdata['kpi_revenue'])/0.03

twdata.head()
f, axes = plt.subplots(3,2,figsize=(20,20))

DOF = [x for x in twdata.columns if 'tower_' in x]

for i in range(len(DOF)):

    sb.jointplot(x=DOF[i],y='kpi_simple_return',data=twdata,kind='kde',ax=axes.flat[i])

    axes.flat[i].title.set_text(DOF[i])

plt.tight_layout()
parameters[(parameters.group=='climate') & (parameters.confidence<0.8) ]
cldata = pd.pivot_table(cases[cases['seriesid']==6], index='caseid',values='value',

                        columns='parameter', aggfunc='mean')

cldata['kpi_tco'] = cldata['kpi_capex'] + (cldata['kpi_opex']-cldata['kpi_revenue'])/0.03

cldata.head()
DOF = [x for x in cldata.columns if 'climate_' in x]

ftable, score = run_ml(cldata,DOF,'kpi_simple_return')

score
ftable
capex_fields = [x for x in nudata.columns if (('capex_' in x) and not ('kpi_' in x)) ]

opex_fields = [x for x in nudata.columns if (('opex_' in x) and not ('kpi_' in x)) ]



cpx_avg = nudata[capex_fields].mean()

opx_avg = nudata[opex_fields].mean()



cpx = pd.DataFrame({'group':[x.replace('capex_','') for x in cpx_avg.index],'values':cpx_avg})

opx = pd.DataFrame({'group':[x.replace('opex_','') for x in opx_avg.index],'values':opx_avg})



cpx.sort_values('values',inplace=True,ascending=False)

opx.sort_values('values',inplace=True,ascending=False)
f, ax = plt.subplots(1,2,figsize=(10,5))

sb.barplot(x='values',y='group',data=cpx,ax=ax.flat[0])

sb.barplot(x='values',y='group',data=opx,ax=ax.flat[1])

plt.tight_layout()
cpx