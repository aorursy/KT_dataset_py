#Exponential curve fitting exercise for basic prediction of COVID19 - week2 kaggle

#Try to learn from countries that passed the infection peak for duration

#For fatalities rate use country demographics to predict by and check for feature importance - used xgboost

#Analysis is based on daily new cases curve, normalized by population and smoothed before extracting parameters



import os

import pandas as pd

import numpy as np

from scipy.interpolate import Rbf

from scipy.optimize import curve_fit

from datetime import timedelta

from sklearn.metrics import mean_squared_log_error

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

import xgboost



import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('ggplot')



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
EVAL_MODE = False #set to false for full analysis of train data without splitting for eval data
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

df_test  = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')

df_sub   = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')
df_train['Date'] = pd.to_datetime(df_train['Date'])

df_test['Date']  = pd.to_datetime(df_test['Date'])
df_train.tail(3)
#prevent NA rows dissapear in grouby operations

df_train['Province_State'].fillna(' ',inplace=True)

df_test['Province_State'].fillna(' ',inplace=True)
df_test.head(3)
df_sub.head(3)
#calculate daily additions to confirmed cases and fatalities in absolute values and percentages

by_ctry_prov = df_train.groupby(['Country_Region','Province_State'])[['ConfirmedCases','Fatalities']]

df_train[['NewCases','NewFatalities']]= by_ctry_prov.transform(lambda x: x.diff().fillna(0))

df_train[['NewCasesPct','NewFatalitiesPct']]= by_ctry_prov.transform(lambda x: x.pct_change().fillna(0))
df_train.sort_values('NewCases',ascending = False).head()
#check for inconsistencies in daily new cases, cumulative count should only increase or remain equal

df_train[df_train.NewCases < 0].sort_values('NewCases')
#check for inconsistencies in daily new fatalities, cumulative count should only increase or remain equal

df_train[df_train.NewFatalities < 0].sort_values('NewFatalities')
#more deaths than confirmed cases

df_train[df_train.Fatalities > df_train.ConfirmedCases]
#more than 40% increase in ConfirmedCases with at least 1000 new cases - Hubei 13 Feb example

df_train[(df_train.NewCasesPct > 0.4) & (df_train.NewCases > 1000)]
#more than 80% increase in Fatalities with at least 50 new cases

df_train[(df_train.NewFatalitiesPct > 0.8) & (df_train.NewFatalities > 50)]
def lin_interpolate(indx1,indx2,col,df):

    start_val = df.loc[indx1,col]

    end_val = df.loc[indx2,col]

    i = indx1+1

    while i < indx2:

        df.loc[i,col] = start_val + int((i-indx1)/(indx2-indx1)*(end_val-start_val))

        i+=1
#Shandong 21 Feb

df_train[(df_train['Country_Region']=='China') & 

         (df_train['Province_State']=='Shandong') &

         (df_train.Date > '2020-02-18')].head(5)
#Hubei 13 Feb

df_train[(df_train['Country_Region']=='China') & 

         (df_train['Province_State']=='Hubei') &

         (df_train.Date > '2020-02-8')].head(8)
#fix China Hubei 12-14 Feb reporting and Shandong 21 Feb

#need to check indexes don't change every new training set

#decided to fix on new cases only

lin_interpolate(5178,5183,'NewCases',df_train) #Hubei 12-14 Feb

lin_interpolate(6035,6037,'NewCases',df_train) #Shandong 21 Feb
all_ctry_lst = list(df_train.set_index(['Country_Region','Province_State']).index.unique())

len(all_ctry_lst)
df_train['Country_Region'].nunique()
c = df_train[df_train['Province_State'] != 'UniqueProvince'].groupby('Country_Region')['Province_State'].unique()

[(x,list(c.loc[x])) for x in c.index]
df_train[df_train['Country_Region']=='China']['Province_State'].unique()
EVAL_DATE       = pd.to_datetime('3/26/2020')

if EVAL_MODE:

    TRAIN_LAST_DATE = EVAL_DATE

else:

    TRAIN_LAST_DATE = df_train.Date.max()

PRED_LAST_DATE  = df_test.Date.max()
df_train.drop(['Id','NewCasesPct','NewFatalitiesPct'],axis=1,inplace=True)
#remove data for evaluation

df_eval  = df_train[df_train['Date'] >  EVAL_DATE]

df_train = df_train[df_train['Date'] <= TRAIN_LAST_DATE]
#add ForecastId column

df_eval = df_eval.merge(df_test,on=['Date', 'Country_Region', 'Province_State'],how='left',validate='1:1')

df_eval
df_corona = df_train[['Province_State','Country_Region','Date','ConfirmedCases','Fatalities','NewCases']].copy()
#population per province dataset from kaggle

df_pop = pd.read_csv('/kaggle/input/covid19-global-forecasting-locations-population/locations_population.csv')
df_pop['Province.State'].fillna(' ',inplace=True)

df_pop.drop(labels=['Provenance'],axis=1,inplace=True)

df_pop
#manually fix for locations_population.csv file

df_pop = df_pop.append({'Province.State':' ','Country.Region':'Botswana','Population':2254068},ignore_index=True)

df_pop = df_pop.append({'Province.State':' ','Country.Region':'Burma','Population':53582855 },ignore_index=True)

df_pop = df_pop.append({'Province.State':' ','Country.Region':'Burundi','Population':11745876 },ignore_index=True)

df_pop = df_pop.append({'Province.State':' ','Country.Region':'Sierra Leone','Population':7092113 },ignore_index=True)

df_pop = df_pop.append({'Province.State':' ','Country.Region':'West Bank and Gaza','Population':4550000 },ignore_index=True)

df_pop = df_pop.append({'Province.State':' ','Country.Region':'MS Zaandam','Population':1800 },ignore_index=True)
a = set(df_corona['Country_Region'])
b = set(df_pop['Country.Region'])
a.difference(b) #issues with country names between the files
df_corona = df_corona.merge(df_pop,"left",left_on=['Country_Region','Province_State'],

                            right_on=['Country.Region','Province.State'],indicator=True,validate='many_to_one')
df_corona._merge.value_counts()
#df_corona[df_corona._merge == 'left_only']
df_corona['new_cases_per_pop']=df_corona['NewCases']/df_corona['Population']*100000

df_corona.drop(['_merge'],axis=1,inplace=True)

df_corona
#check distribution of known total cases per country/province

df_corona.groupby(['Country_Region','Province_State']).ConfirmedCases.max().describe()
len(df_corona.groupby(['Country_Region','Province_State'])) #number of countries/provinces in dataset
#df_corona.Country_Region.unique()
def filter_out(x):

    f1 = x['ConfirmedCases'].max() > 30      #at least 30 sick cases

    f2 = x['Population'].min() > 2000000     #pop more than 2M, small countries with distorted percent of confirmed cases

    f3 = x['new_cases_per_pop'].max() > 0.1  #at least 0.1 cases per 100K pop

    return f1 & f2 & f3
#create reduced filtered dataset for analysis

df_analysis = df_corona.groupby(['Province_State','Country_Region']).filter(filter_out).copy()

len(df_analysis.groupby(['Country_Region','Province_State'])) #number of countries in dataset
#df_analysis['Country_Region'].unique()
#for each country/province, filter out all dates before there are at least 5 new daily cases

df_filt = pd.DataFrame(columns=df_analysis.columns)

for grp in df_analysis.groupby(['Province_State','Country_Region']):

    start_indx = grp[1].loc[grp[1].NewCases >= 5,:].index

    if len(start_indx) > 0:

        df_filt = pd.concat([df_filt,grp[1].loc[start_indx[0]:,:]])
#remove countries with less than 5 days of meaningful new daily cases

df_filt = df_filt.groupby(['Province_State','Country_Region']).filter(lambda x: len(x.Date) > 4)
len(df_filt.groupby(['Country_Region','Province_State'])) #final number of countries/provinces after filtering
#df_filt['Country_Region'].unique()
ctry_prov_lst = list(df_filt.set_index(['Country_Region','Province_State']).index.unique())
def country_prov_select(cntry,prov,df):

    df_sel = df[(df['Country_Region'] == cntry) & (df['Province_State'] == prov)].reset_index(drop=True)

    #consistency check, fails for China after fixing 13/2 reports

    #assert(all(df_sel.ConfirmedCases == df_sel.NewCases.cumsum()))

    return df_sel.fillna(0)
fig, ax = plt.subplots()

df_filt.groupby(['Province_State','Country_Region']).plot(y='ConfirmedCases',use_index=False,ax=ax,figsize = (10,6), marker='.',

                                 legend=False,title='Confirmed cases for each country and region')

ax.set_xlabel('Days from measurement start')

ax.set_ylabel('Confirmed cases')

plt.show()
#show maximum daily new cases as a percentage of population, "wall of shame" for country measures :-)

df_filt.groupby(['Province_State','Country_Region']).new_cases_per_pop.max().sort_values(ascending=False)[:30]
df_filt = df_filt.set_index(['Province_State','Country_Region'])

df_filt['max_ratio'] = df_filt.groupby(['Province_State','Country_Region']).new_cases_per_pop.max()

df_filt.reset_index(inplace=True)

df_filt.sort_values(by=['max_ratio'],ascending=False,inplace=True,kind='mergesort') #mergesort is stable
#show graphs of daily new cases in batches of 10 to be able to see difference, sorted by severity

i = 0

for ctry,grp in df_filt.groupby(['Province_State','Country_Region'],sort=False):

    if i%10==0: #new plot for group

        fig, ax = plt.subplots()

        ax.set_xlabel('Days from measurement start')

        ax.set_ylabel('New daily cases per 100K')

    grp.plot(y='new_cases_per_pop',use_index=False, ax=ax, figsize=(10,6), label=ctry[1]+" "+ctry[0], marker='.',

                                legend=True,title='New daily cases per 100K for each country/region')

    i +=1



plt.show()
#candidate functions definition for curve fitting

def exp_func(x,a,b):

    return a*np.exp(b*x)



def poly_func(x,a,b):

    return a*(x**b)



def sigmoid_func(x,a,b,c):

    return a/(1+np.exp(-b*(x-c)))
#loop over all countries/provinces

#check curve fit for exponential and polynomial function for countries which passed the peak of the infection

#uses new cases per day per 100K of the country population as time series values for radial basis function interpolation

#saves peak duration list for countries that seem to be after infection peak and the curve fitting params

peak_duration = {}

param_list = []

for (ctry,prov) in ctry_prov_lst:

    fig, ax = plt.subplots()

    df_country = country_prov_select(ctry,prov,df_filt)

    plt.plot(df_country.new_cases_per_pop,marker='.')

    rbf=Rbf(df_country.index,df_country.new_cases_per_pop,smooth=5)(df_country.index)

    plt.plot(rbf,marker='+')

    ax.set_title(ctry+" "+prov)

    plt.show()

    peak_indx = np.argmax(rbf)

    peak_date = df_country.Date.iloc[peak_indx]

    #are we at least 5 days after peak of new cases and peak detection 5 days after start?

    peak_reached = (len(df_country) > peak_indx+4) & (peak_indx > 5)

    if peak_reached:

        print("Peak reached for {0} {1} in {2} days on {3}".format(ctry,prov,peak_indx,peak_date))

        peak_duration[(ctry,prov)]=peak_indx

        #exponential fit check

        params1, cov_params1 = curve_fit(exp_func,np.arange(peak_indx),rbf[:peak_indx],maxfev=10000,p0=[0.1,0.4])

        print("\nExponential fit parameters: a:{0:.4f} b:{1:.2f}".format(params1[0],params1[1]))

        print("Covariance matrix of parameters:")

        print(cov_params1)

        param_list.append(params1)

        #polynomial fit check

        #params2, cov_params2 = curve_fit(poly_func,np.arange(peak_indx),rbf[:peak_indx],maxfev=10000,p0=[0.01,0.5])

        #print("\nPolynomial fit parameters: a:{0:.4f} b:{1:.2f}".format(params2[0],params2[1]))

        #print("Covariance matrix of parameters:")

        #print(cov_params2)

        #sigmoid fit check

        #params3, cov_params3 = curve_fit(sigmoid_func,np.arange(peak_indx)+0.1,rbf[:peak_indx],maxfev=10000)

        #print("\nSigmoid fit parameters: a:{0:.2f} b:{1:.2f} c:{2:.2f}".format(params3[0],params3[1],params3[2]))

        #print("Covariance matrix of parameters:")

        #print(cov_params3) 

        #rough check for goodness of fit

        #if np.diag(cov_params1).sum() > np.diag(cov_params2).sum():

        #    print("\nPolynomial fit seems better.")

        #else:

        #    print("\nExponential fit seems better.")

    else:

        print("Peak not yet reached for {0} {1}".format(ctry,prov))
#df_filt[(df_filt['Country_Region']=='Kuwait') & (df_filt['Province_State']==' ')]
peak_duration
peaked_countries = list(peak_duration.keys())
param_list #curve fitting result parameters
avg_param = np.mean(param_list,axis=0)

avg_param
#taking the 80% quantile as average since we have a bias from the first countries that recovered

avg_duration = round(np.quantile(list(peak_duration.values()),0.8))

avg_duration
max_duration = max(list(peak_duration.values()))

max_duration
#rough predict based on average/max duration to peak using exponential fitting parameters

FLAT = 5    #guess for number of days remaining at peak TBD optimize based on fitting to peaked countries

MAX_INC = 5 #arbitrary additional days prediction till peak for countries that lasted longer than history



def predict_new_cases(ctry,prov,df):

    df_country = country_prov_select(ctry,prov,df)

    pop_ctry = df_pop[(df_pop['Country.Region']==ctry) & (df_pop['Province.State']==prov)].Population

    pop_factor = int(pop_ctry/100000)

    ctry_dates = df_country[['Date']].copy()

    ctry_confirmed = df_country[['ConfirmedCases']].copy()

    

    if len(df_country) > 1:

        rbf=Rbf(df_country.index,df_country.new_cases_per_pop,smooth=5)(df_country.index).clip(0)

    else:

        rbf=np.array(df_country.new_cases_per_pop).clip(0)

    peak_indx = np.argmax(rbf)

    peak_date = df_country.Date.iloc[peak_indx]

    #are we at least 5 days after peak of new cases and peak detection 5 days after start?

    peak_reached = (len(df_country) > peak_indx+4) & (peak_indx > 5)

    

    #exponential fitting TBD take parameters fitted from other peaked countries with similar characteristics

    if (peak_indx > 5) and (len(df_country) > 9):

        params, _ = curve_fit(exp_func,np.arange(peak_indx),rbf[:peak_indx],maxfev=10000,p0=[0.1,0.3])

    else:

        params = avg_param

    

    if peak_reached:

        print("Peak reached for {0} {1} in {2} days on {3}".format(ctry,prov,peak_indx,peak_date))

        dur = peak_indx

        if len(df_country) < dur*2+1+FLAT:

            #add predictions to decrease

            for j in range(len(df_country),dur*2+1+FLAT):

                pred = exp_func(2*dur-j+FLAT,params[0],params[1])

                rbf = np.append(rbf,pred)

                ctry_dates = ctry_dates.append(ctry_dates.iloc[-1]+timedelta(1))

                ctry_confirmed = ctry_confirmed.append(ctry_confirmed.iloc[-1]+pred*pop_factor)

    else:

        print("Peak not yet reached for {0} {1}".format(ctry,prov))

        #naive predict duration till new cases per day peak

        if len(df_country) < int(np.ceil(avg_duration)):

            dur = int(np.ceil(avg_duration))

        elif len(df_country) < max_duration:

            dur = max_duration

        else:

            dur = len(df_country)+MAX_INC

            print("Past due peaking for {0} {1}, arbitrary {2} days prediction".format(ctry,prov,MAX_INC))

        if (np.mean(df_country.NewCases[-5:])>5) and (df_country.ConfirmedCases.iloc[-1]>30):

            #add predictions till peak based on estimated function

            for j in range(len(df_country),dur+1):

                pred = exp_func(j,params[0],params[1])

                rbf = np.append(rbf, pred)

                ctry_dates = ctry_dates.append(ctry_dates.iloc[-1]+timedelta(1))

                ctry_confirmed = ctry_confirmed.append(ctry_confirmed.iloc[-1]+pred*pop_factor)

            #add predictions for remaining flat TBD curve fit to peaked countries

            rbf = np.append(rbf,np.ones(FLAT)*rbf[-1])

            for i in np.arange(FLAT):

                ctry_dates = ctry_dates.append(ctry_dates.iloc[-1]+timedelta(1))

                ctry_confirmed = ctry_confirmed.append(ctry_confirmed.iloc[-1]+pred*pop_factor)

            #add predictions to decrease

            for j in range(dur+FLAT,dur*2+1+FLAT):

                pred = exp_func(2*dur-j+FLAT,params[0],params[1])

                rbf = np.append(rbf,pred)

                ctry_dates = ctry_dates.append(ctry_dates.iloc[-1]+timedelta(1))

                ctry_confirmed = ctry_confirmed.append(ctry_confirmed.iloc[-1]+pred*pop_factor)

    

    while ctry_dates['Date'].iloc[-1] < PRED_LAST_DATE:

        rbf = np.append(rbf,np.mean(rbf[-3:]))

        ctry_dates = ctry_dates.append(ctry_dates.iloc[-1]+timedelta(1))

        ctry_confirmed = ctry_confirmed.append(ctry_confirmed.iloc[-1]+rbf[-1]*pop_factor)

    

    cases = rbf * pop_factor

    fig, ax = plt.subplots()

    plt.plot(np.arange(len(df_country)),cases[:len(df_country)],marker='+')

    plt.plot(np.arange(len(df_country),len(cases)),cases[len(df_country):],marker='.')

    ax.set_title(ctry+" "+prov+" predicted new daily cases")

    plt.show()

    print("Total predicted infected for {0} {1}: {2:d}\n".format(ctry,prov,int(ctry_confirmed.iloc[-1])))

    cases_df = pd.DataFrame(cases.astype(int),columns=['NewCases'])

    ctry_df = pd.DataFrame([ctry for i in range(len(cases))],columns=['Country/Region'])

    ctry_confirmed = ctry_confirmed.reset_index(drop=True).astype(int)

    pop_tmp = pd.DataFrame([pop_ctry.values[0] for i in range(len(cases))], columns=['Population'])

    return(pd.concat([ctry_df,ctry_dates.reset_index(drop=True),ctry_confirmed,cases_df,pop_tmp],axis=1))
#country_prov_select('Indonesia',' ',df_corona)#.ConfirmedCases.iloc[-1]
ctry = 'Israel'

prov = ' '

pred = predict_new_cases(ctry,prov,df_corona)
df_last_day_train = df_train.groupby(['Country_Region','Province_State']).last().reset_index().copy()
BACK_HISTORY = 8

def add_history(df,lag = BACK_HISTORY):

    for i in range(BACK_HISTORY):

        colname = "Confirmed_d" + str(i+1) 

        df[colname] = df.ConfirmedCases.shift(i+1)
df_history = df_corona[['Province_State','Country_Region','ConfirmedCases','Fatalities','Population']].copy()

df_history
add_history(df_history)

df_history = df_history.groupby(['Country_Region','Province_State'],

                                as_index=False).apply(lambda x: x[BACK_HISTORY:]).reset_index(drop=True)

df_history
df_history.corr()
#country descriptive information from kaggle database

df_ctryinfo = pd.read_csv('/kaggle/input/countryinfo/covid19countryinfo.csv')
df_ctryinfo = df_ctryinfo.iloc[:193,:29]

df_ctryinfo = df_ctryinfo[df_ctryinfo.region.isnull()] #keep only main countries

df_ctryinfo
df_ctryinfo = df_ctryinfo[['country','density','medianage','urbanpop','hospibed','smokers',

                           'sexratio','lung','femalelung','malelung']]
df_hist = df_history.merge(df_ctryinfo,left_on='Country_Region',right_on='country',how='left')

df_hist.drop('country',axis=1,inplace=True)

df_hist
x = df_hist.iloc[:,2:].drop('Fatalities',axis=1)

y = df_hist[['Fatalities']]
params = [{'n_estimators':[100,1000],

          'max_depth':[3,4], 'learning_rate':[0.01,0.1]}]
model = xgboost.XGBRegressor(objective = 'reg:squaredlogerror')
grid_model = GridSearchCV(model, params, cv=3, verbose=2, n_jobs=-1)

grid_model.fit(x,y)
grid_model.best_params_
opt_params = grid_model.best_params_
#fat_model = grid_model.best_estimator_

best_model = xgboost.XGBRegressor(objective = 'reg:squaredlogerror',learning_rate = opt_params['learning_rate'],

                                 max_depth = opt_params['max_depth'], n_estimators = opt_params['n_estimators'])

fat_model = best_model.fit(x,y)
pd.DataFrame(fat_model.feature_importances_,index=x.columns,columns=['Importance']).sort_values('Importance',ascending=False)
model_preds = np.maximum(0,fat_model.predict(x))

mean_squared_log_error(model_preds,y)
df_last_day_train = df_train.groupby(['Country_Region','Province_State']).last().reset_index().copy()
#loop over countries to initialize prediction with last known values

df_sub.set_index('ForecastId',inplace=True)

for i in df_last_day_train.index:

    predc = df_last_day_train.loc[i,'ConfirmedCases']

    predf = df_last_day_train.loc[i,'Fatalities']

    ctry = df_last_day_train.loc[i,'Country_Region']

    prvn = df_last_day_train.loc[i,'Province_State']

    test_id = df_test[(df_test['Country_Region'] == ctry) & (df_test['Province_State'] == prvn)]['ForecastId']

    df_sub.loc[test_id,'ConfirmedCases']=predc

    df_sub.loc[test_id,'Fatalities']=predf

df_sub.reset_index(inplace=True)
#df_sub.reset_index(inplace=True)
#loop over countries in ctry_lst which we predict exponentially

df_sub.set_index('ForecastId',inplace=True)

for (ctry,prov) in ctry_prov_lst:

    pred = predict_new_cases(ctry,prov,df_filt)

    add_history(pred)

    pred = pred[(pred['Date'] > TRAIN_LAST_DATE) & (pred['Date'] <= PRED_LAST_DATE)]

    pred = pred.merge(df_ctryinfo,left_on='Country/Region',right_on='country',how='left')

    pred.drop('country',axis=1,inplace=True)

    fat_pred = fat_model.predict(pred.iloc[:,2:].drop('NewCases',axis=1))

    fat_pred = np.round(np.maximum(fat_pred,0))

    test_id = df_test[(df_test['Country_Region'] == ctry) & 

                      (df_test['Province_State'] == prov) &

                      (df_test['Date'] > TRAIN_LAST_DATE)]['ForecastId']

    df_sub.loc[test_id,'ConfirmedCases']=pred['ConfirmedCases'].values

    df_sub.loc[test_id,'Fatalities']=fat_pred

df_sub.reset_index(inplace=True)
#create df of countries not in exponential group and filter out all dates before there are at least 5 new daily cases

df_corona_noexp = pd.DataFrame(columns=df_corona.columns)

c = [c for (c,p) in ctry_prov_lst]

p = [p for (c,p) in ctry_prov_lst]

#countries that were not predicted by exponential fit

df_noexp = df_corona[~(df_corona['Country_Region'].isin(c)) | ~(df_corona['Province_State'].isin(p))]

for grp in df_noexp.groupby(['Province_State','Country_Region']):

    start_indx = grp[1].loc[grp[1].NewCases >= 5,:].index

    if len(start_indx) > 0:

        df_corona_noexp = pd.concat([df_corona_noexp,grp[1].loc[start_indx[0]:,:]])
df_corona_noexp.Country_Region.unique()
ctry_prov_noexp_lst = list(df_corona_noexp.set_index(['Country_Region','Province_State']).index.unique())

len(ctry_prov_noexp_lst)
#df_sub.reset_index(inplace=True)
#loop over countries not in ctry_lst, smaller countries or those which did not yet start to get sick

#split loop only for convenient testing, could refactor to same loop above

df_sub.set_index('ForecastId',inplace=True)

for (ctry,prov) in ctry_prov_noexp_lst:

    pred = predict_new_cases(ctry,prov,df_corona_noexp)

    add_history(pred)

    pred = pred[(pred['Date'] > TRAIN_LAST_DATE) & (pred['Date'] <= PRED_LAST_DATE)]

    pred = pred.merge(df_ctryinfo,left_on='Country/Region',right_on='country',how='left')

    pred.drop('country',axis=1,inplace=True)

    fat_pred = fat_model.predict(pred.iloc[:,2:].drop('NewCases',axis=1))

    fat_pred = np.round(np.maximum(fat_pred,0))

    test_id = df_test[(df_test['Country_Region'] == ctry) & 

                      (df_test['Province_State'] == prov) &

                      (df_test['Date'] > TRAIN_LAST_DATE)]['ForecastId']

    df_sub.loc[test_id,'ConfirmedCases']=pred['ConfirmedCases'].values

    df_sub.loc[test_id,'Fatalities']=fat_pred

df_sub.reset_index(inplace=True)
df_sub['ConfirmedCases'] = df_sub['ConfirmedCases'].apply(lambda x: max(x,1))
#check fatality rate per country to prepare for prediction by country factors

df_fat = df_last_day_train.groupby('Country_Region')[['ConfirmedCases','Fatalities']].sum().reset_index()

df_fat['FatRate'] = df_fat['Fatalities']/df_fat['ConfirmedCases']

df_fat.sort_values('FatRate',ascending=False)
df_fat.describe()
df_fat.FatRate.plot(kind='hist')
fatalities_avg = df_last_day_train.Fatalities.sum()/df_last_day_train.ConfirmedCases.sum()

fatalities_avg
fatalities_ctry_mean = np.mean(df_last_day_train.Fatalities/df_last_day_train.ConfirmedCases)

fatalities_ctry_mean
df_sub.describe()
df_score = df_eval.merge(df_sub, on=['ForecastId'], right_index=True, validate='1:1',suffixes=('_act','_pred'))

df_score['ConfirmedCasesError']=(np.log1p(df_score['ConfirmedCases_act'])-np.log1p(df_score['ConfirmedCases_pred']))**2

df_score['FatalitiesError']=(np.log1p(df_score['Fatalities_act'])-np.log1p(df_score['Fatalities_pred']))**2

score_c = np.sqrt(mean_squared_log_error(df_score['ConfirmedCases_act'], df_score['ConfirmedCases_pred']))

score_f = np.sqrt(mean_squared_log_error(df_score['Fatalities_act'], df_score['Fatalities_pred']))

print("Confirmed score: {0:.2f}, Fatalities score:{1:.2f}, Mean: {2:.2f}".format(score_c,score_f,np.mean([score_c, score_f])))
df_score.describe()
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

plt.scatter(x = np.log1p(df_score['ConfirmedCases_act']), y= np.log1p(df_score['ConfirmedCases_pred']),marker='.',color='b')

plt.ylabel("Predicted")

plt.xlabel("Actual")

plt.title("ConfirmedCases")

plt.subplot(1,2,2,)

plt.scatter(x = np.log1p(df_score['Fatalities_act']), y= np.log1p(df_score['Fatalities_pred']), marker='.',color='b')

plt.ylabel("Predicted")

plt.xlabel("Actual")

plt.title("Fatalities")

plt.show()
df_sub.to_csv("submission.csv",index=False)