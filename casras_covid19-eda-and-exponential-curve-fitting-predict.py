#EDA and exponential curve fitting exercise for basic prediction

#Using dataset from World Bank for population data and demographic info from dataset compiled by My Koryto

#Try to learn from countries that passed the infection peak for duration

#For fatalities rate use country demographics to check for feature importance (used xgboost)



import os

import pandas as pd

import numpy as np

from scipy.interpolate import Rbf

from scipy.optimize import curve_fit

from datetime import timedelta

from sklearn.metrics import mean_squared_log_error

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

import xgboost



import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('ggplot')



#Uploaded population per country file from World Bank

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
TRAIN_LAST_DATE = pd.to_datetime('3/11/2020')

PRED_LAST_DATE  = pd.to_datetime('4/23/2020')
#Population per country from World Bank at: https://data.worldbank.org/indicator/SP.POP.TOTL

df_pop = pd.read_excel(os.path.join('/kaggle/input','wdi-world-population-data',

                                    'API_SP.POP.TOTL_DS2_en_excel_v2_887218.xls'),skip_rows=3,header=3)
df_pop.head()
#using 2018 data, latest available at World Bank as of now

df_pop = df_pop.loc[:,['Country Name','2018']]

df_pop.rename(columns=({'2018':'Population'}),errors='raise',inplace=True)
df_train = pd.read_csv(os.path.join('/kaggle/input/covid19-global-forecasting-week-1','train.csv'))

df_test  = pd.read_csv(os.path.join('/kaggle/input/covid19-global-forecasting-week-1','test.csv'))

df_sub   = pd.read_csv(os.path.join('/kaggle/input/covid19-global-forecasting-week-1','submission.csv'))
df_train['Date'] = pd.to_datetime(df_train['Date'])

df_test['Date']  = pd.to_datetime(df_test['Date'])
#keep list of countries with more than 1 region

multiple_regions = list(df_train[df_train['Province/State'].notnull()]['Country/Region'].unique())

multiple_regions
#prevent NA rows dissapear in grouby operations

df_train['Province/State'].fillna('UniqueProvince',inplace=True)

df_test['Province/State'].fillna('UniqueProvince',inplace=True)
#remove data after last train data to evaluation df

df_eval  = df_train[df_train['Date'] >  TRAIN_LAST_DATE]

df_train = df_train[df_train['Date'] <= TRAIN_LAST_DATE]
#add ForecastId column

df_eval = df_eval.merge(df_test,on=['Date', 'Country/Region', 'Province/State'],how='left',validate='1:1')

df_eval.drop(columns=['Lat_x','Lat_y','Long_x','Long_y'],inplace=True)
df_eval
df_test
df_sub.head()
df_train
#calculate daily additions to confirmed cases

new_cases = df_train.groupby(['Country/Region','Province/State'])['ConfirmedCases'].transform(lambda x: x.diff().fillna(0))

df_train['NewCases'] = new_cases
#Japan data has problems, negative new cases

df_train[df_train['Country/Region']=='Japan'].head(20)
all_ctry_lst = list(df_train['Country/Region'].unique())
df_train['Country/Region'].nunique()
df_corona = df_train.groupby(['Country/Region','Date'])[['ConfirmedCases','Fatalities']].sum().reset_index()
#calculate daily additions to confirmed cases

new_cases = df_corona.groupby('Country/Region')['ConfirmedCases'].transform(lambda x: x.diff().fillna(0))

df_corona['NewCases'] = new_cases

df_corona
#replace day with 15K new cases caused by measure change in China with average of near dates

maxindx = df_corona.loc[df_corona['Country/Region']=='China',:].NewCases.idxmax()

avg_smooth = (df_corona.NewCases[maxindx-1]+df_corona.NewCases[maxindx+1])/2

df_corona.loc[maxindx,'NewCases']=avg_smooth
a = set(df_corona['Country/Region'].unique())

b = set(df_pop['Country Name'])

a.difference(b) #issues with country names betweent the files
#modify country names to match other databases - work in progress

df_pop.loc[df_pop['Country Name']=='Brunei Darussalam','Country Name'] = 'Brunei'

df_pop.loc[df_pop['Country Name']=='Congo, Rep.','Country Name'] = 'Congo (Brazzaville)'

df_pop.loc[df_pop['Country Name']=='Congo, Dem. Rep.','Country Name'] = 'Congo (Kinshasa)'

df_pop.loc[df_pop['Country Name']=='Czech Republic','Country Name'] = 'Czechia'

df_pop.loc[df_pop['Country Name']=='Egypt, Arab Rep.','Country Name'] = 'Egypt'

df_pop.loc[df_pop['Country Name']=='Russian Federation','Country Name'] = 'Russia'

df_pop.loc[df_pop['Country Name']=='Iran, Islamic Rep.','Country Name'] = 'Iran'

df_pop.loc[df_pop['Country Name']=='United States','Country Name'] = 'US'

df_pop.loc[df_pop['Country Name']=='Slovak Republic','Country Name'] = 'Slovakia'

df_pop.loc[df_pop['Country Name']=='Korea, Rep.','Country Name'] = 'Korea, South'

df_pop.loc[df_pop['Country Name']=='Bahamas, The','Country Name'] = 'The Bahamas'

#df_pop.loc[df_pop['Country Name']=='Gambia, The','Country Name'] = 'The Gambia' #both exist in corona db

df_pop.loc[df_pop['Country Name']=='Venezuela, RB','Country Name'] = 'Venezuela'

df_pop.loc[df_pop['Country Name']=='Kyrgyz Republic','Country Name'] = 'Kyrgyzstan'

df_pop.loc[df_pop['Country Name']=='St. Vincent and the Grenadines','Country Name'] = 'Saint Vincent and the Grenadines'
df_corona = df_corona.merge(df_pop,"left",left_on='Country/Region',right_on='Country Name',

                            indicator=True,validate='many_to_one')
df_corona._merge.value_counts()
#TBD fix country names match to population database

df_corona[df_corona._merge=='left_only']['Country/Region'].unique()
df_corona['new_cases_per_pop']=df_corona['NewCases']/df_corona['Population']*100000

df_corona.drop(['_merge'],axis=1,inplace=True)

df_corona
#check distribution of known total cases per country

df_corona.groupby('Country/Region').ConfirmedCases.max().describe()
df_corona['Country/Region'].nunique() #number of countries in dataset
def filter_out(x):

    f1 = x['ConfirmedCases'].max() > 50      #at least 50 sick cases

    f2 = x['Population'].min() > 1000000     #pop more than 1M, small countries with distorted percent of confirmed cases

    f3 = x['new_cases_per_pop'].max() > 0.1  #at least 0.1 cases per 100K pop

    return f1 & f2 & f3
#create reduced filtered dataset for analysis

df_analysis = df_corona.groupby('Country/Region').filter(filter_out).copy()
df_analysis['Country/Region'].nunique() #number of countries for analysis
df_analysis['Country/Region'].unique()
#for each country filter out all dates before there are at least 5 new daily cases

df_filt = pd.DataFrame(columns=df_analysis.columns)

for grp in df_analysis.groupby('Country/Region'):

    start_indx = grp[1].loc[grp[1].NewCases >= 5,:].index

    if len(start_indx) > 0:

        df_filt = pd.concat([df_filt,grp[1].loc[start_indx[0]:,:]])
#remove countries with less than 5 days of meaningful new daily cases

df_filt = df_filt.groupby('Country/Region').filter(lambda x: len(x.Date) > 4)
df_filt.groupby('Country/Region').Date.count().describe()
ctry_lst = list(df_filt['Country/Region'].unique())

len(ctry_lst) #final number of countries for analysis after filtering
def country_select(cntry,df):

    df_sel = df[df['Country/Region'] == cntry].reset_index(drop=True)

    #consistency check, fails for China after fixing 13/2 reports

    #assert(all(df_sel.ConfirmedCases == df_sel.NewCases.cumsum()))

    return df_sel.fillna(0)
fig, ax = plt.subplots()

df_filt.groupby('Country/Region').plot(y='NewCases',use_index=False,ax=ax,figsize = (10,6), marker='.',

                                 legend=False,title='New daily cases for each country')

ax.set_xlabel('Days from measurement start')

ax.set_ylabel('New daily cases')

plt.show()
#show maximum daily new cases as a percentage of population, "wall of shame" for country measures :-)

df_filt.groupby('Country/Region').new_cases_per_pop.max().sort_values(ascending=False)
df_filt = df_filt.set_index('Country/Region')

df_filt['max_ratio'] = df_filt.groupby('Country/Region').new_cases_per_pop.max()

df_filt.reset_index(inplace=True)

df_filt.sort_values(by=['max_ratio'],ascending=False,inplace=True,kind='mergesort') #mergesort is stable
#show graphs of daily new cases in batches of 5 to be able to see difference, sorted by severity

i = 0

for ctry,grp in df_filt.groupby('Country/Region',sort=False):

    if i%5==0: #new plot for group

        fig, ax = plt.subplots()

        ax.set_xlabel('Days from measurement start')

        ax.set_ylabel('New daily cases per 100K')

    grp.plot(y='new_cases_per_pop',use_index=False, ax=ax, figsize=(10,6), label=ctry, marker='.',

                                legend=True,title='New daily cases per 100K for each country')

    i +=1



plt.show()
#candidate functions definition for curve fitting

def exp_func(x,a,b):

    return a*np.exp(b*x)



def poly_func(x,a,b):

    return a*(x**b)



def sigmoid_func(x,a,b,c):

    return a/(1+np.exp(-b*(x-c)))
#loop over all countries

#check curve fit for exponential and polynomial function for countries which passed the peak of the infection

#uses new cases per day per 100K of the country population as time series values for radial basis function interpolation

#saves peak duration list for countries that seem to be after infection peak and the curve fitting params

peak_duration = {}

param_list = []

for ctry in ctry_lst:

    fig, ax = plt.subplots()

    df_country = country_select(ctry,df_filt)

    plt.plot(df_country.new_cases_per_pop,marker='.')

    rbf=Rbf(df_country.index,df_country.new_cases_per_pop,smooth=5)(df_country.index)

    plt.plot(rbf,marker='+')

    ax.set_title(ctry)

    plt.show()

    peak_indx = np.argmax(rbf)

    peak_date = df_country.Date.iloc[peak_indx]

    #are we at least 5 days after peak of new cases and peak detection 5 days after start?

    peak_reached = (len(df_country) > peak_indx+4) & (peak_indx > 5)

    if peak_reached:

        print("Peak reached in {0} days on {1}".format(peak_indx,peak_date))

        peak_duration[ctry]=peak_indx

        #exponential fit check

        params1, cov_params1 = curve_fit(exp_func,np.arange(peak_indx),rbf[:peak_indx],p0=[0.01,0.5])

        print("\nExponential fit parameters: a:{0:.4f} b:{1:.2f}".format(params1[0],params1[1]))

        print("Covariance matrix of parameters:")

        print(cov_params1)

        param_list.append(params1)

        #polynomial fit check

        params2, cov_params2 = curve_fit(poly_func,np.arange(peak_indx),rbf[:peak_indx],maxfev=10000,p0=[0.01,0.5])

        print("\nPolynomial fit parameters: a:{0:.4f} b:{1:.2f}".format(params2[0],params2[1]))

        print("Covariance matrix of parameters:")

        print(cov_params2)

        #sigmoid fit check

        #params3, cov_params3 = curve_fit(sigmoid_func,np.arange(peak_indx)+0.1,rbf[:peak_indx],maxfev=10000)

        #print("\nSigmoid fit parameters: a:{0:.2f} b:{1:.2f} c:{2:.2f}".format(params3[0],params3[1],params3[2]))

        #print("Covariance matrix of parameters:")

        #print(cov_params3) 

        #rough check for goodness of fit

        if np.diag(cov_params1).sum() > np.diag(cov_params2).sum():

            print("\nPolynomial fit seems better.")

        else:

            print("\nExponential fit seems better.")

    else:

        print("Peak not yet reached")
peak_duration
peaked_countries = list(peak_duration.keys())
param_list #curve fitting result parameters
avg_param = np.mean(param_list,axis=0)
#taking the 80% quantile as average since we have a bias from the first countries that recovered

avg_duration = round(np.quantile(list(peak_duration.values()),0.8))

avg_duration
max_duration = max(list(peak_duration.values()))

max_duration
#rough predict based on average/max duration to peak using exponential fitting parameters

FLAT = 5 #guess for number of days remaining at peak TBD optimize based on fitting to peaked countries

MAX_INC = 5 #arbitrary additional days prediction till peak for countries that lasted longer than history



def predict_new_cases(ctry):

    df_country = country_select(ctry,df_filt)

    pop_factor = int(df_pop[df_pop['Country Name']==ctry].Population)/100000

    ctry_dates = df_country[['Date']].copy()

    ctry_confirmed = df_country[['ConfirmedCases']].copy()

    

    rbf=Rbf(df_country.index,df_country.new_cases_per_pop,smooth=5)(df_country.index)

    peak_indx = np.argmax(rbf)

    #print(peak_indx)

    peak_date = df_country.Date.iloc[peak_indx]

    #are we at least 5 days after peak of new cases and peak detection 5 days after start?

    peak_reached = (len(df_country) > peak_indx+4) & (peak_indx > 5)

    

    #exponential fitting TBD take parameters fitted from other peaked countries

    if peak_indx > 5:

        params, _ = curve_fit(exp_func,np.arange(peak_indx),rbf[:peak_indx],p0=[0.01,0.5])

    else:

        params = avg_param

    #print("\nExponential fit parameters: a:{0:.2f} b:{1:.2f}".format(params[0],params[1]))

    

    if peak_reached:

        print("Peak reached in {0} days on {1}".format(peak_indx,peak_date))

        dur = peak_indx

        if len(df_country) < dur*2+1+FLAT:

            #add predictions to decrease

            for j in range(len(df_country),dur*2+1+FLAT):

                pred = exp_func(2*dur-j+FLAT,params[0],params[1])

                rbf = np.append(rbf,pred)

                ctry_dates = ctry_dates.append(ctry_dates.iloc[-1]+timedelta(1))

                ctry_confirmed = ctry_confirmed.append(ctry_confirmed.iloc[-1]+pred*pop_factor)

    else:

        print("Peak not yet reached")

        #naive predict duration till new cases per day peak

        if len(df_country) < int(np.ceil(avg_duration)):

            dur = int(np.ceil(avg_duration))

        elif len(df_country) < max_duration:

            dur = max_duration

        else:

            dur = len(df_country)+MAX_INC

            print("Past due peaking, arbitrary {} days prediction".format(MAX_INC))

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

        rbf = np.append(rbf,rbf[-1])

        ctry_dates = ctry_dates.append(ctry_dates.iloc[-1]+timedelta(1))

        ctry_confirmed = ctry_confirmed.append(ctry_confirmed.iloc[-1]+rbf[-1]*pop_factor)

    

    cases = rbf * pop_factor

    fig, ax = plt.subplots()

    plt.plot(np.arange(len(df_country)),cases[:len(df_country)],marker='+')

    plt.plot(np.arange(len(df_country),len(cases)),cases[len(df_country):],marker='.')

    ax.set_title(ctry+" predicted new daily cases")

    plt.show()

    

    print("Total predicted infected: {0:d}".format(int(ctry_confirmed.iloc[-1])))

    cases_df = pd.DataFrame(cases.astype(int),columns=['NewCases'])

    ctry_df = pd.DataFrame([ctry for i in range(len(cases))],columns=['Country/Region'])

    ctry_confirmed = ctry_confirmed.reset_index(drop=True).astype(int)

    return(pd.concat([ctry_df,ctry_dates.reset_index(drop=True),ctry_confirmed,cases_df],axis=1))
ctry = 'Iran'

pred = predict_new_cases(ctry)
df_last_day_train = df_train.groupby(['Country/Region','Province/State']).last().reset_index().copy()

df_last_day_train
#loop over countries to initialize prediction with last known values

df_sub.set_index('ForecastId',inplace=True)

for i in df_last_day_train.index:

    pred = df_last_day_train.loc[i,'ConfirmedCases']

    ctry = df_last_day_train.loc[i,'Country/Region']

    prvn = df_last_day_train.loc[i,'Province/State']

    test_id = df_test[(df_test['Country/Region'] == ctry) & (df_test['Province/State'] == prvn)]['ForecastId']

    df_sub.loc[test_id,'ConfirmedCases']=pred

df_sub.reset_index(inplace=True)
#loop over countries in ctry_lst which we predict exponentially

df_sub.set_index('ForecastId',inplace=True)

for ctry in ctry_lst:

    if ctry not in multiple_regions: #TBD to handle multiple regions

        pred = predict_new_cases(ctry)

        pred = pred[(pred['Date'] > TRAIN_LAST_DATE) & (pred['Date'] <= PRED_LAST_DATE)]

        test_id = df_test[df_test['Country/Region'] == ctry]['ForecastId']

        df_sub.loc[test_id,'ConfirmedCases']=pred['ConfirmedCases'].values

df_sub.reset_index(inplace=True)
df_sub['ConfirmedCases'] = df_sub['ConfirmedCases'].apply(lambda x: max(x,1))
#check fatality rate per country to prepare for prediction by country factors

df_fat = df_last_day_train.groupby('Country/Region')[['ConfirmedCases','Fatalities']].sum().reset_index()

df_fat['FatRate'] = df_fat['Fatalities']/df_fat['ConfirmedCases']

df_fat
df_fat.describe()
df_fat.FatRate.plot(kind='hist')
fatalities_avg = df_last_day_train.Fatalities.sum()/df_last_day_train.ConfirmedCases.sum()

fatalities_avg
fatalities_ctry_mean = np.mean(df_last_day_train.Fatalities/df_last_day_train.ConfirmedCases)

fatalities_ctry_mean
#country descriptive information from kaggle database

df_ctryinfo = pd.read_csv(os.path.join('/kaggle/input/countryinfo/','covid19countryinfo.csv'))
df_ctryinfo = df_ctryinfo.iloc[:164,:22]

df_ctryinfo
df_fat_pred = df_fat.merge(df_ctryinfo,left_on='Country/Region',right_on='country',how='left',validate='1:1')

df_fat_pred = df_fat_pred[['Country/Region','FatRate','density','medianage','urbanpop','hospibed','smokers',

                           'sexratio','lung','femalelung','malelung']].set_index('Country/Region').copy()
df_fat_pred.info()
df_fat_pred
imput = SimpleImputer(strategy='median')

fat_nona = pd.DataFrame(imput.fit_transform(df_fat_pred),columns=df_fat_pred.columns, index=df_fat_pred.index)

fat_nona
fat_nona.corr()
x=fat_nona.iloc[:,1:]

y=fat_nona[['FatRate']]
#clf = RandomForestRegressor(max_depth=3)

#clf.fit(x,np.ravel(y))

clf = xgboost.XGBRegressor(objective='reg:squarederror')

clf.fit(x,y)
#hospital beds and male lung cancer seem to be the most predictive

pd.DataFrame(clf.feature_importances_,index=x.columns,columns=['Importance'])
clf_preds = np.maximum(0,clf.predict(x))

mean_squared_log_error(clf_preds,y)
df_fat['FatRate'] = clf_preds
#loop over countries to set country specific fatality rates

df_sub.set_index('ForecastId',inplace=True)

for i in df_last_day_train.index:

    ctry = df_last_day_train.loc[i,'Country/Region']

    frate = df_fat.loc[df_fat['Country/Region']==ctry,'FatRate']

    prvn = df_last_day_train.loc[i,'Province/State']

    test_id = df_test[(df_test['Country/Region'] == ctry) & (df_test['Province/State'] == prvn)]['ForecastId']

    df_sub.loc[test_id,'Fatalities']=round(df_sub.loc[test_id,'ConfirmedCases']*frate.values)

df_sub.reset_index(inplace=True)
#placeholder for calculating fatalities factor per country/cluster

#fatalities_avg = df_last_day_train.ConfirmedCases.sum()/df_last_day_train.Fatalities.sum()

#df_sub['Fatalities'] = round(df_sub['ConfirmedCases']/fatalities_avg)
df_sub.describe()
df_score = df_eval.merge(df_sub, on=['ForecastId'], right_index=True, validate='1:1',suffixes=('_act','_pred'))

df_score['ConfirmedCasesError']=(np.log1p(df_score['ConfirmedCases_act'])-np.log1p(df_score['ConfirmedCases_pred']))**2

df_score['FatalitiesError']=(np.log1p(df_score['Fatalities_act'])-np.log1p(df_score['Fatalities_pred']))**2



score_c = np.sqrt(mean_squared_log_error(df_score['ConfirmedCases_act'], df_score['ConfirmedCases_pred']))

score_f = np.sqrt(mean_squared_log_error(df_score['Fatalities_act'], df_score['Fatalities_pred']))

print("Confirmed score: {0:.2f}, Fatalities score:{1:.2f}, Mean: {2:.2f}".format(score_c,score_f,np.mean([score_c, score_f])))
df_score.describe()
plt.scatter(x = np.log1p(df_score['ConfirmedCases_act']), y= np.log1p(df_score['ConfirmedCases_pred']), marker='.')
plt.scatter(x = np.log1p(df_score['Fatalities_act']), y= np.log1p(df_score['Fatalities_pred']), marker='.')
df_sub.to_csv("submission.csv",index=False)