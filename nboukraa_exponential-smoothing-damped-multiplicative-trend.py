import numpy as np 

import pandas as pd 



import warnings

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib import dates

import datetime as dt



from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

from sklearn.neighbors import DistanceMetric, KNeighborsRegressor, RadiusNeighborsRegressor

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline



from functools import reduce

from statsmodels.tsa.holtwinters import ExponentialSmoothing



idx = pd.IndexSlice
df = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv', index_col=0)
# Typo

df['Country_Region'].replace('Taiwan*', 'Taiwan', inplace=True)

df = df[df['Country_Region'] != 'Diamond Princess']
# Country Codes

df_codes = (pd.read_csv('../input/my-covid19-dataset/iso-country-codes.csv')

            .rename(columns={'Country Name': 'Country_Region', 'Alpha-3 code': 'Country Code'})

            .loc[:,['Country_Region', 'Country Code']])

df = (pd.merge(df, df_codes, how='left', on='Country_Region'))
# Locations

def location(state, country):

    if type(state)==str and not state==country: 

        return country + ' - ' + state

    else:

        return country



# Timeline 

df['Date'] = df['Date'].apply(lambda x: (dt.datetime.strptime(x, '%Y-%m-%d')))

df['Location'] = df[['Province_State', 'Country_Region']].apply(lambda row: location(row[0],row[1]), axis=1)



t_start = df['Date'].unique()[0]

t_end = df['Date'].unique()[-1]



print('Number of Locations: ' + str(len(df['Location'].unique())))

print('Dates: from ' + np.datetime_as_string(t_start, unit='D') + ' to ' + 

                                  np.datetime_as_string(t_end, unit='D') + '\n')
lst_out = ['Cayman Islands', 'Curacao', 'Faroe Islands', 'French Guiana', 'French Polynesia', 'Guadeloupe', 

           'Mayotte', 'Martinique', 'Reunion', 'Saint Barthelemy', 'St Martin', 'Aruba', 'Channel Islands', 

           'Gibraltar', 'Montserrat', 'Diamond Princess', 'From Diamond Princess', 'Puerto Rico', 

           'Virgin Islands', 'Guam']



df_loc = (df.loc[[((c not in lst_out) and (p not in lst_out)) 

                 for (c, p) in zip(df['Province_State'], df['Country_Region'])], 

                ['Location','Province_State', 'Country_Region','Country Code']]

          .drop_duplicates())
df_pop = pd.read_csv('../input/my-covid19-dataset/citypopulation-de/population.csv')
df_loc = pd.merge(df_loc, df_pop, how='left', on=['Province_State', 'Country_Region'])
# Population estimate as of July 2020 published by the UN (in '000 people)

df_pop = (pd.read_csv('../input/my-covid19-dataset/un-org/population-2020.csv')

          .rename(columns={'ISO 3166-1 alpha code': 'Country Code'}))
df_pop = df_pop.loc[~df_pop['Country Code'].isin(['CHN', 'USA', 'CAN', 'AUS']), ['Country Code', 'Population']]
df_pop = pd.merge(df_loc, df_pop, how='left', on='Country Code', suffixes=('_P/S','_C/R'))
# Population ('000) of the State if available, of the Country otherwise

df_pop['Population'] = (df_pop[['Population_P/S','Population_C/R']]

                        .apply(lambda x: x[1] if np.isnan(x[0]) else x[0], axis=1))
df_pop = df_pop.loc[:,['Location','Population']].set_index('Location', verify_integrity=True)
df = pd.merge(df, df_pop, how='left', on='Location')
df['Fatalities per Million'] = df['Fatalities'] / df['Population'] * 1000

df['Confirmed Cases per Million'] = df['ConfirmedCases'] / df['Population'] * 1000
# Day count since first confirmed case (resp. first fatality)

df['Day Count Confirmed'] = (df['ConfirmedCases']>0).groupby(df['Location']).cumsum().astype('int')

df['Day Count Fatalities'] = (df['Fatalities']>0).groupby(df['Location']).cumsum().astype('int')
# We correct a few inconsistencies in the dataset to make sure that day counts are strictly monotonous

# (the tuple (location, day counts) will be used as index)

df['Fatalities'] = df.loc[df['Day Count Fatalities']>0, 'Fatalities'].apply(lambda x: max(x,1))

df['Day Count Fatalities'] = (df['Fatalities']>0).groupby(df['Location']).cumsum().astype('int')
# New confirmed cases (resp. fatalities)

df['New Cases'] = df['ConfirmedCases'].groupby(df['Location']).diff() / df['ConfirmedCases']

df['New Fatalities'] = df['Fatalities'].groupby(df['Location']).diff() / df['Fatalities']



# Case Fatality Rate (i.e. ratio between confirmed cases and confirmed fatalities)

# (may help identify outliers, i.e. countries where actual cases may be particularly underestimated)

df['Case Fatality Rate'] = df['Fatalities'] / df['ConfirmedCases']
#################################################################################################################

# Locations where the number of fatalities remains very limited and progresses very slowly are ignored 

df_all = df



df = df[(df['Day Count Fatalities']>0) & 

        ((df['Fatalities per Million']>1) | # at least 1 fatality per million

        (df['New Fatalities']>1/7))] # at least doubling every week

#################################################################################################################
df_testing = (pd.read_csv('../input/my-covid19-dataset/ourworldindata/tests/tests-vs-confirmed-cases-covid-19.csv')

              .loc[:,['Entity', 'Total COVID-19 tests']]

              .rename(columns={'Entity':'Location', 'Total COVID-19 tests': 'Tests'}))
df = pd.merge(df, df_testing, how='left', on='Location')
# Number of confirmed cases divided by number of tests (where the number of tests was unknown, we have made 

# the (strong) assumption that only infected people were tested)

df['Tests per Million'] = df['Tests'] / df['Population'] * 1000

df['Tests per Million'].fillna(df['Confirmed Cases per Million'], inplace=True)

df.drop(columns=['Province_State', 'Country_Region', 'Tests', 'Population'], inplace=True)



# 'Confirmed Rate' is defined as the proportion of confirmed cases in the tested population

df['Confirmed Rate'] = df['Confirmed Cases per Million'] / df['Tests per Million']
df_plt = df.set_index(['Location','Day Count Fatalities'], verify_integrity=True)



mask = (~df_plt.index.get_level_values(0).duplicated(keep='last')) & (df_plt['Confirmed Rate']<1)

df_plt = df_plt.loc[mask, ['Fatalities per Million','Tests per Million','Confirmed Rate']].reset_index()



df_plt.plot(x='Day Count Fatalities', y='Confirmed Rate', c='Tests per Million', 

            kind='scatter', colormap='coolwarm_r', sharex=False, figsize=(17.5,7.5))



# Annotations

x = df_plt['Day Count Fatalities'].values

y = df_plt['Confirmed Rate'].values

z = df_plt['Location'].values



for i, txt in enumerate(z):

    if y[i] > .25:

        plt.text(x[i]+.005, y[i]+.005, txt, rotation=0, rotation_mode='anchor')



plt.title('Proportion of confirmed cases among tested population\n (Color = Number of Tests per Million)')

plt.xlabel('Number of days since first fatality')

plt.ylabel('')

plt.show()
df_plt = df.set_index(['Location','Day Count Fatalities'], verify_integrity=True)



mask = ~df_plt.index.get_level_values(0).duplicated(keep='last')

df_plt = df_plt.loc[mask, ['Fatalities per Million','Tests per Million','Case Fatality Rate']].reset_index()



df_plt.plot(x='Day Count Fatalities', y='Case Fatality Rate', c='Tests per Million', 

            kind='scatter', colormap='coolwarm_r', sharex=False, figsize=(17.5,7.5))



# Annotations

x = df_plt['Day Count Fatalities'].values

y = df_plt['Case Fatality Rate'].values

z = df_plt['Location'].values



for i, txt in enumerate(z): # annotate outliers (case fatality rate above 5%)

    if (y[i]>.05) and (y[i]<.25):

        plt.text(x[i]+.005, y[i]+.005, txt, rotation=0, rotation_mode='anchor')



plt.ylim((0,.25))

plt.title('Proportion of fatalities among confirmed cases\n (Color = Number of Tests per Million)')

plt.xlabel('Number of days since first fatality')

plt.ylabel('')

plt.show()
# Compute key statistics... 

df_frate = (df.loc[:,['Day Count Fatalities', 'Case Fatality Rate']].groupby('Day Count Fatalities')

            .agg(['count', 'mean', 'std']))



y_count = df_frate.loc[:,('Case Fatality Rate', 'count')]

y_mean = df_frate.loc[:,('Case Fatality Rate', 'mean')]

y_std  = df_frate.loc[:,('Case Fatality Rate', 'std')]



# ... and plot them

plt.plot(df_frate.index, y_mean, c='w')

plt.fill_between(df_frate.index, y_mean - y_std, y_mean + y_std, alpha=.5)

plt.ylim(bottom=0)

plt.title('Mean Case Fatality Rate (% of confirmed cases, since first fatality)')

plt.show()



plt.bar(df_frate.index, y_count, color='grey')

plt.title('Number of countries')

plt.gca().set_xlabel('Number of days since first fatality')

plt.show()
# Fatality rate 30 days after the first fatality

df_plt = df[df['Day Count Fatalities']==30]

df_plt.plot(x='Day Count Confirmed', y='Case Fatality Rate', c='Fatalities per Million', 

            kind='scatter', colormap='coolwarm', sharex=False, figsize=(10,7.5))



# Annotations

x = df_plt['Day Count Confirmed'].values

y = df_plt['Case Fatality Rate'].values

z = df_plt['Location'].values



for i, txt in enumerate(z):

    # fatality rates are expected to be close to 2%

    # (the number of confirmed cases is probably underestimated otherwise)

    plt.text(x[i]+.005, y[i]+.005, txt, rotation=0, rotation_mode='anchor')



plt.title('Case Fatality Rate 30 days after the first fatality')

plt.xlabel('Number of days since first confirmed case')

plt.show()
# We extract the most recent data available for each location and plot correlations

mask = (~df.set_index(['Location','Date'], verify_integrity=True)

        .index.get_level_values(0)

        .duplicated(keep='last'))

df_plt = (df[mask].drop(columns=['Date','Country Code',

                                 'Day Count Confirmed','ConfirmedCases','Confirmed Cases per Million',

                                 'Confirmed Rate','Fatalities'])

          .set_index(['Location'], verify_integrity=True))
# Correlation matrix

df_plt.corr().style.background_gradient(cmap='Reds').set_precision(2)
pd.plotting.scatter_matrix(df_plt, figsize=(15,15))

plt.show()
# Last available data

df = df.set_index(['Location','Day Count Fatalities'], verify_integrity=True).sort_index()

mask = ~df.index.get_level_values(0).duplicated(keep='last')
# Standardized case fatality rates

df_cfr = df.loc[mask,:].reset_index(level=1, drop=True).loc[:,['Country Code', 'Case Fatality Rate']]

df_cfr[['Case Fatality Rate']] = df_cfr[['Case Fatality Rate']].apply(lambda x: (x-x.min()) / (x.max()-x.min()))
# Selected state, and forecasting period in days

state = 'US - New York'

fperiod = 90
# Fatalities per million for the selected state

fatalities = df.loc[idx[state, :], 'Fatalities per Million'].reset_index(drop=True)
# Guess parameters

init_alpha = .3 #.5

init_beta = .7 #.1

init_phi = .8

initial_level = fatalities[0]

initial_slope = fatalities[1] / fatalities[0]

start_params = [init_alpha, init_beta, initial_level, initial_slope, init_phi]



# Search for best parameters

with warnings.catch_warnings():

    warnings.simplefilter('ignore')

    fmodel = (ExponentialSmoothing(fatalities, trend='mul', damped=True, seasonal=None)

              .fit(start_params=start_params, remove_bias=True, use_basinhopping=True))

    fcast = fmodel.forecast(fperiod)
# Visualisation 

plt.figure(figsize=(10,5))

fcast.plot(style='--', marker='', color='green', legend=True, label='Forecast')

fmodel.fittedvalues.plot(style='--', marker='', color='blue', legend=True, label='Smoothed')

fatalities.plot(linestyle='', marker='.', color='red', legend=True, label='Actual values')



keys = ['smoothing_level', 'smoothing_slope', 'damping_slope']

alpha, beta, phi = list(map(fmodel.model.params.get, keys))

txt_params = ('Exponential smoothing with parameters:\n\n\t' + r'$\alpha=${}'.format(alpha) + '\n\t' + 

              r'$\beta=${}'.format(beta) + '\n\t' + r'$\phi=${}'.format(phi))



plt.gcf().text(1, 0.5, txt_params)

plt.title(r'Fatalities per Million in {}'.format(state) + '\n' + '(multiplicative damped trend)')

plt.xlabel('Day Count Fatalities')

plt.show()
# Define placeholders for states of interests

df_params = pd.DataFrame(index=df.index.unique(level='Location'), 

                         columns=['Alpha', 'Beta', 'Phi' ,'Forecast per Million'])

keys = ['smoothing_level', 'smoothing_slope', 'damping_slope']



# Loop through all locations

for state in df.index.get_level_values(level=0).unique():

    fatalities = df.loc[idx[state, :], 'Fatalities per Million'].reset_index(drop=True)

    

    # At least two data points are required to run the model

    if len(fatalities.dropna().index) < 2: continue

    

    # Get optimal parameters

    with warnings.catch_warnings():

        warnings.simplefilter('ignore')

        

        fmodel = (ExponentialSmoothing(fatalities, trend='mul', damped=True, seasonal=None)

                  .fit(remove_bias=True, smoothing_seasonal=0))

        _, beta, phi = list(map(fmodel.model.params.get, keys))

        

        # Re-run optimization if forecast is not constant

        if not ((beta==0) and (phi==0)):

            

            # Guess parameters

            init_alpha = .3 #.5

            init_beta = .7 #.1

            init_phi = .8

            initial_level = fatalities[0]

            initial_slope = fatalities[1] / fatalities[0]

            start_params = [init_alpha, init_beta, initial_level, initial_slope, init_phi]



            # Search for best parameters

            fmodel = (ExponentialSmoothing(fatalities, trend='mul', damped=True, seasonal=None)

                      .fit(start_params=start_params, remove_bias=True, use_basinhopping=True))

            

        # Consistency check: re-run where fatalities exceed 2% of the entire population

        if fcast.iloc[-1]>.02*1e3:

            

            # Guess alpha and beta subject to phi=0.80

            fmodel = (ExponentialSmoothing(fatalities, trend='mul', damped=True, seasonal=None)

                      .fit(damping_slope=.8, remove_bias=True, use_basinhopping=True))

    

    # Save model parameters

    df_params.loc[state, ['Alpha', 'Beta', 'Phi']] = list(map(fmodel.model.params.get, keys))

    

    # Save forecast (cumulated number of fatalities at the end of the [90]-day period)

    df_params.loc[state, 'Forecast per Million'] = fmodel.forecast(fperiod).iloc[-1]
# Convert data to numeric values

df_params = df_params.apply(pd.to_numeric, errors='ignore')
# Training set (list of countries with interpretable results)

mask = (

    (pd.isnull(df_params['Phi'])) | # no solution found

    ((df_params['Beta']==0) & (df_params['Phi']==1)) | # no damping (fatalities increase indefinitely)

    (df_params['Phi']==0) # dummy forecast (fatality counts remain constant)

)

        

idx_data = df_params.index[~mask]
# Parameters

df_plt = df_params.dropna().reset_index()

s, x, y, z = zip(*df_plt[['Location', 'Alpha', 'Beta', 'Phi']].values)
# Plot smoothing level vs smoothing slope

plt.figure(figsize=(10,10))

plt.scatter(x=x, y=y, marker='.')



for i, txt in enumerate(s):

    if not ((x[i] in [0,1]) or (y[i] in [0,1]) or (abs(x[i]-y[i])<1e-2)): # annotate non-naive model parameters

        plt.text(x[i]-.02, y[i]+.02, i, rotation=0, rotation_mode='anchor', fontsize=8)



plt.title(r'Model parameters: x-axis$=\alpha$, y-axis$=\beta$')

plt.xlabel(r'Smoothing level ($\alpha$)')

plt.ylabel(r'Smoothing slope ($\beta$)')

plt.tight_layout()

plt.show()
# Plot smoothing slope vs damping factor

plt.figure(figsize=(10,10))

sc = plt.scatter(x=y, y=z, c=x, vmin=0, vmax=1, marker='.', cmap='Blues')

plt.colorbar(sc)



for i, txt in enumerate(s):

    if y[i]*(1-z[i])>1e-2: # annotate 'nicest' model parameters (i.e. s-shaped forecasts)

        plt.text(y[i]-.02, z[i]+.01, i, rotation=0, rotation_mode='anchor', fontsize=8)



t = np.arange(.01, 1., .01)

plt.plot(t, 1-1e-2/t, 'r--')

plt.ylim((0,1))

plt.xlim((0,1))        

        

plt.title(r'Model parameters: x-axis$=\beta$, y-axis$=\phi$ and $color=\alpha$')

plt.xlabel(r'$\beta$')

plt.ylabel(r'$\phi$')

plt.tight_layout()

plt.show()
df_plt.loc[(df_plt['Beta']==0)&(df_plt['Phi']==0)]
df_plt.loc[(df_plt['Beta']==0)&(df_plt['Phi']==1)]
df_plt.loc[df_plt['Forecast per Million']>.01*1e6]
df_plt.loc[[not ((a in [0,1]) or 

                 (b in [0,1]) or 

                 (abs(a-b)<1e-2)) for (a,b) in df_plt[['Alpha','Beta']].values]]
df_plt = pd.merge(df_plt, df_pop, how='left', on='Location')

df_plt['Forecast'] = df_plt['Forecast per Million'] * df_plt['Population'] / 1000
df_plt.sort_values(by='Forecast').iloc[100:].plot(

    x='Location', y='Forecast', kind='barh', fontsize=18, legend=False)

plt.gcf().set_size_inches(30, 50)

plt.gca().set_xscale('log')

plt.grid(color='grey', linestyle='--', linewidth=.5)

plt.ylabel(None)

txt = 'Predicted number of fatalities within the next {} days (logarithmic scale)'

plt.title(txt.format(fperiod), fontsize=24)

plt.show()
df_plt.sort_values(by='Forecast', ascending=False).iloc[:10,:].plot(

    x='Location', y='Forecast', kind='bar', legend=False)

txt = 'Locations with the highest predicted number of fatalities within the next {} days'

plt.title(txt.format(fperiod))

plt.gcf().set_size_inches(12.5, 7.5)

plt.xticks(rotation=45)

plt.xlabel(None)

plt.show()
# Position of each state on the Earth (in radians)

df_r = pd.read_csv('../input/my-covid19-dataset/latlong.csv').set_index('Location', verify_integrity=True).loc[:,['Lat', 'Long']]
# Pairwise distances between countries (in km, 6371 is the Earth's radius in km)

R = 6371

hs = DistanceMetric.get_metric('haversine')

df_dist = pd.DataFrame(data=hs.pairwise(np.radians(df_r)) * R, index=df_r.index, columns=df_r.index)



# standardisation

df_dist /= df_dist.values.max()
df_wdi = (pd.read_csv('../input/my-covid19-dataset/world-bank/world-development-indicators.csv')

          .dropna(subset=['Country Code','Series Name'], how='all')

          .set_index(['Country Code', 'Series Name'], verify_integrity=True)

          .drop(columns=['Country Name', 'Series Code'])

          .replace({'..': np.nan})

          .dropna(how='all')

          .astype(float))
# Take the latest data available (2018 figures in most cases)

def last_available_data(row):

    res = [d for d in row if not np.isnan(d)]

    return float(res[-1])



df_wdi = (df_wdi

          .apply(lambda row: last_available_data(row), axis=1)

          .reset_index()

          .pivot(index='Country Code', columns='Series Name', values=0))
# Fill in missing values with world values if available, or global medians otherwise

wdi_default = df_wdi.fillna(df_wdi.median()).loc['WLD',:]

df_wdi = df_wdi.fillna(wdi_default).reset_index()
# We use Singapore as proxy for Taiwan for macroeconomic and demographic data

# (Note: fatality figures are not available for Hong-Kong and Macao on a stand-alone basis)

new_row = df_wdi.loc[df_wdi['Country Code']=='SGP'].copy(deep=True)

new_row.loc[:,'Country Code'] = 'TWN'

df_wdi = df_wdi.append(new_row, ignore_index=True)
# Risk factors: obesity (2016 figures, averaged between women and men)

df_ncd1 = (pd.read_csv('../input/my-covid19-dataset/ncd-risc/obesity/NCD_RisC_Lancet_2017_BMI_age_standardised_country.csv')

              .loc[:,['ISO', 'Sex', 'Prevalence of BMI>=30 kg/m2 (obesity)']]

              .rename(columns={'ISO': 'Country Code'})

              .groupby('Country Code')

              .mean()

              .reset_index())
# Risk factors: blood pressure (2015 figures, averaged between women and men)

df_ncd2 = (pd.read_csv('../input/my-covid19-dataset/ncd-risc/blood-pressure/NCD_RisC_Lancet_2016_BP_age_standardised_countries.csv')

               .loc[:,['ISO', 'Sex', 'Prevalence of raised blood pressure']]

               .rename(columns={'ISO': 'Country Code'})

               .groupby('Country Code')

               .mean()

               .reset_index())
# Preexisting health conditions: cancer prevalence, cardiovascucular diseases, chronic respiratory condition, 

# diabetes and kidney diseases (2017 figures, age-standardized)

df_ihme = pd.read_csv('../input/my-covid19-dataset/ihme-gdb/IHME-GBD_2017_DATA-8e93cebf-1.csv')

mask = [cause_name in ['Neoplasms','Cardiovascular diseases',

                       'Chronic respiratory diseases','Diabetes and kidney diseases'] 

        for cause_name in df_ihme['cause_name']]



df_ihme = (df_ihme.rename(columns={'Neoplasms': 'Cancer prevalence'})

           .loc[mask, ['location_name','cause_name','val']]

           .pivot(index='location_name', columns='cause_name', values='val')

           .reset_index()

           .rename(columns={'location_name': 'Location'})

           .merge(df_codes, how='left', left_on='Location', right_on='Country_Region')

           .drop(columns=['Location','Country_Region']))
# Additional country-specific demographic features are merged together

lst_df = [df_wdi, df_ncd1, df_ncd2, df_ihme]

df_feat = reduce(lambda df_left, df_right: pd.merge(df_left, df_right, how='left', on='Country Code'), lst_df)
# Last, we add (location-specific, standardized) case fatality rates

df_feat = (df_cfr.reset_index()

          .merge(df_feat, how='left', on='Country Code').drop(columns=['Country Code']))
# Features are standardized

df_feat = df_feat.set_index('Location', verify_integrity=True).apply(lambda x: (x-x.min()) / (x.max()-x.min()))



# We replace missing values for obesity, diabetes, pressure and testing by medians

# (this is certainly too simplistic)

df_feat.fillna(df_feat.median(), inplace=True)
# Full set of features

#df_feat.loc[df_feat.isnull().any(axis=1)]

df_feat.describe().T
# Model parameters

#df_params.loc[df_params.isnull().any(axis=1)]

df_params.describe()
# Data for locations with meaningful model parameters 

X_data = (df_params

          .loc[idx_data, ['Alpha', 'Beta', 'Phi']]

          .merge(df_feat, how='left', left_index=True, right_index=True))



# Parameters to predict

idx_pred = [idx for idx in df_params.index if idx not in idx_data]



X_pred = (df_params

          .loc[idx_pred, ['Alpha', 'Beta', 'Phi']]

          .merge(df_feat, how='left', left_index=True, right_index=True))



# Full dataset

X = X_data.iloc[:, 3:].values # standardised features

y = X_data.iloc[:, :3].values # model parameters
# Training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=99)
# Cross validation to guess k=n_neighbors (one output at a time)

lbl = [r'$\alpha$', r'$\beta$']

c = ['green', 'blue']

rng_k = range(1,15)



for i in range(2):

    scores = []

    

    for k in rng_k:

        rgr = KNeighborsRegressor(n_neighbors=k, weights='distance')

        rgr.fit(X_train, y_train[:,i])

        scores.append(rgr.score(X_test, y_test[:,i]))

    

    plt.plot(rng_k, scores, label=lbl[i], color=c[i], linestyle='dashed', marker='.', markerfacecolor='grey')



plt.title('Coefficient of determination of the prediction')

plt.xlabel('Number of neighbors')

plt.legend(loc='lower right')

plt.show()
# Cross validation to guess k=n_neighbors (multi-output) and metric ('minkowski' with p=3)

rng_k = range(1,15)

lst_weights = ['distance', 'uniform']

lst_metrics = ['minkowski', 'chebyshev']

rng_p = range(1,4)



def rgr_plot(w, m, p):

    scores = []

    

    for k in rng_k:

        rgr = KNeighborsRegressor(n_neighbors=k, weights=w, metric=m, p=p)

        rgr.fit(X_train, y_train)

        y_pred = rgr.predict(X_test)

        score = r2_score(y_test, y_pred, multioutput='uniform_average')

        scores.append(score)

    

    label = w + ' - ' + m + ' - ' + str(p)

    plt.plot(rng_k, scores, linestyle='dashed', marker='.', label=label)
for w in lst_weights:       

    for m in lst_metrics:

        if m=='minkowski':

            for p in rng_p:

                rgr_plot(w, m, p)

        elif m=='chebyshev':

            rgr_plot(w, m, p=-1)



plt.title('Coefficient of determination of the prediction')

plt.xlabel('Number of neighbors')

plt.legend()

plt.show()
# Prediction

n_neighbors = 10

knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights='uniform', metric='minkowski', p=2)
# Cross validation to guess best radius

rng_r = [t*.25 for t in range(1,7)]



def rgr_plot(w, m, p):

    scores = []

    

    for r in rng_r:

        rgr = RadiusNeighborsRegressor(radius=r, weights=w, metric=m, p=p)

        rgr.fit(X_train, y_train)

        

        with warnings.catch_warnings():

            warnings.simplefilter('ignore')

            y_pred = rgr.predict(X_test)



        if np.any(np.isnan(y_pred)):

            score = np.nan

        else:

            score = r2_score(y_test, y_pred, multioutput='uniform_average')

        scores.append(score)

    

    label = w + ' - ' + m + ' - ' + str(p)

    plt.plot(rng_r, scores, linestyle='dashed', marker='.', label=label)
for w in lst_weights:       

    for m in lst_metrics:

        if m=='minkowski':

            for p in rng_p:

                rgr_plot(w, m, p)

        elif m=='chebyshev':

            rgr_plot(w, m, p=-1)



plt.title('Coefficient of determination of the prediction')

plt.xlabel('Radius')

plt.legend()

plt.show()
# Prediction

radius = 1.0

rnn = RadiusNeighborsRegressor(radius=radius, weights='uniform', metric='minkowski', p=2)
lnr = LinearRegression(copy_X=True, fit_intercept=False)
lnr.fit(X_train,y_train)

y_pred = lnr.predict(X_test)

score = r2_score(y_test, y_pred, multioutput='uniform_average')



print('Coefficient of determination of the linear regression: {:.2%}'.format(score))
rdg = RidgeCV(alphas=[10**n for n in range(-4,4)])
rdg.fit(X_train,y_train)

y_pred = rdg.predict(X_test)

score = r2_score(y_test, y_pred, multioutput='uniform_average')



print('Coefficient of determination of the linear regression: {:.2%}'.format(score))
# Cross validation to guess d=degree of the polynom

scores = []

rng_d = range(1,5)



for d in rng_d:

    pln = Pipeline([('poly', PolynomialFeatures(degree=d)), 

                    ('linear', LinearRegression(fit_intercept=True))]) 

    pln.fit(X_train, y_train)

    y_pred = pln.predict(X_test)

    score = r2_score(y_test, y_pred, multioutput='uniform_average')

    scores.append(score)



plt.plot(rng_d, scores, color='red', linestyle='dashed', marker='.', markerfacecolor='grey')

plt.title('Coefficient of determination of the prediction (Linear)')

plt.xlabel('Degree')

plt.show()
# Cross validation to guess d=degree of the polynom (with regularisation)

scores = []

rng_d = range(1,6)



for d in rng_d:

    pln = Pipeline([('poly', PolynomialFeatures(degree=d)), 

                    ('ridge', Ridge(alpha=1e1, copy_X=True, fit_intercept=True))]) 

    pln.fit(X_train, y_train)

    y_pred = pln.predict(X_test)

    score = r2_score(y_test, y_pred, multioutput='uniform_average')

    scores.append(score)



plt.plot(rng_d, scores, color='red', linestyle='dashed', marker='.', markerfacecolor='grey')

plt.title('Coefficient of determination of the prediction (Ridge)')

plt.xlabel('Degree')

plt.show()
pln = Pipeline([('poly', PolynomialFeatures(degree=2)), 

                ('ridge', Ridge(alpha=1e1, copy_X=True, fit_intercept=False))])
pln.fit(X_train,y_train)

y_pred = pln.predict(X_test)

score = r2_score(y_test, y_pred, multioutput='uniform_average')



print('Coefficient of determination of the linear regression: {:.2%}'.format(score))
rgr = knn # KNN Regressor

#rgr = rnn # Radius Neighbors Regressor

#rgr = lnr # Linear Regressor

#rgr = rdg # Ridge

#rgr = pln # Polynomial Regression

#rgr = lgr # Logistic Regression
# Fit with all data available

rgr.fit(X, y)



# Prediction based on selected regressor

y_pred = rgr.fit(X, y).predict(X_pred.iloc[:, 3:].values)

X_pred.loc[:,['Alpha', 'Beta','Phi']] = y_pred
# Update model parameters

df_RGR = (pd.concat([X_data, X_pred]).loc[:,['Alpha','Beta','Phi']].reset_index()

          .merge(df_params.reset_index(), how='left', on='Location', suffixes=('_RGR',''))

          .drop_duplicates().set_index('Location', verify_integrity=True))



df_RGR.loc[idx_pred, ['Alpha','Beta','Phi']] = (df_RGR.loc[idx_pred, ['Alpha_RGR','Beta_RGR','Phi_RGR']]

                                                .apply(pd.to_numeric).values)



df_params = df_RGR.drop(columns=['Alpha_RGR','Beta_RGR','Phi_RGR'])
# Interpolation using customized parameters

fperiod = 90 # forecasting period in days

df = (df_all

      .loc[df_all['Day Count Fatalities']>0]

      .set_index(['Location','Day Count Fatalities'], verify_integrity=True)

      .sort_index(level=[0, 1], ascending=[1, 1]))
# Pick a state at random

state = np.random.choice(idx_pred, 1)[0]

print(state)
# Historical curve

mask = (df.index.get_level_values(0)==state) & (df.index.get_level_values(1)>0)

fatalities = df.loc[mask, 'Fatalities'].reset_index(drop=True)



# Exponential smoothing parameters

alpha, beta, phi = df_params.loc[idx[state], ['Alpha', 'Beta', 'Phi']].values



if len(fatalities)>2:

    

    with warnings.catch_warnings():

        warnings.simplefilter('ignore')

        fmodel = (ExponentialSmoothing(fatalities, trend='mul', damped=True, seasonal=None)

                  .fit(smoothing_level=alpha, smoothing_slope=beta, damping_slope=phi, 

                       remove_bias=True, use_basinhopping=True))

        fcast = fmodel.forecast(fperiod)
# Visualisation 

if len(fatalities)>2:

    

    plt.figure(figsize=(10,5))

    fcast.plot(style='--', marker='', color='green', legend=True, label='Forecast')

    fmodel.fittedvalues.plot(style='--', marker='', color='blue', legend=True, label='Smoothed')

    fatalities.plot(linestyle='', marker='.', color='red', legend=True, label='Actual values')



    keys = ['smoothing_level', 'smoothing_slope', 'damping_slope']

    alpha, beta, phi = list(map(fmodel.model.params.get, keys))

    txt_params = ('Exponential smoothing with parameters:\n\n\t' + r'$\alpha=${}'.format(alpha) + '\n\t' + 

                  r'$\beta=${}'.format(beta) + '\n\t' + r'$\phi=${}'.format(phi))



    plt.gcf().text(1, 0.5, txt_params)

    plt.title(r'Fatalities in {}'.format(state) + '\n' + '(multiplicative damped trend)')

    plt.xlabel('Day Count Fatalities')

    plt.show()
# Loop through all locations

lst_states = df.index.get_level_values(0).unique()

lst_params = df_params.index.unique()



for state in lst_states:



    # Historical curve

    mask = (df.index.get_level_values(0)==state) & (df.index.get_level_values(1)>0)

    fatalities = df.loc[mask, 'Fatalities'].reset_index(drop=True)



    # Exponential smoothing parameters 

    alpha = beta = phi = 0 # constant forecast by default

    if state in lst_params:

        alpha, beta, phi = df_params.loc[idx[state], ['Alpha', 'Beta', 'Phi']].values



    # At least two data points are needed for exponential smoothing

    if len(fatalities)>1:



        # Deal with a few inconsistencies

        if (np.min(fatalities)==0) or (not fatalities.is_monotonic):

            print('Inconsistent data identified for {}. Please check.'.format(state))

            fatalities = fatalities.apply(lambda x: max(x,1)) # at least one fatality

            # (must be strictly positive when using multiplicative trend)

        

        with warnings.catch_warnings():

            warnings.simplefilter('ignore')

            fmodel = (ExponentialSmoothing(fatalities, trend='mul', damped=True, seasonal=None)

                      .fit(smoothing_level=alpha, smoothing_slope=beta, damping_slope=phi, 

                           remove_bias=True, use_basinhopping=True))

            fcast = fmodel.forecast(fperiod)

            

        # Add forecast to the dataset

        date_start = df.loc[idx[state,:], 'Date'][-1]  + dt.timedelta(days=1)

        rng_dt = pd.date_range(start=date_start, periods=fperiod)

        

        arrays = [[state]*len(fcast.index), fcast.index]

        idx_fcast = pd.MultiIndex.from_arrays(arrays, names=('Location', 'Day Count Fatalities'))



        df = df.append(pd.DataFrame(index=idx_fcast, 

                               data={'Fatalities': fcast.values, 'Date': rng_dt}), sort=True)
df_train = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv', index_col=0)

df_test = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv', index_col=0)

df_submit = pd.read_csv('../input/covid19-global-forecasting-week-2/submission.csv', index_col=0)
df_train['Country_Region'].replace('Taiwan*', 'Taiwan', inplace=True)

df_test['Country_Region'].replace('Taiwan*', 'Taiwan', inplace=True)
df_train['Date'] = df_train['Date'].apply(lambda x: (dt.datetime.strptime(x, '%Y-%m-%d')))

df_train['Location'] = (df_train[['Province_State','Country_Region']]

                        .apply(lambda row: location(row[0],row[1]), axis=1))



df_test['Date'] = df_test['Date'].apply(lambda x: (dt.datetime.strptime(x, '%Y-%m-%d')))

df_test['Location'] = (df_test[['Province_State','Country_Region']]

                       .apply(lambda row: location(row[0],row[1]), axis=1))
# Fill in the test dataset

df_test = (df_test.reset_index()

           .merge(df_train.reset_index().loc[:,['Location','Date','ConfirmedCases','Fatalities']], 

                  how='left', on=['Location','Date'])

           .merge(df.reset_index().loc[:,['Location','Date','Fatalities']], 

                  how='left', on=['Location','Date'], suffixes=('_Train','_Forecast'))

           .set_index('ForecastId', verify_integrity=True))
# There is some overlap between the train and test timelines

df_test['Fatalities'] = df_test.loc[:,['Fatalities_Train','Fatalities_Forecast']].apply(

    lambda x: x[0] if not pd.isnull(x[0]) else x[1], axis=1)



df_test.drop(columns=['Fatalities_Train','Fatalities_Forecast'], inplace=True)
# Locations with day counts less than 1

lst_states = df_test.loc[df_test['Fatalities'].isnull(),'Location'].unique()



# Latest data available

dt_latest = df_train['Date'].max()



df_flat = (df_test

          .loc[df_test['Date']==dt_latest]

          .set_index('Location', verify_integrity=True)

          .loc[idx[lst_states]]

          .sort_values('Fatalities', ascending=True))



# Plot 

df_flat.loc[df_flat['Fatalities']>0, 'Fatalities'].plot(figsize=(10,7.5), kind='barh')

plt.title('Fatalities as of {} in locations with day counts less than 1:'.format(dt_latest))

plt.ylabel('')

plt.show()
# Flat forecast

for state in lst_states:

    mask = (df_test['Location']==state) & (df_test['Fatalities'].isnull())

    df_test.loc[mask, ['ConfirmedCases','Fatalities']] = df_flat.loc[idx[state], 

                                                                     ['ConfirmedCases','Fatalities']].values
# Get latest case fatality rates

df_train['Case Fatality Rate'] = (df_train[['Fatalities','ConfirmedCases']]

                                  .apply(lambda x: 0 if ((x[0]==0) and (x[1]==0)) else x[0]/x[1], axis=1))



df_train = df_train.set_index(['Location','Date'], verify_integrity=True).sort_index()

mask = ~df_train.index.get_level_values(0).duplicated(keep='last')



df_cfr = df_train.loc[mask,:].reset_index(level=1, drop=True).loc[:,['Case Fatality Rate']]
# Estimate confirmed cases based on fatalities and case fatality rates

mask = df_test['ConfirmedCases'].isnull()

df_test.loc[mask,'ConfirmedCases'] = (df_test

                                      .reset_index()

                                      .merge(df_cfr.reset_index(), how='left', on='Location')

                                      .set_index('ForecastId', verify_integrity=True)

                                      .loc[mask,['Fatalities','Case Fatality Rate']]

                                      .apply(lambda x: 0 if x[1]==0 else x[0]/x[1], axis=1))
# Round float values to the nearest integer

df_test['ConfirmedCases'] = df_test['ConfirmedCases'].apply(lambda x: round(x, 0)).astype('int')

df_test['Fatalities'] = df_test['Fatalities'].apply(lambda x: round(x, 0)).astype('int')
# Reset

df_test = df_test.reset_index().loc[:,['ForecastId','ConfirmedCases','Fatalities']]

df_submit = df_submit.reset_index().drop(columns=['ConfirmedCases','Fatalities'])
# Update

df_submit = df_submit.merge(df_test, how='left', on='ForecastId').set_index('ForecastId', verify_integrity=True)
# Submit

df_submit.to_csv('submission.csv', index=True)