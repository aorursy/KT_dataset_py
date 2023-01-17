! apt install libgeos-dev

! pip uninstall -y shapely; pip install --no-binary :all: shapely==1.6.4

! pip uninstall -y cartopy; pip install --no-binary :all: cartopy==0.17.0

! pip install geoviews==1.6.6 hvplot==0.5.2 panel==0.8.0 bokeh==1.4.0
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

from operator import add, mul

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import hvplot.pandas

import holoviews as hv

import cartopy.crs as ccrs

import geopandas as gpd

from toolz.curried import map, partial, pipe, reduce

from statsmodels.regression.linear_model import OLS

from statsmodels.tools.tools import add_constant

import matplotlib.pyplot as plt

import statsmodels.api as sm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# Any results you write to the current directory are saved as output.

hv.extension('bokeh')
countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).replace('United States of America', 'US')

covid = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv', parse_dates=['Date'], index_col='Id')

indicators = pd.read_csv('/kaggle/input/countries-of-the-world/countries of the world.csv', decimal=',').replace('United States', 'US')



country_indicators = (countries.assign(name = lambda df: df.name.astype(str).str.strip())

                     .merge(indicators.assign(Country = lambda df: df.Country.astype(str).str.strip()), 

                            left_on = 'name', right_on='Country', how='inner'))

weeks = (covid

         .assign(dayofweek = lambda df: df.Date.dt.dayofweek)

         .set_index('Date')

         .drop(columns=['Province_State'])

         .groupby(['Country_Region', pd.Grouper(freq='W')]).agg({'ConfirmedCases':'sum', 'Fatalities':'sum', 'dayofweek':'max'})

         .reset_index()

         .where(lambda df: df.ConfirmedCases > 0)

         .dropna(0)

         .groupby('Country_Region')

         .apply(lambda df: (df

                            .sort_values('Date')

                            .assign(week_of_infection = lambda df: pd.np.arange(df.shape[0]))))

         .where(lambda df: df.dayofweek >= 6)

         .drop(columns=['dayofweek'])

         .dropna(0)

         .reset_index(drop=True)

         .merge(country_indicators, left_on='Country_Region', right_on='name', how='inner')

         .pipe(lambda df: gpd.GeoDataFrame(df, geometry='geometry'))

         .assign(ConfirmedCases_per_capita = lambda df: (df.ConfirmedCases / df.pop_est),

                 Fatalities_per_capita= lambda df: (df.Fatalities / df.pop_est),

                 land_area = lambda df: df.area.astype('float'),

                 week_of_infection_exp = lambda df: df.week_of_infection.apply(np.exp))

         .groupby('Country_Region')

         .apply(lambda df: (df

                            .assign(week_since_first_death = lambda x: (x.week_of_infection - x.where(lambda y: y.Fatalities > 0)

                                                                        .week_of_infection.min())

                                                                        .clip(lower=0)

                                                                        .fillna(0))))

         .assign(week_since_first_death_exp = lambda df: df.week_since_first_death.apply(np.exp))

         .drop(columns = 'gdp_md_est'))

weeks
X, y_cases = (weeks

     .select_dtypes(include=['number'])

     .drop(columns=['ConfirmedCases', 'Fatalities', 'ConfirmedCases_per_capita', 'Fatalities_per_capita'])

     .replace(0, 1e-8)# add jitter

     .transform(np.log)

     .pipe(lambda df: df.fillna(df.mean()))

     .rename(columns = lambda name: '%Δ ' + name)

     .rename(columns = {'%Δ week_of_infection_exp': 'week_of_infection'})

     .rename(columns = {'%Δ week_since_first_death_exp': 'week_since_first_death'})

     .pipe(lambda df: pd.concat([df, pd.get_dummies(weeks.name, drop_first=True).rename(columns =lambda s: 'is_'+s)], axis=1))

     .assign(const = 1),

        

    weeks

    .loc[:, ['ConfirmedCases_per_capita']]

    .rename(columns={'ConfirmedCases_per_capita': 'Cases/capita'})

    .replace(0, 1e-8)# add jitter

    .transform(np.log)  

    .rename(columns = lambda name: '%Δ ' + name)

    )



X.head()
y_cases.hvplot.kde(title='Kernel Density Estimation of %Δ Confirmed Cases Response')
def stepwise_selection(X, y, 

                       initial_list=[], 

                       threshold_in=0.015, 

                       threshold_out = 0.05, 

                       verbose=True):

    """ Perform a forward-backward feature selection 

    based on p-value from statsmodels.api.OLS

    Arguments:

        X - pandas.DataFrame with candidate features

        y - list-like with the target

        initial_list - list of features to start with (column names of X)

        threshold_in - include a feature if its p-value < threshold_in

        threshold_out - exclude a feature if its p-value > threshold_out

        verbose - whether to print the sequence of inclusions and exclusions

    Returns: list of selected features 

    Always set threshold_in < threshold_out to avoid infinite looping.

    See https://en.wikipedia.org/wiki/Stepwise_regression for the details

    """

    included = list(initial_list)

    while True:

        changed=False

        # forward step

        excluded = list(set(X.columns)-set(included))

        new_pval = pd.Series(index=excluded)

        for new_column in excluded:

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:

            best_feature = new_pval.idxmin()

            included.append(best_feature)

            changed=True

            if verbose:

                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))



        # backward step

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()

        # use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        worst_pval = pvalues.max() # null if pvalues is empty

        if worst_pval > threshold_out:

            changed=True

            worst_feature = pvalues.argmax()

            included.remove(worst_feature)

            if verbose:

                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:

            break

    return included



params_cases = stepwise_selection(X.loc[:, np.random.permutation(X.columns)], y_cases, threshold_in=0.015)



model_cases = OLS(y_cases, X.loc[:, params_cases])

results_cases = model_cases.fit()
results_cases.summary()
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(17, 5))

fig = sm.graphics.influence_plot(results_cases, ax=axes[0], criterion="cooks")

fig = sm.graphics.plot_leverage_resid2(results_cases, ax=axes[1])

res = results_cases.resid # residuals

fig = sm.qqplot(res, ax=axes[2])

fig.tight_layout()

fig = plt.figure(figsize=(30,60))

sm.graphics.plot_partregress_grid(results_cases, fig=fig)
y_fatalities = (weeks

    .loc[:, ['Fatalities_per_capita']]

    .rename(columns={'Fatalities_per_capita': 'Fatalities/capita'})

    .replace(0, 1e-8)# add jitter

    .transform(np.log)  

    .rename(columns = lambda name: '%Δ ' + name))



y_fatalities.hvplot.kde(title='Kernel Density Estimation of %Δ Fatalities Response')
params_fatalities = stepwise_selection(X.loc[:, np.random.permutation(X.columns)], y_fatalities,  threshold_in=0.025)



model_fatalities = OLS(y_fatalities, X.loc[:, params_fatalities])

results_fatalities = model_fatalities.fit()
results_fatalities.summary()
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(17, 5))

fig = sm.graphics.influence_plot(results_fatalities, ax=axes[0], criterion="cooks")

fig = sm.graphics.plot_leverage_resid2(results_fatalities, ax=axes[1])

res = results_fatalities.resid # residuals

fig = sm.qqplot(res, ax=axes[2])

fig.tight_layout()

fig = plt.figure(figsize=(30,60))

sm.graphics.plot_partregress_grid(results_fatalities, fig=fig)
(pd.concat([results_cases.params.to_frame(name='Coefficient').assign(Response = '%Δ Cases/capita'),

            results_fatalities.params.to_frame(name='Coefficient').assign(Response = '%Δ Fatalities/capita')], axis=0)

 .drop(index=['const'])

 .reset_index().rename(columns={'index': 'Covariate'})

 .where(lambda s: ~s.Covariate.str.startswith('is_')).dropna().set_index('Covariate')

 .hvplot.bar(title='COVID-19: Coefficients on (%Δ) Covariate against (%Δ) Response', by='Response', rot=90)

 .opts(width=1200, height=400))
coefficients = (results_cases.params.to_frame('Cases')

                .join(results_fatalities.params.to_frame('Fatalities'), how='outer')

                .fillna(0))
formula = (coefficients

 .Cases

 .loc[results_fatalities.params.index]

 .reset_index()

 .rename(columns={'index':'Name'})

 .assign(formula = lambda df: df.Name.astype(str) + ' = ' + df.Cases.astype(str) + ' ,')

 .formula

 .sum())[:-1]



T_test = results_fatalities.t_test(formula)

T_test.summary_frame().assign(names = model_fatalities.exog_names).set_index('names').round(3)
formula = (coefficients

 .Fatalities

 .loc[results_cases.params.index]

 .reset_index()

 .rename(columns={'index':'Name'})

 .assign(formula = lambda df: df.Name.astype(str) + ' = ' + df.Fatalities.astype(str) + ' ,')

 .formula

 .sum())[:-1]



T_test = results_cases.t_test(formula)

T_test.summary_frame().assign(names = model_cases.exog_names).set_index('names').round(3)