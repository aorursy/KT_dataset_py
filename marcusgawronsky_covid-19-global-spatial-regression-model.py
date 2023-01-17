! apt install libgeos-dev

! pip uninstall -y shapely; pip install --no-binary :all: shapely==1.6.4

! pip uninstall -y cartopy; pip install --no-binary :all: cartopy==0.17.0

! conda install -y geoviews=1.6.6 hvplot=0.5.2 bokeh==1.4.0 pysal
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

from operator import add, mul

from time import time

from functools import wraps

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

import pysal as ps

from sklearn.manifold import MDS, smacof

from sklearn.metrics.pairwise import euclidean_distances



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# Any results you write to the current directory are saved as output.

hv.extension('bokeh')
def timing(f):

    @wraps(f)

    def wrap(*args, **kw):

        ts = time()

        result = f(*args, **kw)

        te = time()

        print('func:%r took: %2.4f sec' % \

          (f.__name__, te-ts))

        return result

    return wrap



def shape(f, outputs=True, inputs=True):

    @wraps(f)

    def wrap(*args, **kw):

        ts = time()

        result = f(*args, **kw)

        te = time()

        

        if inputs:

            print('func:%r input shape:%r' % \

              (f.__name__, args[0].shape))

            

        if outputs:

            print('func:%r output shape:%r' % \

              (f.__name__, result.shape))

        return result

    return wrap
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
points = (weeks.geometry

          .representative_point()

          .apply(lambda df: pd.Series([df.x, df.y]))

          .rename(columns={0:'x',1:'y'}))

hv.Labels(points.assign(names = weeks.name).drop_duplicates(), kdims=['x','y'], vdims='names').opts(height=500, width=800, title='Midpoints of Countries')
distances = euclidean_distances(points)

D = pd.DataFrame(distances, index = weeks.name, columns = weeks.name).drop_duplicates().T.drop_duplicates()

Z, score = smacof(D, n_components=2)



(hv.Labels(pd.DataFrame(Z, columns=['x','y'])

           .assign(names = D.index)

           .drop_duplicates(), kdims=['x','y'], vdims='names').opts(height=500, width=800))
X, y_fatalities = (weeks

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

    .loc[:, ['Fatalities_per_capita']]

    .rename(columns={'Fatalities_per_capita': 'Fatalities/capita'})

    .replace(0, 1e-8)# add jitter

    .transform(np.log)  

    .rename(columns = lambda name: '%Δ ' + name)

    )



X.head()
def stepwise_selection(X, y, 

                       initial_list=[], 

                       n_iter = 1,

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

    for _ in range(n_iter):

        while True:

            changed=False

            # forward step

            excluded = list(set(X.columns)-set(included))

            new_pval = pd.Series(index=excluded).sample(frac=1)

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

                worst_feature = pvalues.idxmax()

                included.remove(worst_feature)

                if verbose:

                    print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

            if not changed:

                break

    return included



params_cases = stepwise_selection(X.loc[:, np.random.permutation(X.columns)], y_fatalities,

                                  n_iter=3,

                                  threshold_in=0.015, threshold_out=0.025)



model_cases = OLS(y_fatalities, X.loc[:, params_cases])

results_cases = model_cases.fit()
results_cases.summary()
X_prime = X.loc[:,params_cases]

X_prime = X_prime.loc[:, X_prime.min(0) != X_prime.max(0)]

w = ps.lib.weights.full2W(distances)



model = ps.model.spreg.OLS(y_fatalities.values, X_prime.values, w=w, spat_diag=True, name_x=X_prime.columns.tolist(), name_y=y_fatalities.columns.tolist()[0])

print(model.summary)
from sklearn.pipeline import make_pipeline

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, PowerTransformer



X_prime_all = X.loc[:, ~X.columns.str.startswith('is_')].drop(columns=['const'])

X_prime_all = X_prime_all.loc[:, X_prime_all.min(0) != X_prime_all.max(0)]



pipeline = make_pipeline(PowerTransformer(), StandardScaler(), PCA(7))

Z = pd.concat([pd.DataFrame(pipeline.fit_transform(X_prime_all)), X_prime.loc[:, X_prime.columns.str.startswith('is_')]], axis=1)



(hv.Bars(pipeline.named_steps['pca'].explained_variance_ratio_)

.opts(title='Variance Explained Ratio of Princple Components', ylabel='Variance Explained', xlabel='Component', width=600))
pcr_model = ps.model.spreg.OLS(y_fatalities.values, Z.values, w=w, spat_diag=True, name_x=Z.columns.tolist(), name_y=y_fatalities.columns.tolist()[0])

print(pcr_model.summary)
betas = np.array(pcr_model.betas[1:-Z.columns.astype(str).str.startswith('is').sum()])

coef_ = pipeline.named_steps['pca'].inverse_transform(betas.reshape(1,-1))

(pd.DataFrame(coef_.flatten(), index = X_prime_all.columns, columns=['Reprojected Coefficients'])

 .hvplot.bar().opts(xrotation=90, height=400))