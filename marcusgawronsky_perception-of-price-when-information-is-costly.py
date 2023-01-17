! pip install --upgrade pip

! pip install hvplot
# load imports and set extension

from bisect import bisect

from functools import reduce

from operator import add



import holoviews as hv

import hvplot.pandas

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import probplot

from statsmodels.regression.linear_model import OLS





hv.extension('bokeh')
# Set seed

pd.np.random.seed(0)



FILTER = 0.5 # filter for stratified sampling groups with little coverage

FRAC = 0.1 # subsample the data



# Filter for domestic dwellings, "SOCIAL": 0 excluded as different schedule

INCOME = {

          "DOMESTICO BAJA": 0,

          "DOMESTICO MEDIO": 1,

          "DOMESTICO RESIDENCIAL": 2,

          }



# Spanish Month Mappings

ABBREVIATIONS = {'ENE': 0,

               'FEB': 1,

               'JUL': 2,

               'JUN': 3,

               'MAR': 4,

               'MAY': 5,

               'NOV': 6,

               'OCT': 7,

               'SEP': 8,

               'ABR': 9,

               'AGO': 10,

               'DIC': 11}





# Proxy block-tarrif structure

# http://aguadehermosillo.gob.mx/aguah/tarifas/ : accessed 6 March 2020

RATES = [0, 8.57, 11.09, 11.09, 11.09, 11.27, 17.17, 17.17, 17.17, 54.01, 54.01, 54.01, 54.96, 54.96, 59.09]   

BRACKETS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 56, 70, 75]

BASE_RATE = 86.11
# Tarrif calculator

base_rate = np.multiply(np.diff([0, *BRACKETS]), 

                        RATES[:-1]).cumsum()

def block_pricing(units):

    i = bisect(BRACKETS, units)

    if not i:

        return 0

    rate = RATES[i]

    bracket = BRACKETS[i-1]

    tarrif_in_bracket = units - bracket

    tarrif_in_bracket = tarrif_in_bracket * rate

    total_tarrif = base_rate[i-1] + tarrif_in_bracket

    return total_tarrif + BASE_RATE



def marginal_rate(units):

    i = bisect(BRACKETS, units)

    

    return RATES[i]



def block(units):

    i = bisect(BRACKETS, units)

    

    return i
data = pd.read_csv('/kaggle/input/water-consumption-in-a-median-size-city/AguaH.csv')



resample_after_dropna = (data

                         .groupby(['TU', 'USO2013'])

                         .apply(lambda df: 1- (df

                                            .isna()

                                            .any(1)

                                            .mean()))

                         .reset_index()

                         .rename(columns={0:'resampling'}))

        

        

data = (data

        .dropna()

        .merge(resample_after_dropna, on=['TU', 'USO2013'])

        .sample(frac=FRAC, weights='resampling'))



date_mapping = {date: i for i, date in enumerate(data.columns[data.columns.str.startswith('f.1_')])}

reverse_date_mapping = {i: date for date, i in date_mapping.items()}

reverse_year_mapping = {i: 2000 + int(d[-2:]) for d, i in date_mapping.items()}

reverse_month_mapping = {i: ABBREVIATIONS[date.split('_')[1]] for date, i in date_mapping.items()}
filter_groups = (data

 .where(lambda x: x.TU.isin(list(INCOME.keys())))

 .groupby(['TU', 'USO2013'])['M'].count())



filter_groups
filter_groups = (filter_groups

                 .reset_index()

                 .where(lambda df: df.M > df.M.quantile(FILTER))

                 .rename(columns={'M': 'filter'}))
tidy = (data

        .where(lambda x: x.TU.isin(list(INCOME.keys()))) # filter for domestic users

        .merge(filter_groups, on=['TU', 'USO2013'], how='left').dropna().drop(columns=['filter', 

                                                                                       'resampling'])

         .merge(filter_groups, on=['TU', 'USO2013'], how='left').dropna().drop(columns=['filter'])

#             .groupby(['TU', 'USO2013'])

#             .apply(lambda df: df.sample(frac=FRAC)) # stratified sample

#             .reset_index(drop=True) # stratified sampling of domestic user

            .dropna()

        .reset_index()

        .rename(columns=dict(**date_mapping,

                             **{'index':'user'}))

        .melt(id_vars=['user','USO2013','TU','DC','M','UL'],

              var_name='date',

              value_name='quantity') # melt measurement timeseries

        .assign(month = lambda df: df.date.replace(reverse_month_mapping), # get month

                year = lambda df: df.date.replace(reverse_year_mapping), # get year

                income = lambda df: df.TU.replace(INCOME), # get pseudo income

                block = lambda df: df.quantity.apply(block),

                tarrif = lambda df: df.quantity.apply(block_pricing), # get tarrif

                marginal =  lambda df: df.quantity.apply(marginal_rate)) # get marginal rate

        .assign(average = lambda df: df.tarrif / df.quantity) # compute average rate

        .dropna())
# get quantity at lag

tidy = tidy.merge((tidy

                   .loc[:,['date','user','quantity']]

                   .assign(date = lambda df: df.date + 1,

                           quantitylag = lambda df: df.quantity)

                   .drop(columns=['quantity'])), on=['date','user'], how='left').dropna()



tidy.head()
# Correlation matrix

(tidy

 .loc[:,['quantity','marginal','average', 'income', 'quantitylag', 'year']]

 .corr())
# construct shin design matrix

d_matrix = (tidy

            .loc[:,['quantity','marginal','average','income','quantitylag', 'month', 'year', 'block']]

            .assign(marginal_over_average = lambda df: df.marginal / df.average)

            .assign(winter = lambda df: df.month.apply(lambda x: (np.cos(2 * np.pi * (x/11))))

                                                       .add(1) # shift the cosine up

                                                       .divide(2) # make between 0 and 1

                                                       .add(1e-6) # add jitter for log transform

                                                      )

            .assign(block = lambda df: df.block.apply(pd.np.exp))

            .assign(year = lambda df: df.year - df.year.min() + 1e-6)

            .drop(columns=['average', 'month'])

            .transform(np.log)

            .assign(bias = 1)

            .replace({np.inf: np.nan, -np.inf: np.nan})

            .dropna())



block_pricing_mixed_effects = pd.get_dummies(d_matrix['block'].astype(str) 

                                             + '_' 

                                             + d_matrix['year'].apply(np.exp).add(tidy.year.min() - 1e-6).astype(str), 

                                             prefix = 'block')



landuse_indicators = pd.get_dummies(tidy['USO2013'],

                                    prefix = 'landuse').drop(columns=['landuse_MX'])





d_matrix = (pd.concat([d_matrix.drop(columns=['block']),

                      block_pricing_mixed_effects], axis=1)

            .join(landuse_indicators))



d_matrix
d_matrix.iloc[:, :7].corr()
# do regression analysis

X, Y = d_matrix.drop(columns=['quantity']), d_matrix.loc[:,['quantity']]

model = OLS(Y,X)

results = model.fit()



results.summary()
# plot errors distribution

residuals = results.predict(X).rename('errors').subtract(Y.quantity)



(((residuals

 .hvplot.kde(label='Residuals', xlabel='residuals'))) *

(pd.Series(np.random.normal(0,residuals.std(),size=(1000)))

 .hvplot.kde(label='Distribution of Centred Normal', xlabel='epsilon')))
## qq plot

theoretical_quantiles, sample_quantiles = probplot(residuals / residuals.std())[0]



(hv.Curve([[-2.5, -2.5], [2.5,2.5]]).opts(line_width=1) *

 pd.DataFrame({'theoretical_quantiles': theoretical_quantiles, 'sample_quantiles': sample_quantiles})

 .sample(frac=0.005)

 .hvplot.scatter(x='theoretical_quantiles', y='sample_quantiles', title='QQ Plot', 

                 xlabel = 'Theoretical Quantiles', ylabel = 'Sample Quantiles',

                 size=1))
# plot heteroskedasticity

(reduce(add, [(X

               .loc[:,[col]]

               .assign(residuals = residuals)

               .dropna()

               .hvplot.scatter(y='residuals', x=col, datashade=True, width=350, height=250)) 

              for col in X.columns 

              if not (col.startswith('block') or 

                      col.startswith('bias') or 

                      col.startswith('landuse'))])

 .cols(2)

 .opts(title="Heteroskedasticity"))