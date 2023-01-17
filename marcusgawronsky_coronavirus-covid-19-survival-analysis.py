# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from statsmodels.duration.hazard_regression import PHReg

from statsmodels.graphics import regressionplots

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
ages = ['0s','10s','20s','30s','40s','50s','60s','70s','80s','90s','100s']

patientinfo = pd.read_csv('/kaggle/input/coronavirusdataset/PatientInfo.csv',

            index_col='patient_id',

            dtype={'patient_id': pd.Int64Dtype(),

                   'global_num': pd.Int64Dtype(),  'sex': pd.CategoricalDtype(), 'birth_year': pd.Int64Dtype(), 

                   'age': pd.CategoricalDtype(categories=ages, ordered=True),

                   'country': pd.CategoricalDtype(), 'province': pd.CategoricalDtype(), 'city': pd.CategoricalDtype(),

                   'disease': pd.CategoricalDtype(), 'infection_case':  pd.CategoricalDtype(),'infected_by': pd.Int64Dtype(),

                   'contact_number': pd.Int64Dtype(), 

                   'state': pd.CategoricalDtype()},

            parse_dates =['symptom_onset_date', 'confirmed_date', 'released_date', 'deceased_date'])



patientinfo
region = pd.read_csv('/kaggle/input/coronavirusdataset/Region.csv',

                     dtype={'province': pd.CategoricalDtype(), 'city': pd.CategoricalDtype()})

region
data = patientinfo.merge(region, on=['province', 'city'])

data
LIFE_EXPECTANCY = 83 * 365

X, y, died, started = (pd.get_dummies(data

                         .loc[:,['sex', 'birth_year', 'country',

                                 'elderly_population_ratio'

                                 ]]

                         .assign(country = lambda df: df.country.cat.remove_categories(['Mongolia', 'Thailand']))

                         .assign(age = lambda df: 2020 - df.birth_year)

                         .drop(columns='birth_year'), columns = ['sex', 'country'], drop_first=True)

                         .assign(age_squared = lambda df: (df.age - df.age.median()).clip(lower=0).pow(2))

                         .astype(np.float)

                         .pipe(lambda x: x.fillna(x.mean())),

                       

                       (data

                        .loc[:, ['symptom_onset_date', 'confirmed_date', 'released_date', 'deceased_date', 'birth_year']]

                        .assign(age = lambda df: 2020 - df.birth_year)

                        .assign(days_to_exit_study = lambda df: (df.loc[:, ['symptom_onset_date', 'confirmed_date', 'released_date', 'deceased_date']].max(1)

                                                           - df.loc[:, ['symptom_onset_date', 'confirmed_date', 'released_date', 'deceased_date']].min(1).min()).dt.days)

                        .assign(days_life_expectancy = lambda df: df.days_to_exit_study.where(df.released_date.isna(), df.days_to_exit_study + (LIFE_EXPECTANCY - df.age.where(lambda x: x < LIFE_EXPECTANCY,  LIFE_EXPECTANCY))* 365))

                        .loc[:, 'days_life_expectancy']

                        .astype(np.float)),

                       

                         (~data.deceased_date.isna().to_numpy()),

                       

                         (data.confirmed_date - data.confirmed_date.min()).dt.days.to_numpy())



# design matrix

X.corr()
vif = pd.DataFrame([variance_inflation_factor(X.to_numpy(), i) for i in range(X.shape[1])], columns=['VIF'], index = X.columns)

vif
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.decomposition import PCA



pipeline = make_pipeline(StandardScaler(), PCA(2))

Z = pipeline.fit_transform(X.fillna(X.mean()))



evr = pipeline.named_steps['pca'].explained_variance_ratio_

columns = [f'Component {i} ({round(e*100)}%)' for i, e in enumerate(evr)]



died  = pd.Series(died)

released = ~data.released_date.isna()



components = pd.DataFrame(Z, columns=columns)



ax = (components

 .where(~released & ~died).dropna()

 .plot.scatter(x=columns[0], y=columns[1], c='LightGrey',

               label='Hospital', title='Principle Components',

               alpha=0.25))

ax = (components

      .where(died).dropna()

      .plot.scatter(x=columns[0], y=columns[1], c='Red', label='Dead', ax=ax))

plot = (components

      .where(released).dropna()

      .plot.scatter(x=columns[0], y=columns[1], c='LightGreen', label='Released', ax=ax))

plot
model = PHReg(endog=y, exog=X, status=died.astype(np.int), entry=started, ties="efron")

model_data = pd.DataFrame(model.exog, columns=model.exog_names)



results = model.fit()

results.summary()
(pd.DataFrame(np.exp(results.params), index=model.exog_names, columns=['Hazard Ratio'])

 .plot.bar(title='Times the Chance of Fatality after Case Confirmed - Not controlling for preexisting condition'))
bch = results.baseline_cumulative_hazard

bch = bch[0] # Only one stratum here

time, cumhaz, surv = tuple(bch)

plt.clf()

plt.plot(time, cumhaz, '-o', alpha=0.6)

plt.grid(True)

plt.xlabel("Days")

plt.ylabel("Cumulative hazard")

plt.title("Cumulative hazard against Days")
plt.plot(model_data.age, results.martingale_residuals, 'o', alpha=0.5)

plt.xlabel("age")

plt.ylabel("Martingale residual")

plt.title('Martingale residual against age')
sr = results.schoenfeld_residuals

col = np.argwhere(np.array(model.exog_names) == 'age').item()

ii = np.flatnonzero(pd.notnull(sr[:,col]))



plt.plot(model.endog[ii], sr[ii,col], 'o')

plt.xlabel("Days")

plt.ylabel("Age Schoenfeld residual")