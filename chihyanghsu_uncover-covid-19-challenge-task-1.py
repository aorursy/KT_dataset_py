# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
csv_list = []

for dirname, _, filenames in os.walk('/kaggle/input/uncover/UNCOVER_v4'):

    for filename in filenames:

        input_file = os.path.join(dirname, filename)

        ext = os.path.splitext(input_file)[-1]

        if ext == '.csv':

            csv_list.append(input_file)

            print(input_file)

            df = pd.read_csv(input_file, low_memory=False)

            print(df.columns)
ont_df = pd.read_csv('/kaggle/input/uncover/UNCOVER_v4/UNCOVER/ontario_government/confirmed-positive-cases-of-covid-19-in-ontario.csv')

print('Number of Cases:', len(ont_df))

print(ont_df.head())
ont_df['age_group'].value_counts().plot(kind='barh', title='Age')
ont_df['client_gender'].value_counts().plot(kind='barh', title='Gender')
wor_df = pd.read_csv('/kaggle/input/uncover/UNCOVER_v4/UNCOVER/worldometer/worldometer-confirmed-cases-and-deaths-by-country-territory-or-conveyance.csv')

print('Number of Countries:', len(wor_df))

print(wor_df.head())
bool_country = wor_df['sl_no'] > 0

print(wor_df[bool_country].head())
bool_tot_cases_1m = wor_df['total_cases_per_1m_pop'] >= 0

bool_tot_test_1m = wor_df['total_tests_per_1m_pop'] >= 0





wor_df['percent_confirmed'] = wor_df['total_cases_per_1m_pop'].div(wor_df['total_tests_per_1m_pop'])

bool_le = wor_df['total_cases_per_1m_pop'].le(wor_df['total_tests_per_1m_pop'])



wor_df[bool_country & bool_tot_cases_1m & bool_tot_test_1m & bool_le].sort_values(by='percent_confirmed',axis=0).plot.barh(x='country', y=['total_cases_per_1m_pop', 'total_tests_per_1m_pop', 'percent_confirmed'],

                                                                      figsize=(10,100), logx=True, title='Total Cases/ Test Cases (1M) (log)').legend(loc='upper right')
our_df = pd.read_csv('/kaggle/input/uncover/UNCOVER_v4/UNCOVER/our_world_in_data/coronavirus-disease-covid-19-statistics-and-research (1).csv')

print(our_df.head())

countries = our_df['location'].unique()

print(countries)

print('Number of Countries:', len(countries))
#bool_loc = our_df['location'] == 'World'

#print(our_df[bool_loc])



bool_date = our_df['date'] == '2020-05-21'

#print(our_df[bool_date])



our_df_0521 = our_df[bool_date]



bool_all_nan = our_df_0521.isna().all()

all_nan_features = our_df.columns[bool_all_nan]

our_df_0521 = our_df_0521.drop(labels=all_nan_features, axis=1)



bool_nan = our_df_0521.isna().any()

nan_features = our_df_0521.columns[bool_nan]



for feature in nan_features:

    mean_val = our_df_0521[feature].mean()

    our_df_0521[feature] = our_df_0521[feature].fillna(mean_val)



print(our_df_0521.isna().any())
from sklearn.feature_selection import VarianceThreshold

from scipy.stats import zscore



features = our_df_0521[['stringency_index', 'population_density', 'median_age', 'aged_65_older', 'aged_70_older',

                        'gdp_per_capita', 'extreme_poverty', 'cvd_death_rate', 'diabetes_prevalence', 'female_smokers',

                        'male_smokers', 'handwashing_facilities', 'hospital_beds_per_100k']]

target = our_df_0521['total_cases_per_million']



features_zscore = pd.DataFrame()

for column in features.columns:

    z = (features[column]-features[column].mean())/features[column].std(ddof=0)

    features_zscore[column] = z

    our_df_0521[column+'_z'] = z

    

target_zscore = (target-target.mean())/target.std(ddof=0)

our_df_0521['total_cases_per_million_z'] = target_zscore

# Variance Threshold

#sel = VarianceThreshold(threshold=(.8 * (1 - .8)))

#sel.fit(features)

#print(features.columns[sel.get_support(indices=True)])



# Univariate

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression



sel = SelectKBest(f_regression, k=5)

sel.fit(features_zscore, target_zscore)

print(features.columns[sel.get_support(indices=True)])



# LASSO

from sklearn.linear_model import LassoCV

clf = LassoCV().fit(features_zscore, target_zscore)

importance = np.abs(clf.coef_)

print(clf.coef_)

print(features.columns[importance>0])



our_df_0521.plot.barh(figsize=(5,100), x='location', y=['aged_65_older_z', 'gdp_per_capita_z', 'total_cases_per_million_z'])