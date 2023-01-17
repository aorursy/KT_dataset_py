!pip install pycaret
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.offline as py

import plotly.graph_objs as go

import plotly.express as px

import seaborn as sns

import warnings

from pandas_profiling import ProfileReport 

from pycaret.regression import *





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/hackathon/task_2-owid_covid_data-21_June_2020.csv')

df.head()
report_df = ProfileReport(df)

report_df
# Numerical features

Numerical_feat = [feature for feature in df.columns if df[feature].dtypes != 'O']

print('Total numerical features: ', len(Numerical_feat))

print('\nNumerical Features: ', Numerical_feat)
index_int_float = ['aged_65_older', 'aged_70_older', 'total_tests', 'cvd_death_rate', 'diabetes_prevalence', 'extreme_poverty', 'female_smokers', 'gdp_per_capita', 'handwashing_facilities', 'hospital_beds_per_thousand', 'life_expectancy', 'male_smokers', 'new_cases', 'new_deaths_per_million', 'new_tests_smoothed_per_thousand', 'new_tests_smoothed', 'new_tests', 'new_tests_per_thousand', 'population', 'population_density', 'stringency_index', 'total_cases_per_million', 'new_cases_per_million', 'total_deaths_per_million', 'total_tests_per_thousand', 'new_tests_per_thousand', 'total_deaths', 'new_deaths', 'median_age']      



plt.figure(figsize=[20,12])

i = 1

for col in index_int_float :

    plt.subplot(4,10,i)

    sns.violinplot(x=col, data= df, orient='v')

    sns.despine()

    i = i+1

plt.tight_layout()

plt.show()
# categorical features

categorical_feat = [feature for feature in df.columns if df[feature].dtypes=='O']

print('Total categorical features: ', len(categorical_feat))

print('\n',categorical_feat)
index_str = ['iso_code', 'continent', 'location', 'date', 'tests_units']



plt.figure(figsize=[30,10])

i = 1

for col in index_str :

    plt.subplot(4,10,i)

    sns.scatterplot(x=col, y = 'total_tests' ,data= df)

    sns.despine()

    i = i+1

plt.tight_layout()

plt.show()
df.dtypes
int_features = ['population']

        

float_features = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'total_cases_per_million', 'new_cases_per_million', 'total_deaths_per_million', 'new_deaths_per_million', 'new_tests', 'total_tests_per_thousand', 'new_tests_per_thousand', 'new_tests_smoothed', 'new_tests_smoothed_per_thousand', 'stringency_index', 'population_density', 'median_age', 'aged_65_older', 'aged_70_older', 'gdp_per_capita', 'extreme_poverty', 'cvd_death_rate', 'diabetes_prevalence', 'female_smokers', 'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand', 'life_expectancy']



obj_features = ['iso_code', 'continent', 'location', 'date', 'tests_units']



exp_reg = setup(df, #Train Data

                target = 'total_tests',  #Target

                categorical_features = obj_features, # Categorical Features

                numeric_features = int_features + float_features, # Numeric Features

                normalize = True, # Normalize Dataset

                remove_outliers = True, # Remove 5% Outliers

                remove_multicollinearity = True, # Remove Multicollinearity

                silent = True # Process Automation

               )
compare_models(blacklist = ['tr', 'catboost'], sort = 'RMSLE')
model_br = create_model('br')

model_lightgbm = create_model('lightgbm')

model_xgboost = create_model('xgboost')

model_ridge = create_model('ridge')
tuned_br = tune_model('br')

tuned_lightgbm = tune_model('lightgbm')

tuned_xgboost = tune_model('xgboost')

tuned_ridge = tune_model('ridge')
plot_model(tuned_br, plot = 'learning')
plot_model(tuned_lightgbm, plot = 'learning')
plot_model(tuned_xgboost, plot = 'learning')
plot_model(tuned_ridge, plot = 'learning')
blender = blend_models(estimator_list = [tuned_br, tuned_lightgbm, tuned_xgboost, tuned_ridge])
display(plot_model(blender, plot = 'learning'))
predictions = predict_model(blender, data = df)

df['total_tests'] = np.expm1(predictions['Label'])

df.to_csv('submission.csv',index=False)