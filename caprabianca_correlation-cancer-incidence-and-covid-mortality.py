# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
datasets_dirs = [

    '/kaggle/input/uncover/UNCOVER/us_cdc/us_cdc/u-s-chronic-disease-indicators-cdi.csv',

    '/kaggle/input/uncover/UNCOVER/WHO/who-situation-reports-covid-19.csv',

    '/kaggle/input/uncover/UNCOVER/nextstrain/covid-19-genetic-phylogeny.csv',

    '/kaggle/input/uncover/UNCOVER/county_health_rankings/county_health_rankings/us-county-health-rankings-2020.csv',

    '/kaggle/input/uncover/UNCOVER/covid_19_canada_open_data_working_group/individual-level-mortality.csv',

    '/kaggle/input/uncover/UNCOVER/USAFacts/confirmed-covid-19-cases-in-us-by-state-and-county.csv',

    '/kaggle/input/uncover/UNCOVER/USAFacts/confirmed-covid-19-deaths-in-us-by-state-and-county.csv'

]



datasets = []



for datasets_dir in datasets_dirs:

    df = pd.read_csv(datasets_dir)

    datasets.append(df)
#Delete Cols with no values

del_cols = []

for column in datasets[0].columns:

    pct_na = datasets[0][column].isna().sum() / len(datasets[0][column])

    if pct_na == 1:

        del_cols.append(column)

        

datasets[0].drop(del_cols, axis=1, inplace=True)    
#We are only interested in Cancer indicators

datasets[0] = datasets[0].loc[datasets[0].topic == 'Cancer']

datasets[0].drop('topic', axis=1, inplace=True)
#For now, we're only interested in overall indicators. Not Gender nor Race/Ethnicity

datasets[0] = datasets[0].loc[datasets[0].stratificationcategory1 == 'Overall']

datasets[0].drop('stratificationcategory1', axis=1, inplace=True)
#For a first exploration, let's take all the data from South Carolina only and only the essential features

south_carolina_df = datasets[0].loc[datasets[0].locationdesc=='South Carolina']

cols_of_interest = ['yearstart', 'yearend','question', 'questionid', 'datavalueunit', 'datavaluetype', 'datavalue', ]

south_carolina_df = south_carolina_df[cols_of_interest]
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

print(south_carolina_df.question.unique())
south_carolina_df[south_carolina_df['questionid']=='CAN8_1'].head()
cancer_incidence = datasets[0][(datasets[0]['questionid']=='CAN4_1') & 

                               (datasets[0]['datavaluetype']=='Average Annual Crude Rate') & 

                               (datasets[0]['yearend']==2016)].groupby('locationabbr').datavalue.mean()
cancer_incidence_ageadj = datasets[0][(datasets[0]['questionid']=='CAN4_1') & 

                               (datasets[0]['datavaluetype']=='Average Annual Age-adjusted Rate') & 

                               (datasets[0]['yearend']==2016)].groupby('locationabbr').datavalue.mean()
#Let's also isolate lungs cancer

lung_cancer_incidence = datasets[0][(datasets[0]['questionid']=='CAN8_1') & 

                               (datasets[0]['datavaluetype']=='Average Annual Crude Rate') & 

                               (datasets[0]['yearend']==2016)].groupby('locationabbr').datavalue.mean()
#Let's also isolate lungs cancer

lung_cancer_incidence_ageadj = datasets[0][(datasets[0]['questionid']=='CAN8_1') & 

                               (datasets[0]['datavaluetype']=='Average Annual Age-adjusted Rate') & 

                               (datasets[0]['yearend']==2016)].groupby('locationabbr').datavalue.mean()
#Mortality Rate as of April 5 by state

mortality_rate = ((datasets[6][datasets[6].date=='2020-04-05'].groupby('state_name').deaths.sum()) /

                  (datasets[5][datasets[5].date=='2020-04-05'].groupby('state_name').confirmed.sum()))
states_pop = pd.read_csv('/kaggle/input/united-states-population-by-state/nst-est2019-01.csv', index_col=0)

confirmed_by_state = datasets[5][datasets[5].date=='2020-04-05'].groupby('state_name').confirmed.sum() 

temp_df = pd.concat([confirmed_by_state.sort_index(), states_pop.sort_index()], axis=1, sort=True, ignore_index=True)

temp_df.columns = ['ConfirmedCases', 'TotalPopulation']

incidence_rate = temp_df.ConfirmedCases / temp_df.TotalPopulation * 100
im_conclusion_1 = pd.concat([cancer_incidence.drop('US'), mortality_rate], axis=1)

im_conclusion_1.columns = ['cancer_incidence', 'mortality_rate']

im_conclusion_2 = pd.concat([cancer_incidence_ageadj.drop('US'), mortality_rate], axis=1)

im_conclusion_2.columns = ['cancer_incidence_ageadj', 'mortality_rate']

im_conclusion_3 = pd.concat([lung_cancer_incidence_ageadj.drop('US'), mortality_rate], axis=1)

im_conclusion_3.columns = ['lung_cancer_incidence_ageadj', 'mortality_rate']

im_conclusion_4 = pd.concat([lung_cancer_incidence.drop('US'), mortality_rate], axis=1)

im_conclusion_4.columns = ['lung_cancer_incidence', 'mortality_rate']
im_conclusion_3.plot.scatter(x='lung_cancer_incidence_ageadj',y='mortality_rate')
im_conclusion_3.corr()
ii_conclusion_1 = pd.concat([cancer_incidence.drop('US'), incidence_rate], axis=1)

ii_conclusion_1.columns = ['cancer_incidence', 'incidence_rate']

ii_conclusion_2 = pd.concat([cancer_incidence_ageadj.drop('US'), incidence_rate], axis=1)

ii_conclusion_2.columns = ['cancer_incidence_ageadj', 'incidence_rate']

ii_conclusion_3 = pd.concat([lung_cancer_incidence_ageadj.drop('US'), incidence_rate], axis=1)

ii_conclusion_3.columns = ['lung_cancer_incidence_ageadj', 'incidence_rate']

ii_conclusion_4 = pd.concat([lung_cancer_incidence.drop('US'), incidence_rate], axis=1)

ii_conclusion_4.columns = ['lung_cancer_incidence', 'incidence_rate']
ii_conclusion_1.plot.scatter(x='cancer_incidence',y='incidence_rate')

print(ii_conclusion_1.corr())
ii_conclusion_2.plot.scatter(x='cancer_incidence_ageadj',y='incidence_rate')

print(ii_conclusion_2.corr())
ii_conclusion_3.plot.scatter(x='lung_cancer_incidence_ageadj',y='incidence_rate')

print(ii_conclusion_3.corr())
ii_conclusion_4.plot.scatter(x='lung_cancer_incidence',y='incidence_rate')

print(ii_conclusion_4.corr())