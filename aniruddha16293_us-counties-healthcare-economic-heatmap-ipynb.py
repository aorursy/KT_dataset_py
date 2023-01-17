# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # visualization

import matplotlib.pyplot as plt # visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load the counties.csv file into a dataframe

input_dir = '../input/covid19-us-countylevel-summaries/'

df = pd.read_csv(os.path.join(input_dir, 'counties.csv'))
# A correlation heatmap for socioeconomic factors on US counties



socio_economic_df = df[['Less than a high school diploma 2014-18',

 'High school diploma only 2014-18',

 "Some college or associate's degree 2014-18",

 "Bachelor's degree or higher 2014-18",

 'Percent of adults with less than a high school diploma 2014-18',

 'Percent of adults with a high school diploma only 2014-18',

 "Percent of adults completing some college or associate's degree 2014-18",

 "Percent of adults with a bachelor's degree or higher 2014-18",

 'POVALL_2018',

 'CI90LBAll_2018',

 'CI90UBALL_2018',

 'PCTPOVALL_2018',

 'CI90LBALLP_2018',

 'CI90UBALLP_2018',

 'POV017_2018',

 'CI90LB017_2018',

 'CI90UB017_2018',

 'PCTPOV017_2018',

 'CI90LB017P_2018',

 'CI90UB017P_2018',

 'POV517_2018',

 'CI90LB517_2018',

 'CI90UB517_2018',

 'PCTPOV517_2018',

 'CI90LB517P_2018',

 'CI90UB517P_2018',

 'MEDHHINC_2018',

 'CI90LBINC_2018',

 'CI90UBINC_2018',

 'Civilian_labor_force_2018',

 'Employed_2018',

 'Unemployed_2018',

 'Unemployment_rate_2018',

 'Median_Household_Income_2018',

 'Med_HH_Income_Percent_of_State_Total_2018']]



corr = socio_economic_df.corr()

cmap = sns.diverging_palette(220, 10, as_cmap=True)

mask = np.triu(np.ones_like(corr, dtype=np.bool))

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# A correlation heatmap for economic-healthcare factors on US counties



economic_health_df = df[['POVALL_2018',

 'MEDHHINC_2018',

 'Median_Household_Income_2018',

 'Med_HH_Income_Percent_of_State_Total_2018',

 'Active Physicians per 100000 Population 2018 (AAMC)',

 'Total Active Patient Care Physicians per 100000 Population 2018 (AAMC)',

 'Active Primary Care Physicians per 100000 Population 2018 (AAMC)',

 'Active Patient Care Primary Care Physicians per 100000 Population 2018 (AAMC)',

 'Active General Surgeons per 100000 Population 2018 (AAMC)',

 'Active Patient Care General Surgeons per 100000 Population 2018 (AAMC)',

 'Fraction of Active Physicians Who Are Female 2018 (AAMC)',

 'Fraction of Active Physicians Who Are International Medical Graduates (IMGs) 2018 (AAMC)']]



corr = economic_health_df.corr()

cmap = sns.diverging_palette(220, 10, as_cmap=True)

mask = np.triu(np.ones_like(corr, dtype=np.bool))

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})