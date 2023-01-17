import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from pandas_profiling import ProfileReport 
world_atlas_df = pd.read_csv('/kaggle/input/hackathon/task_2-BCG_world_atlas_data-bcg_strain-7July2020.csv', delimiter=',')
world_atlas_df.head()
country_df = pd.read_csv('/kaggle/input/hackathon/BCG_country_data.csv', delimiter=',')
country_df.head()
country_and_bcg_atlas_df = pd.merge(world_atlas_df, country_df, left_on='country_code',right_on='alpha_2_code',how='outer',suffixes=('_left','_right'))
country_and_bcg_atlas_df
covid_deaths_df = pd.read_csv('/kaggle/input/hackathon/task_2-COVID-19-death_cases_per_country_after_fifth_death-till_26_June.csv', delimiter=',')
covid_deaths_df.head()
country_bcg_atlas_and_covid_deaths_df = pd.merge(country_and_bcg_atlas_df, covid_deaths_df, left_on='alpha_3_code',right_on='alpha_3_code',how='outer',suffixes=('_left','_right'))
country_bcg_atlas_and_covid_deaths_df
report_df = ProfileReport(country_bcg_atlas_and_covid_deaths_df)
report_df