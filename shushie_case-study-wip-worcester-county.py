# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib as mpl
import matplotlib.pylab as plt
plt.style.use('ggplot')
import os
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dept_11_pov = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_poverty/ACS_16_5YR_S1701_with_ann.csv')
dept_11_edu_25 = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_education-attainment-over-25/ACS_16_5YR_B15003_with_ann.csv')
dept_11_edu = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_education-attainment/ACS_16_5YR_S1501_with_ann.csv')
dept_11_housing = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_owner-occupied-housing/ACS_16_5YR_S2502_with_ann.csv')
dept_11_race = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_race-sex-age/ACS_15_5YR_DP05_with_ann.csv')
total_population = pd.to_numeric(dept_11_race['HC01_VC03'][1:]).sum()
total_male = pd.to_numeric(dept_11_race['HC01_VC04'][1:])
total_female = pd.to_numeric(dept_11_race['HC01_VC05'][1:])

male_percent = (total_male.sum() / total_population) * 100
female_percent = (total_female.sum() / total_population) * 100
housing_units = pd.to_numeric(dept_11_housing['HC01_EST_VC01'][1:]).sum()
below_poverty_line = pd.to_numeric(dept_11_pov['HC02_EST_VC01'][1:]).sum()
percent_below_poverty_line = (below_poverty_line / total_population) * 100
race_df = pd.DataFrame({
    'white': pd.to_numeric(dept_11_race['HC01_VC49'][1:]),
    'af_am':pd.to_numeric(dept_11_race['HC01_VC50'][1:]),
    'asian':pd.to_numeric(dept_11_race['HC01_VC56'][1:]),
    'native':pd.to_numeric(dept_11_race['HC01_VC51'][1:]),
    'hispanic': pd.to_numeric(dept_11_race['HC01_VC88'][1:]),
    'other': pd.to_numeric(dept_11_race['HC01_VC69'][1:])
}, columns=['white','af_am','asian','native', 'hispanic','other'])
fig, ax = plt.subplots(figsize=(15,7))
race_df.plot.box(ax=ax)
age_hist = pd.DataFrame({
    '< 5': pd.to_numeric(dept_11_pov['HC01_EST_VC04'][1:]),
    '5-17': pd.to_numeric(dept_11_pov['HC01_EST_VC05'][1:]),
    '18-34': pd.to_numeric(dept_11_pov['HC01_EST_VC08'][1:]),
    '35-64':pd.to_numeric(dept_11_pov['HC01_EST_VC09'][1:]),
    '65+':pd.to_numeric(dept_11_pov['HC01_EST_VC11'][1:])
}, columns=['< 5','5-17','18-34','35-64','65+'])
fig, ax = plt.subplots(figsize=(15,7))
age_hist.plot.box(ax=ax)
sex_df = pd.DataFrame({
    'total': pd.to_numeric(dept_11_race['HC01_VC03'][1:]),
    'male': pd.to_numeric(dept_11_race['HC01_VC04'][1:]),
    'female': pd.to_numeric(dept_11_race['HC01_VC05'][1:])
}, columns=['total','male','female'])
locations_df = pd.DataFrame({
    'geo_id_1':dept_11_race['GEO.id'],
    'geo_id_2':dept_11_race['GEO.id2'],
    'county': dept_11_race['GEO.display-label']
    
}, columns=['geo_id_1', 'geo_id_2','county'])
age_race = pd.concat([locations_df,sex_df,age_hist, race_df ], sort=True, join='inner', axis=1)
age_race.head() # A Cleaner dataframe to work with
