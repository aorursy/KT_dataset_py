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
df = pd.read_excel('/kaggle/input/water-resource-project-data/CEE498_projectdata.xlsx', sheet_name = 'Data')
df = df.rename(columns={"Unnamed: 0": "Country",'Unnamed: 1':'Features'})
df
df_f = df.melt(id_vars=["Country","Features"],var_name=["Year"],value_name="vals")
df_f = pd.pivot_table(df_f, index = ['Country','Year'], values='vals', columns=['Features'])
df_f.reset_index()
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
df_f.info()
corr_matrix = df_f.corr()
sns.heatmap(corr_matrix);
msno.matrix(df_f);
msno.heatmap(df_f);
full_features = df_f[['Cultivated area (arable land + permanent crops) (1000 ha)','GDP per capita (current US$/inhab)','Long-term average annual precipitation in volume (10^9 m3/year)','Population density (inhab/km2)','Total renewable water resources (10^9 m3/year)','Total water withdrawal per capita (m3/inhab/year)']]
full_features = full_features.groupby(['Country'], as_index = False).mean()
corr_matrix_2 = full_features.corr()
sns.heatmap(corr_matrix_2);
fig, (ax1, ax2) = plt.subplots(1, 2)
plt.subplots_adjust(wspace = 0.7)
full_features.plot('Long-term average annual precipitation in volume (10^9 m3/year)','Total renewable water resources (10^9 m3/year)', kind = 'scatter', color='orange', ax=ax1);
ax1.set_xlabel('Avg Annual Precipitation');
full_features.plot('Long-term average annual precipitation in volume (10^9 m3/year)','Cultivated area (arable land + permanent crops) (1000 ha)', kind = 'scatter', color='blue', ax=ax2);
ax2.set_xlabel('Avg Annual Precipitation');
full_features
df_ww = df.loc[(df['Features']=='Total water withdrawal per capita (m3/inhab/year)')]
for country in df_ww['Country'].unique():
     df_ww[df_ww['Country']==country].plot.bar(x='Features',title=country)
     plt.xticks(rotation = 0);
df_greece = df.loc[(df['Country']=='Greece')]
for feature in df_greece['Features'].unique():
     df_greece[df_greece['Features']==feature].plot.bar(x='Features',title=feature)
     plt.xticks(rotation = 0)
df_israel = df.loc[(df['Country']=='Israel')]
for feature in df_israel['Features'].unique():
     df_israel[df_israel['Features']==feature].plot.bar(x='Features',title=feature)
     plt.xticks(rotation = 0)
avg_ww = df_f.groupby(['Country'], as_index = True).mean()
avg_ww = avg_ww[['Total water withdrawal per capita (m3/inhab/year)']]
avg_ww.reset_index()
avg_ww.describe()
avg_ww.hist(column = 'Total water withdrawal per capita (m3/inhab/year)', bins = 20);
avg_ww_low = df_f.groupby(['Country'], as_index = True).mean()
avg_ww_low = avg_ww_low.loc[avg_ww_low['Total water withdrawal per capita (m3/inhab/year)'] < 400]
avg_ww_low.reset_index()
avg_ww_low.hist(figsize=(30,30), bins = 20);
avg_ww_high = df_f.groupby(['Country'], as_index = True).mean()
avg_ww_high = avg_ww_high.loc[avg_ww_high['Total water withdrawal per capita (m3/inhab/year)'] > 400]
avg_ww_high.reset_index()
avg_ww_high.hist(figsize=(30,30), bins=20);
high_ag = avg_ww_high[['Agricultural water withdrawal as % of total water withdrawal (%)']]
high_ag.reset_index()
print(high_ag.mean(), high_ag.median(), high_ag.max())
low_ag = avg_ww_low[['Agricultural water withdrawal as % of total water withdrawal (%)']]
low_ag.reset_index()
print(low_ag.mean(), low_ag.median(), low_ag.max())
high_dens = avg_ww_high[['Population density (inhab/km2)']]
high_dens.reset_index()
print(high_dens.mean(), high_dens.median(), high_dens.max())
low_dens = avg_ww_low[['Population density (inhab/km2)']]
low_dens.reset_index()
print(low_dens.mean(), low_dens.median(), low_dens.max())
high_ca = avg_ww_high[['Cultivated area (arable land + permanent crops) (1000 ha)']]
high_ca.reset_index()
print(high_ca.mean(), high_ca.median(), high_ca.max())
low_ca = avg_ww_low[['Cultivated area (arable land + permanent crops) (1000 ha)']]
low_ca.reset_index()
print(low_ca.mean(), low_ca.median(), low_ca.max())
high_gdp = avg_ww_high[['GDP per capita (current US$/inhab)']]
high_gdp.reset_index()
print(high_gdp.mean(), high_gdp.median(), high_gdp.max())
low_gdp = avg_ww_low[['GDP per capita (current US$/inhab)']]
low_gdp.reset_index()
print(low_gdp.mean(), low_gdp.median(), low_gdp.max())
high_ta = avg_ww_high[['Total area of the country (excl. coastal waters) (1000 ha)']]
high_ta.reset_index()
print(high_ta.mean(), high_ta.median(), high_ta.max())
low_ta = avg_ww_low[['Total area of the country (excl. coastal waters) (1000 ha)']]
low_ta.reset_index()
print(low_ta.mean(), low_ta.median(), low_ta.max())
high_mun = avg_ww_high[['Municipal water withdrawal as % of total withdrawal (%)']]
high_mun.reset_index()
print(high_mun.mean(), high_mun.median(), high_mun.max())
low_mun = avg_ww_low[['Municipal water withdrawal as % of total withdrawal (%)']]
low_mun.reset_index()
print(low_mun.mean(), low_mun.median(), low_mun.max())