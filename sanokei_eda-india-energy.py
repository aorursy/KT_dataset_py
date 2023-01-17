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
region = pd.read_csv('/kaggle/input/daily-power-generation-in-india-20172020/State_Region_corrected.csv')
power = pd.read_csv('/kaggle/input/daily-power-generation-in-india-20172020/file.csv')
region.head()
power.head(6)
region.info()
power.info()
region = pd.pivot_table(region, index='Region', values=['Area (km2)','National Share (%)'], aggfunc=np.sum)
power = power.rename(columns={'Thermal Generation Actual (in MU)':'thermal_act', 
                             'Thermal Generation Estimated (in MU)':'thermal_est', 
                             'Nuclear Generation Actual (in MU)':'nuclear_act', 
                             'Nuclear Generation Estimated (in MU)':'nuclear_est', 
                             'Hydro Generation Actual (in MU)':'hydro_act', 
                             'Hydro Generation Estimated (in MU)':'hydro_est'})
pd.DataFrame(power.isnull().sum(),columns=['null_sum']).transpose()
power = power.fillna(0)
power['Date'] = pd.to_datetime(power['Date'])
power['thermal_est'] = power['thermal_est'].str.replace(',', '')
power['thermal_act'] = power['thermal_act'].str.replace(',', '')
power['thermal_act'] = power['thermal_act'].astype(np.float)
power['thermal_est'] = power['thermal_est'].astype(np.float)
power['total_act'] = power['thermal_act'] + power['nuclear_act'] + power['hydro_act']
power['total_est'] = power['thermal_est'] + power['nuclear_est'] + power['hydro_est']
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
power_total = pd.pivot_table(power, index='Date', values=['thermal_act','thermal_est',
                                                          'nuclear_act','nuclear_est',
                                                          'hydro_act','hydro_est',
                                                          'total_act','total_est'], aggfunc=np.sum).reset_index()
power_total['Date'] = pd.to_datetime(power_total['Date'])
plt.figure(figsize=(22, 6))
sns.lineplot(data=power_total, x='Date', y='total_act', label='total_act')
sns.lineplot(data=power_total, x='Date', y='total_est', label='total_est')
plt.title('total power, actual and estimated')
plt.ylabel('total_power')
plt.figure(figsize=(22, 6))

ax1 = plt.subplot(1,2,1)
ax1 = sns.lineplot(data=power_total, x='Date', y='hydro_act', label='hydro')
ax1 = sns.lineplot(data=power_total, x='Date', y='nuclear_act', label='nuclear')
ax1 = sns.lineplot(data=power_total, x='Date', y='thermal_act', label='thermal')
plt.title('actual energy resorces')

ax2 = plt.subplot(1,2,2)
ax2 = sns.lineplot(data=power_total, x='Date', y='hydro_est', label='hydro')
ax2 = sns.lineplot(data=power_total, x='Date', y='nuclear_est', label='nuclear')
ax2 = sns.lineplot(data=power_total, x='Date', y='thermal_est', label='thermal')
plt.title('estimated energy resorces')
plt.figure(figsize=(22, 6))

ax1 = plt.subplot(1,2,1)
ax1 = sns.lineplot(data=power, x='Date', y='total_act', hue='Region')
plt.title('total actual power by regions')

ax1 = plt.subplot(1,2,2)
ax1 = sns.lineplot(data=power, x='Date', y='total_est', hue='Region')
plt.title('total estimated power by regions')
power_total['year'] = power_total['Date'].dt.year
power_total['month'] = power_total['Date'].dt.month
power_total_month = round(pd.pivot_table(power_total, index=['month'],
                                   values=power_total.columns[1:-2]).reset_index(), 2)
plt.figure(figsize=(22, 6))

ax1 = plt.subplot(1,2,1)
ax1 = plt.bar(power_total_month['month'], power_total_month['thermal_act'], label='thermal')
ax1 = plt.bar(power_total_month['month'], power_total_month['hydro_act'], 
       bottom=power_total_month['thermal_act'], label='hydro')
ax1 = plt.bar(power_total_month['month'], power_total_month['nuclear_act'], 
       bottom=power_total_month['thermal_act'], label='nuclear')

plt.legend(loc='lower left')
plt.xticks(power_total_month['month'])
plt.xlabel('month')
plt.ylabel('power')
plt.title('actual month transition')

ax2 = plt.subplot(1,2,2)
ax2 = plt.bar(power_total_month['month'], power_total_month['thermal_est'], label='thermal')
ax2 = plt.bar(power_total_month['month'], power_total_month['hydro_est'], 
       bottom=power_total_month['thermal_est'], label='hydro')
ax2 = plt.bar(power_total_month['month'], power_total_month['nuclear_est'], 
       bottom=power_total_month['thermal_est'], label='nuclear')

plt.legend(loc='lower left')
plt.xticks(power_total_month['month'])
plt.xlabel('month')
plt.ylabel('power')
plt.title('estimated month transition')