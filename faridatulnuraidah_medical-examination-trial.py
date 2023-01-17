# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/medical-examination-dataset/medical_examination.csv')
# Add 'overweight' column
df['overweight'] = round(df['weight']/pow(df['height']*0.01,2),0)
df['overweight'] = df['overweight'].where(df['overweight']>25,0)
df['overweight'] = df['overweight'].where(df['overweight']<=25,1)
df['overweight'] = df['overweight'].astype(int)
# Normalize data by making 0 always good and 1 always bad. If the value of 'cholestorol' or 'gluc' is 1, 
# make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = df['cholesterol'].where(df['cholesterol']>1,0)
df['cholesterol'] = df['cholesterol'].where(df['cholesterol']==0,1)
df['gluc'] = df['gluc'].where(df['gluc']>1,0)
df['gluc'] = df['gluc'].where(df['gluc']==0,1)
# Create DataFrame for cat plot using `pd.melt` using just the values from 
# 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
df_cat = df.melt(id_vars=['cardio'],value_vars=['active','alco','cholesterol', 'gluc','overweight','smoke'])
df_cat = pd.DataFrame(df_cat.groupby(['variable', 'value', 'cardio'])['value'].count()).rename(columns={'value': 'total'}).reset_index()
# Draw the catplot with 'sns.catplot()'
sns.catplot(x='variable',y='total', hue='value',col='cardio', data=df_cat, kind="bar")
plt.show()
# Clean the data. Filter out the following patient segments that represent incorrect data:
# diastolic pressure is higher then systolic (Keep the correct data with df['ap_lo'] <= df['ap_hi']))
# height is less than the 2.5th percentile (Keep the correct data with (df['height'] >= df['height'].quantile(0.025)))
# height is more than the 97.5th percentile
# weight is less then the 2.5th percentile
# weight is more than the 97.5th percentile

df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 
             (df['height'] >= df['height'].quantile(0.025)) &
             (df['height'] <= df['height'].quantile(0.975)) &
             (df['weight'] >= df['weight'].quantile(0.025)) &
             (df['weight'] <= df['weight'].quantile(0.975))]
df_heat
mcorr = df_heat.corr()
mcorr
sns.heatmap(mcorr)
mask = np.zeros_like(mcorr,dtype=bool)
mask[np.triu_indices_from(mask)] = True
# Draw the heatmap with the mask
plt.subplots(figsize=(10, 12))
sns.heatmap(mcorr, annot=True, fmt='.1f', mask=mask, vmin=.16, vmax=.32, 
            center=0, square=True, linewidths=.5, cbar_kws={'shrink':.45, 'format':'%.2f'})
