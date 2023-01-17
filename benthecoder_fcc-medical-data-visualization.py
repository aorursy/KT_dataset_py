# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/medical_examination.csv')
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2) > 25
medical_dict = { 1: 0, 2: 1, 3: 1}

df['cholesterol'] = df['cholesterol'].map(medical_dict)

df['gluc'] = df['gluc'].map(medical_dict)
# Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.

df_cat = pd.melt(

        frame=df, value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], 

        id_vars=['cardio']

    )



# Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the collumns for the catplot to work correctly.



df_cat = pd.DataFrame(

        df_cat.groupby(

                ['variable', 'value', 'cardio'])['value'].count()).rename(

                columns={'value': 'total'}).reset_index()



# Set up the matplotlib figure and draw the catplot

sns.catplot(x='variable', y='total', data=df_cat, hue='value', col='cardio', kind='bar')

# Clean the data

df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 

        (df['height'] >= df['height'].quantile(0.025)) &

        (df['height'] <= df['height'].quantile(0.975)) &

        (df['weight'] >= df['weight'].quantile(0.025)) & 

        (df['weight'] <= df['weight'].quantile(0.975))]



# Calculate the correlation matrix

corr = df_heat.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

fig, ax = plt.subplots(figsize=(10, 12))



# Draw the heatmap with the mask

sns.heatmap(corr, annot=True, fmt='.1f', mask=mask, vmin=.16, vmax=.32, center=0, square=True, linewidths=.5, cbar_kws={'shrink':.45, 'format':'%.2f'})
