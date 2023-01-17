import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
gym_df = pd.read_csv('../input/data.csv')

gym_df.head()
sns.factorplot(x='month',y='number_people',data=gym_df,hue='is_during_semester',aspect=2)
def get_date(series):

    return series.str.slice(8,11)

gym_df['day_of_month'] = gym_df[['date']].apply(get_date)

month_date_count_df = pd.pivot_table(gym_df, columns=['day_of_month'],index=['month'], values='number_people', aggfunc=np.mean)

month_date_count_df.fillna(0, inplace=True)

fig, ax = plt.subplots(figsize=(18,7)) 

sns.heatmap(month_date_count_df, annot=True, ax=ax, cmap="OrRd")
day_labels = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']

day_hour_df = pd.pivot_table(gym_df, index=['day_of_week'],columns=['hour'], values='number_people', aggfunc=np.mean)

fig, ax = plt.subplots(figsize=(18,7)) 

sns.heatmap(day_hour_df, annot=True, ax=ax, cmap="OrRd", yticklabels=day_labels)
month_date_count_df = pd.pivot_table(gym_df, columns=['day_of_month'],index=['month'], values='number_people', aggfunc=np.mean)

month_date_count_df.fillna(0, inplace=True)

fig, ax = plt.subplots(figsize=(18,7)) 

sns.heatmap(month_date_count_df, annot=True, ax=ax, cmap="OrRd")
month_date_temp_df = pd.pivot_table(gym_df, columns=['day_of_month'],index=['month'], values='temperature', aggfunc=np.mean)

min_temp = month_date_temp_df.min(skipna=True).min()

max_temp = month_date_temp_df.max(skipna=True).max()

month_date_temp_df.fillna(0, inplace=True)

fig, ax = plt.subplots(figsize=(18,7)) 

sns.heatmap(month_date_temp_df, annot=True, ax=ax, cmap="OrRd", vmin=min_temp, vmax=max_temp)