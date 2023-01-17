import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
all_points_df = pd.read_csv('../input/all_data_points.csv')
print('loaded', all_points_df.shape[0], 'points')
all_points_df.sample(3)
all_points_df['date'] = pd.to_datetime(all_points_df['date'], format = '%Y-%m-%d')
act_sum_df = all_points_df.groupby(['date', 'activity']).agg(dict(year = len)).reset_index().rename_axis(dict(year='Points'),1)
fig, ax1 = plt.subplots(1,1, figsize = (8, 8), dpi = 200)
for c_name, c_df in act_sum_df.groupby(['activity']):
    ax1.plot(c_df['date'], c_df['Points'], '.', label = c_name)
ax1.legend();
fig, ax1 = plt.subplots(1,1, figsize = (9, 9), dpi = 200)
for c_name, c_df in act_sum_df.groupby(['activity']):
    ax1.plot(c_df['date'], np.cumsum(c_df['Points']), '.-', label = c_name)
ax1.legend();
import seaborn as sns
act_sum_df['month'] = act_sum_df['date'].dt.month

sns.pairplot(act_sum_df.groupby(['activity','month']).agg(np.sum).reset_index(), 
             hue = 'activity')
sns.factorplot(x = 'month', y = 'Points', 
               hue = 'activity',
               data = act_sum_df.groupby(['activity','month']).agg(np.sum).reset_index())
