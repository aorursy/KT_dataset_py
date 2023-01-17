import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set(style="whitegrid")
df_hrData = pd.read_csv('../input/HR_comma_sep.csv'

                        , header=0, encoding="ISO-8859-1")
df_hrData.head(5)
df_hrData["average_daily_hours"] = df_hrData["average_montly_hours"]/22

df_hrData["sat_idx"] = (df_hrData.satisfaction_level * df_hrData.average_daily_hours)/df_hrData.last_evaluation

df_hrData["stayed"] = (df_hrData["left"] - 1) * (-1)
df_hrData[["average_daily_hours", "left", "stayed"]].head(5)
df_workdata = df_hrData[['sales', 'average_daily_hours', 'left', 'stayed', 'salary', 'sat_idx']]

df_workdata.head(10)
df_workdata.groupby(['sales']).sum().reset_index()
df_workdata['daily_hours'] = df_workdata['average_daily_hours']/1

df_workdata['daily_hours'] = np.floor(df_workdata['daily_hours'])

df_workdata['daily_hours'] = df_workdata['daily_hours'].astype('int')
df_workdata.head()
df_work = df_workdata.groupby(['sales', 'daily_hours']).sum()
df_work['left_percentage'] = df_work['left']*100/(df_work['left'] + df_work['stayed'])



df_work['average_daily_hours'] = df_work['average_daily_hours']/(df_work['left'] + df_work['stayed'])
df_work
df_work.plot(x='average_daily_hours', y='left_percentage', 

             colormap='gist_rainbow')