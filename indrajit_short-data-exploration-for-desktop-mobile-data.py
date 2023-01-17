import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import matplotlib as mpl



mpl.rcParams['axes.facecolor'] = '#ffffff'

mpl.rcParams["axes.edgecolor"] = "0.15"

mpl.rcParams["axes.linewidth"]  = 1.25



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train_1.csv').fillna(0)
for col in train.columns[1:]:

    train[col] = pd.to_numeric(train[col],downcast='integer')
#here we will seperate and make columns for various components of the project

components = pd.DataFrame([i.split("_")[-3:] for i in train["Page"]])

components.columns = ['Project', 'Access', 'Agent']

train[['Project', 'Access', 'Agent']] = components[['Project', 'Access', 'Agent']]

cols = train.columns.tolist()

cols = cols[-3:] + cols[:-3]

train = train[cols]

train.head()
#here wil well make three different dataframes to understand each one of them

df_access = train.groupby(['Access'])[cols].mean()

df_access = df_access.T



df_project = train.groupby(['Project'])[cols].mean()

df_project = df_project.T



df_agent = train.groupby(['Agent'])[cols].mean()

df_agent = df_agent.T
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (20,20), sharex=True)

df_access.plot(ax = ax1)

df_agent.plot(ax = ax2)

df_project.plot(ax = ax3)
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (20,20), sharex=True)

pd.rolling_mean(df_access, window=14).plot(ax = ax1)

pd.rolling_mean(df_agent, window=14).plot(ax = ax2)

pd.rolling_mean(df_project, window=14).plot(ax = ax3)
#basically we will filtr out English project along with each type of access

df_all = train[(train['Access'] == 'all-access') & (train['Project'] == 'en.wikipedia.org')]

df_all = df_all.groupby(['Access'])[cols].mean()

df_all = df_all.T



df_desktop = train[(train['Access'] == 'desktop') & (train['Project'] == 'en.wikipedia.org')]

df_desktop = df_desktop.groupby(['Access'])[cols].mean()

df_desktop = df_desktop.T



df_mobile = train[(train['Access'] == 'mobile-web') & (train['Project'] == 'en.wikipedia.org')]

df_mobile = df_mobile.groupby(['Access'])[cols].mean()

df_mobile = df_mobile.T
df_all.head()
#scaling all three dataframes to understand impact. Min-Max scalling to be used.

df_desktop['desktop'] = (df_desktop.desktop - df_desktop.desktop.min())/(df_desktop.desktop.max() - df_desktop.desktop.min())

df_mobile['mobile-web'] = (df_mobile['mobile-web'] - df_mobile['mobile-web'].min())/(df_mobile['mobile-web'].max() - df_mobile['mobile-web'].min())

df_all['all-access'] = (df_all['all-access'] - df_all['all-access'].min())/(df_all['all-access'].max() - df_all['all-access'].min())
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (20,20), sharex=True)

pd.rolling_mean(df_all, window=14).plot(ax = ax1, style = 'g--')

pd.rolling_mean(df_desktop, window=14).plot(ax = ax2, style = 'g--')

pd.rolling_mean(df_mobile, window=14).plot(ax = ax3, style = 'g--')



print('On an average the traffic on all access is {}'.format(df_all['all-access'].mean()))

print('On an average the traffic on desktop is {}'.format(df_desktop['desktop'].mean()))

print('Deviation on desktop is {}'.format(df_desktop['desktop'].std()))

print('On an average the traffic on mobile is {}'.format(df_mobile['mobile-web'].mean()))

print('Deviation on mobile is {}'.format(df_mobile['mobile-web'].std()))