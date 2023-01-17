# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from mpl_toolkits.basemap import Basemap
from collections import Counter
from sklearn.model_selection import train_test_split
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

df_events = pd.read_csv("../input/events.csv")
df_events.info()
df_events.head(10)
# https://www.kaggle.com/beyondbeneath/talkingdata-mobile-user-demographics/geolocation-visualisations
# Set up plot
df_events_sample = df_events.sample(n=100000)
plt.figure(1, figsize=(12,6))

# Mercator of World
m1 = Basemap(projection='merc',
             llcrnrlat=-60,
             urcrnrlat=65,
             llcrnrlon=-180,
             urcrnrlon=180,
             lat_ts=0,
             resolution='c')

m1.fillcontinents(color='#191919',lake_color='#000000') # dark grey land, black lakes
m1.drawmapboundary(fill_color='#000000')                # black background
m1.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders

# Plot the data
mxy = m1(df_events_sample["longitude"].tolist(), df_events_sample["latitude"].tolist())
m1.scatter(mxy[0], mxy[1], s=3, c="#1292db", lw=0, alpha=1, zorder=5)

plt.title("Global view of events")
plt.show()
counter_at_zero = len(df_events[(df_events["longitude"] == 0) & (df_events["latitude"] == 0)])
counter_all_events = len(df_events)
print("All events", counter_all_events)
print("Incorrect location", counter_at_zero, "which is %.2f%% of data." % round((counter_at_zero/counter_all_events),2))
df_gender=pd.read_csv("../input/gender_age_train.csv")

print(df_gender.gender.value_counts())
gender_plot = sns.countplot(x="gender", data=df_gender)
sns.countplot(x="group", data=df_gender)
sns.plt.title('Male age group count')
plt.show() 
sns.countplot(x="group", data=df_gender[df_gender.gender=="F"])
sns.plt.title('Female age group count')
plt.show()
sns.kdeplot(df_gender.age[df_gender.gender=="M"], label="Male")
sns.kdeplot(df_gender.age[df_gender.gender=="F"],  label="Female")
plt.legend()
plt.title('Age distribution')
plt.show()
def map_column(table, f):
    labels = sorted(table[f].unique())
    mappings = dict()
    for i in range(len(labels)):
        mappings[labels[i]] = i
    table = table.replace({f: mappings})
    return table
df_gender_train = df_gender
#Map group (sush as 'F43+') to numbers and drop age and gender
df_gender_train = df_gender_train.drop(['age'], axis=1)
df_gender_train = df_gender_train.drop(['gender'], axis=1)
df_gender_train = map_column(df_gender_train, 'group')

# Read everything into memory
df_app_events = pd.read_csv('../input/app_events.csv')
df_app_labels = pd.read_csv('../input/app_labels.csv')
df_events = pd.read_csv('../input/events.csv')
df_label_categories = pd.read_csv('../input/label_categories.csv')

#Read, drop duplicates and replace names with numbers 
df_phone_brand_device_model = pd.read_csv('../input/phone_brand_device_model.csv')
df_phone_brand_device_model.drop_duplicates('device_id', keep='first', inplace=True)
df_phone_brand_device_model = map_column(df_phone_brand_device_model, 'phone_brand')
df_phone_brand_device_model = map_column(df_phone_brand_device_model, 'device_model')

# the ONE TABLE to rule them all
df = df_gender_train.merge(df_events, how='left', on='device_id')
df = df.merge(df_phone_brand_device_model, how='left', on='device_id')
df = df.merge(df_app_events, how='left', on='event_id')
df = df.merge(df_app_labels, how='left', on='app_id')
df=  df.merge(df_label_categories, how='left', on='label_id')

df.info()
df = df.iloc[np.random.permutation(len(df))]
df.head(10)
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots()

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, vmax=.5, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
plt.show()
batch = df[:10000]
features = list(batch.columns.values)
train, test, y_train, y_test = train_test_split(
    batch, 
    batch['group'], 
    test_size=0.33, 
    random_state=13)