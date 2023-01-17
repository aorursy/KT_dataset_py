# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import folium



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/egyptianpyramids/pyramids.csv')

df.isna().sum()
df_full = df.loc[:,['Base1 (m)', 'Base2 (m)', 'Height (m)', 'Slope (dec degr)', 'Volume (cu.m)', 'Type']].dropna()

diff = []

for idx, row in df_full.iterrows():

    exp_vol = row['Base1 (m)'] * row['Base1 (m)'] * row['Height (m)'] / 3.

    diff_to_exp = (abs(float(row['Volume (cu.m)'].replace('.','')) - exp_vol))

    diff.append(diff_to_exp / exp_vol)      # We work with ratios

fig, ax = plt.subplots()

ax.set_xlabel('Difference with expected value')

ax.set_ylabel('Count')

plt.hist(diff, range = (0, max(diff)), bins = 10)
print(len(df_full))
for idx, row in df_full.iterrows():

    exp_vol = row['Base1 (m)'] * row['Base1 (m)'] * row['Height (m)'] / 3.

    diff_to_exp = (abs(float(row['Volume (cu.m)'].replace('.','')) - exp_vol))

    dif = diff_to_exp / exp_vol 

    print(row['Type']+' ->  '+str(dif)) 
print(set(df['Dynasty']))
dfh = df[df['Height (m)'] > 0.]

dyn = set(df['Dynasty'])

h_by_dyn = {}

for d in dyn:

    h_by_dyn[d] = dfh['Height (m)'][dfh['Dynasty'] == d]

    print(str(d)+' -> '+str(len(h_by_dyn[d])))
fig, ax = plt.subplots()

ax.boxplot(h_by_dyn.values(), dyn)

ax.set_xlabel('Dynasties')

ax.set_ylabel('Height (m)')
m = folium.Map(location=[28.04, 30.71], zoom_start=7, tiles='Stamen Terrain')

for idx, row in df.iterrows():

    folium.Marker([row['Latitude'], row['Longitude']]).add_to(m)

m
lat = []

h = []

for idx, row in df.iterrows():

    if row['Height (m)'] > 0:

        h.append(row['Height (m)'])

        lat.append(row['Latitude'])

fig, ax = plt.subplots()

ax.set_xlabel('Latitude')

ax.set_ylabel("Pyramid's height")

ax.scatter(lat, h)
lat = []

sl = []

for idx, row in df.iterrows():

    if row['Slope (dec degr)'] > 0:

        sl.append(row['Slope (dec degr)'])

        lat.append(row['Latitude'])

fig, ax = plt.subplots()

ax.set_xlabel('Latitude')

ax.set_ylabel("Pyramid's slope")

ax.scatter(lat, sl)
h = []

sl = []

for idx, row in df.iterrows():

    if row['Slope (dec degr)'] > 0 and row['Height (m)'] > 0:

        sl.append(row['Slope (dec degr)'])

        h.append(row['Height (m)'])

fig, ax = plt.subplots()

ax.set_ylabel("Pyramid's height")

ax.set_xlabel("Pyramid's slope")

ax.scatter(sl, h)