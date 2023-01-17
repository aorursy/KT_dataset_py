from math import pi

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.basemap import Basemap

from bokeh.io import output_notebook, show

from bokeh.palettes import Category20c

from bokeh.plotting import figure

from bokeh.transform import cumsum





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



dir = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        dir.append(os.path.join(dirname, filename))

        



# Any results you write to the current directory are saved as output.
df_loaded = pd.read_csv(dir[0])

#dataset = pd.read_csv(dir[1])
df_loaded.shape
df_loaded.head()
df_loaded.tail()
df_loaded.info()
df_loaded.describe()
df_loaded.isnull().sum()
indek = df_loaded.columns.where((df_loaded.isnull().sum() / len(df_loaded) * 100) >= 50).dropna()

df_clean = df_loaded.drop(columns = indek)
numerikal = ['odometer', 'weather']

kategorikal = ['year', 'manufacturer', 'make', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission',

               'drive', 'paint_color', 'county_fips', 'county_name', 'state_code', 'type']

unused = ['image_url', 'state_fips']
for num in numerikal:

    df_clean[num] = df_clean[num].fillna(df_clean[num].mean())
for kat in kategorikal:

    df_clean[kat] = df_clean[kat].fillna(df_clean[kat].mode().values[0])
df_clean = df_clean.drop(columns = unused)
df_clean.isnull().sum()
df_clean = df_clean.where(df_clean['year']>1885)

df_clean = df_clean.dropna()
#create correlation with heatmap

corr = df_clean.corr(method = 'pearson')



#convert correlation to numpy array

mask = np.array(corr)



#to mask the repetitive value for each pair

mask[np.tril_indices_from(mask)] = False

fig, ax = plt.subplots(figsize = (15,12))

fig.set_size_inches(20,20)

sns.heatmap(corr, mask = mask, vmax = 0.9, square = True, annot = True)
numerikal = ['price', 'odometer', 'lat', 'long', 'weather']

kategorikal = ['year', 'manufacturer', 'make', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission',

               'drive', 'paint_color', 'county_fips', 'county_name', 'state_code', 'type']
fig = plt.figure(figsize = (20,20))

axes = 320

for num in numerikal:

    axes += 1

    fig.add_subplot(axes) 

    sns.boxplot(data = df_clean, x = num).set_title('%s - Boxplot' % num)

plt.show()
plt.figure(figsize=(20,6))

ax = sns.countplot(x='year',data=df_clean);

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right",fontsize=6);
df_clean = df_clean.where(df_clean['year']>=1960)

df_clean = df_clean.dropna()
plt.figure(figsize=(20,6))

ax = sns.countplot(x='year',data=df_clean);

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right",fontsize=6);
# fig = plt.figure(figsize = (20,20))

# axes = 440

# for kat in kategorikal:

#     axes += 1

#     fig.add_subplot(axes) 

#     ax = sns.countplot(x=kat, data=df_clean);

#     ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right",fontsize=6);

# #     sns.countplot(df_clean[kat], order = df_clean[kat].value_counts().index)

# plt.show()
plt.figure(figsize=(10,6))

years = df_clean['year'].value_counts().iloc[:10]

years = pd.DataFrame({'years' : years.index.astype(int), 'count' : years.values.astype(int)})

ax = sns.barplot(x='years', y='count', data=years, order = years['years']);

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right",fontsize=8);

plt.title("Top 10 years car production");# fig = plt.figure(figsize = (20,20))
df_shuffled = df_clean.sample(frac=1)

fig = plt.figure(figsize=(15, 15))

m = Basemap(projection='ortho', resolution=None,

            width=8E6, height=8E6, 

            lat_0=45,lon_0=-100)

m.etopo(scale=0.5, alpha=0.5)



i = 0

for index, row in df_shuffled.iterrows():

    lat = row['lat']

    lon = row['long']

    xpt, ypt = m(lon, lat)

    m.plot(xpt,ypt,'.',markersize=0.2,c="red")

    # stopping criteria

    i = i + 1

    if (i == 10000): break
plt.figure(figsize=(10,6))

years = df_clean['manufacturer'].value_counts().iloc[:10]

years = pd.DataFrame({'manufacturer' : years.index, 'count' : years.values.astype(int)})

ax = sns.barplot(x='manufacturer', y='count', data=years, order = years['manufacturer']);

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right",fontsize=12);

plt.title("Top 10 manufacturer");# fig = plt.figure(figsize = (20,20))
plt.figure(figsize=(10,6))

years = df_clean['type'].value_counts().iloc[:10]

years = pd.DataFrame({'type' : years.index, 'count' : years.values.astype(int)})

ax = sns.barplot(x='type', y='count', data=years, order = years['type']);

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right",fontsize=12);

plt.title("Top 10 type of car");# fig = plt.figure(figsize = (20,20))
plt.figure(figsize=(10,6))

years = df_clean['cylinders'].value_counts().iloc[:5]

years = pd.DataFrame({'cylinders' : years.index, 'count' : years.values.astype(int)})

ax = sns.barplot(x='cylinders', y='count', data=years, order = years['cylinders']);

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right",fontsize=12);

plt.title("Top 10 type of car");# fig = plt.figure(figsize = (20,20))
years = df_clean['cylinders'].value_counts().iloc[:5]

x = {}



for A, B in zip(years.index, years.values.astype(int)):

    x[A] = B

output_notebook()
data = pd.Series(x).reset_index(name='value').rename(columns={'index':'country'})

data['angle'] = data['value']/data['value'].sum() * 2*pi

data['color'] = Category20c[len(x)]



p = figure(plot_height=350, title="Pie Chart dari persebaran cylinders mobil yang sering terjual", toolbar_location=None,

           tools="hover", tooltips="@country: @value", x_range=(-0.5, 1.0))



p.wedge(x=0, y=1, radius=0.4,

        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),

        line_color="white", fill_color='color', legend='country', source=data)



p.axis.axis_label=None

p.axis.visible=False

p.grid.grid_line_color = None



show(p)