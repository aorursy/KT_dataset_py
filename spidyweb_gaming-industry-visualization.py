# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Basic Visualization tools

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

sns.set_palette('husl')





# Bokeh (interactive visualization)

from bokeh.io import show, output_notebook

from bokeh.palettes import Spectral9

from bokeh.plotting import figure

output_notebook() # You can use output_file()



# Special Visualization

from wordcloud import WordCloud # wordcloud

import missingno as msno # check missing value



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv')
data.head()
len(data)
data.describe()
data.Size.sum()
data.Price.sum()
data.isnull().sum()
msno.matrix(data)


plt.figure(figsize = (13, 8))

ax = sns.countplot(x = 'Average User Rating', data = data, palette = 'dark')

ax.set_title(label = 'Rating Count', fontsize = 20)

ax.set_xlabel(xlabel = 'Rating', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()
plt.figure(figsize = (13, 8))

ax = sns.countplot(x = 'Age Rating', data = data, palette = 'dark')

ax.set_title(label = 'Age Restriction', fontsize = 20)

ax.set_xlabel(xlabel = 'Age', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()




plt.figure(figsize = (25, 8))

ax = sns.countplot(x = 'Primary Genre', data = data, palette = 'dark')

ax.set_title(label = 'Primary Genre Insight', fontsize = 20)

ax.set_xlabel(xlabel = 'primary genre', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()
plt.figure(figsize = (13, 8))

ax = sns.countplot(x = 'Price', data = data, palette = 'dark')

ax.set_title(label = 'Price distribution', fontsize = 20)

ax.set_xlabel(xlabel = 'Price', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()
result = data.groupby(["Average User Rating"])['Price'].aggregate(np.median).reset_index().sort_values('Price')

sns.barplot(x='Average User Rating', y="Price", data=data, order=result['Average User Rating']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()
result = data.groupby(["Age Rating"])['Price'].aggregate(np.median).reset_index().sort_values('Price')

sns.barplot(x='Age Rating', y="Price", data=data, order=result['Age Rating']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()
sns.lineplot(x='Age Rating',y='Price',data=data)

plt.show()
data.Genres.unique()
data['Genres'].head()
data['GenreList'] = data['Genres'].apply(lambda s : s.replace('Games','').replace('&',' ').replace(',', ' ').split()) 

data['GenreList'].head()
from sklearn.preprocessing import MultiLabelBinarizer # Similar to One-Hot Encoding



test = data['GenreList']

mlb = MultiLabelBinarizer()

res = pd.DataFrame(mlb.fit_transform(test), columns=mlb.classes_, index=test.index)
corr = res.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(15, 14))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
data['Original Release Date'] = pd.to_datetime(data['Original Release Date'], format = '%d/%m/%Y')

date_size = pd.DataFrame({'size':data['Size']})

date_size = date_size.set_index(data['Original Release Date'])

date_size = date_size.sort_values(by=['Original Release Date'])

date_size.head()
date_size['size'] = date_size['size'].apply(lambda b : b//(2**10)) # B to KB
fig = figure(x_axis_type='datetime',           

             plot_height=250, plot_width=750,

             title='Date vs App Size')

fig.line(y='size', x='Original Release Date', source=date_size)

show(fig)
monthly_size = date_size.resample('M').mean()

tmp = date_size.resample('M')

monthly_size['min'] = tmp.min()

monthly_size['max'] = tmp.max()

monthly_size.head()
fig = figure(x_axis_type='datetime',           

             plot_height=250, plot_width=750,

             title='Date vs App Size (Monthly)')

fig.line(y='size', x='Original Release Date', source=monthly_size, line_width=2, line_color='Green')

show(fig)
yearly_size = date_size.resample('Y').mean()

monthly_size.head()

fig = figure(x_axis_type='datetime',           

             plot_height=250, plot_width=750,

             title='Date vs App Size (Monthly & Yearly)')

fig.line(y='size', x='Original Release Date', source=monthly_size, line_width=2, line_color='Green', alpha=0.5)

fig.line(y='size', x='Original Release Date', source=yearly_size, line_width=2, line_color='Orange', alpha=0.5)

show(fig)
data['Original Release Date'] = pd.to_datetime(data['Original Release Date'], format = '%d/%m/%Y')

date_IAP = pd.DataFrame({'In-app Purchases':data['In-app Purchases']})

date_IAP = date_IAP.set_index(data['Original Release Date'])

date_IAP = date_IAP.sort_values(by=['Original Release Date'])

date_IAP.head()
fig = figure(x_axis_type='datetime',           

             plot_height=250, plot_width=850,

             title='Date vs In-app Purchases')

fig.line(y='In-app Purchases', x='Original Release Date', source=date_IAP)

show(fig)