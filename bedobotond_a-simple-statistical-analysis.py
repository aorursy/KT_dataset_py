# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import scipy.stats

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from bs4 import BeautifulSoup as soup

from urllib.request import urlopen as uReq

import plotly.graph_objects as go

import wordcloud
# The dataset before cleaning

cars = pd.read_csv('/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv')

print(cars.shape)

cars.head()
cars['country'] = cars['country'].str.strip()



# For title_status there is only 2 option, and salvage insurance takes only 6,5% of the database

# so I won't work with it, because it could distort the results.

cars = cars[cars['title_status'] == 'clean vehicle']



# There is a total of 7 data from the 2499 where the country is canada. I will work only with usa.

cars = cars[cars['country'] == 'usa']



# The Unnamed: 0 column is only another index column, and I won't use the lot, because

# the vin code will be my identifier, so I drop them too.

cars.drop(columns = ['Unnamed: 0', 'lot','condition','title_status', 'country'], inplace = True)

print(cars.shape)

cars.head()
sns.set(rc={'figure.figsize':(12,8)})



car_corr = cars.corr()

sns.heatmap(car_corr, annot = True, annot_kws={'size':50}, center = 0, cmap = 'magma')

plt.show()
fig, axes = plt.subplots(1, 3, figsize = (24, 6))



plt.suptitle('Correlation in our numeric data',fontsize = 18, y = 1.05)



sns.regplot(x = 'price', y = 'year', data = cars, marker = '.', ci= False,

            line_kws={'color': '#0A333A'}, scatter_kws={'color':'#7FBCC6'}, ax = axes[0])

cr1 = scipy.stats.pearsonr(cars['price'], cars['year'])

axes[0].set_title(cr1, pad = 20)



sns.regplot(x = 'price', y = 'mileage', data = cars, marker = '.', ci= False,

            line_kws={'color': '#0A333A'}, scatter_kws={'color':'#7FBCC6'}, ax = axes[1])

cr2 = scipy.stats.pearsonr(cars['price'], cars['mileage'])

axes[1].set_title(cr2, pad = 20)



sns.regplot(x = 'year', y = 'mileage', data = cars, marker = '.', ci= False,

            line_kws={'color': '#0A333A'}, scatter_kws={'color':'#7FBCC6'}, ax = axes[2])

cr3 = scipy.stats.pearsonr(cars['year'], cars['mileage'])

axes[2].set_title(cr3, pad = 20)





plt.show()
temp = pd.DataFrame(cars.groupby(['brand']).count()['vin'])

temp.sort_values('vin', ascending = False, inplace = True)

# temp[temp['vin'] > 10].sum().values / temp.sum().values == 0.98239588

# temp[temp['vin'] > 10].count().values / temp.count().values == 0.48148148

brand_list = temp[temp['vin'] > 10].index.values
av_prices = []

for i in brand_list:

    x = cars[cars['brand']==i]

    av_price = sum(x.price)/len(x)

    av_prices.append(av_price)

data = pd.DataFrame({'brand_list': brand_list,'av_prices':av_prices})

new_index = (data['av_prices'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)



sns.barplot(y=sorted_data['brand_list'], x=sorted_data['av_prices'], palette = 'GnBu_d')

plt.xlabel('Average Price ($)', fontsize = 14)

plt.ylabel('Brand', fontsize = 14)

plt.title('Average price per brand', fontsize = 16)

plt.show()
counts = []

for i in brand_list:

    x = cars[cars['brand']==i]

    count = len(x.vin)

    counts.append(count)

data2 = pd.DataFrame({'brand_list': brand_list,'counts':counts})

new_index2 = (data2['counts'].sort_values(ascending=False)).index.values

sorted_data2 = data2.reindex(new_index2)



sns.barplot(y=sorted_data2['brand_list'], x=sorted_data2['counts'], palette = 'GnBu_d')

plt.xlabel('# of brands', fontsize = 14)

plt.ylabel('Brand', fontsize = 14)

plt.title('Number of brands', fontsize = 16)

plt.show()
pf = sorted_data.merge(sorted_data2, on = 'brand_list')

sns.lmplot(x = 'av_prices', y = 'counts', data = pf,

          line_kws={'color': '#0A333A'}, scatter_kws={'color':'#7FBCC6'})

plt.show()

scipy.stats.pearsonr(pf['av_prices'], pf['counts'])
cars['count'] = 1

brand_list = temp[temp['vin'] > 20].index.values

cars_b = cars[np.in1d(cars['brand'],brand_list)]

c_sun = px.sunburst(cars_b, path = ['brand','model'], values = 'count', color = 'price', 

            width = 750, height = 750, color_continuous_scale = 'Teal')

cars.drop(columns = 'count', inplace = True)

c_sun.show()
page_url = 'https://www.infoplease.com/us/postal-information/state-abbreviations-and-state-postal-codes'

uClient = uReq(page_url)

page_soup = soup(uClient.read(), "html.parser")

uClient.close()



containers = page_soup.tbody.findAll('tr')
out_filename = 'state_code.csv'

headers = 'state,code\n'



f = open(out_filename, "w")

f.write(headers)



for container in containers:

    cont = container.findAll('td')

    

    state = cont[0].text

    code = cont[2].text



    f.write(state.strip().lower() + ',' + code + '\n')

    

f.close()
sc = pd.read_csv('state_code.csv')

cars = cars.merge(sc, on = 'state')

cars.head()
carsc = pd.DataFrame(cars.groupby(['code']).count()['vin'])



fig = go.Figure(data=go.Choropleth(locations=carsc.index, z = carsc['vin'],

                                   locationmode = 'USA-states', colorscale = 'Teal',

                                   colorbar_title = '# of cars'))



fig.update_layout(title_text = 'Cars advertised by state', geo_scope='usa')



fig.show()
states = carsc[carsc['vin'] >= 10].index.values



carsp = cars[np.in1d(cars['code'],states)]

carsp = pd.DataFrame(carsp.groupby(['code']).mean()['price'])





fig = go.Figure(data=go.Choropleth(locations=carsp.index, z = carsp['price'],

                                   locationmode = 'USA-states', colorscale = 'Teal',

                                   colorbar_title = 'USD'))



fig.update_layout(title_text = 'Average orice of the cars by states', geo_scope='usa')



fig.show()