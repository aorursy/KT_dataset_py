# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import geopandas as gpd



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Reading the dataset

users = pd.read_csv('../input/list-of-countries-by-number-of-internet-users/List of Countries by number of Internet Users - Sheet1.csv')
users.head(5)

users = users.sort_values('Country or Area', ascending=False).drop_duplicates('Country or Area').sort_index()


df = gpd.read_file('../input/countries-shape-files/ne_10m_admin_0_countries.shp')

df1 = df[['SOVEREIGNT','geometry']]





#df1.plot(figsize = (20,20))

users['Population']=   users['Population'].str.replace(',', '').astype(int)

users['Internet Users']= users['Internet Users'].str.replace(',', '').astype(int)



users['percent'] = users['Internet Users']/users['Population']

users['percent'] = users['percent'] * 100

users.rename(columns={'Country or Area':'SOVEREIGNT'}, inplace=True)



users2 = users[['SOVEREIGNT', 'percent']]

users2.at[2, 'SOVEREIGNT'] = 'United States of America'





merged = df1.merge(users2,how = 'left', on = 'SOVEREIGNT')

merged['percent'].mean()
sns.set_context("poster")

sns.set_style("whitegrid")

cmap = 'YlGn'

figsize = (30, 30)



plt.style.use('fivethirtyeight')



ax = merged.dropna().plot(column= 'percent', cmap= cmap , figsize=figsize,

                          scheme='User_Defined',

                          classification_kwds=dict(bins=[10,25,50,75,100]),

                          edgecolor = 'black', legend= True)

ax.get_legend().set_bbox_to_anchor((0.15, 0.4))

ax.get_legend().set_title('Percentage')

ax.set_title("Percentage of Internet Users Around The World" , size = 30, pad = 20)

ax.axis('off')

ax.text(-50,-60, "60.12% population of the World has access to Internet", horizontalalignment='left', size='large', color='green', weight='semibold')






