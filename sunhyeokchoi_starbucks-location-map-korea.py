# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))



import warnings
warnings.filterwarnings('ignore')
starbucks = pd.read_csv('../input/directory.csv')
starbucks.shape
starbucks.head()
# 국가별 매장수 
con_count = pd.DataFrame(starbucks['Country'].value_counts())
# 'country' 컬럼을 인덱스로 지정해 주고
con_count['국가'] = con_count.index
con_count.columns = ['매장 수', '국가']
# index 컬럼을 삭제하고 순위를 알기 위해 reset_index()를 해준다.
con_count = con_count.reset_index().drop('index', axis=1)
con_count.head(20)
#korea
krdata=starbucks[starbucks['Country']=='KR']
krdata.head(100)
plt.figure(figsize=(10,5))

plt.subplot(2,1,1)
c = pd.value_counts(krdata['Ownership Type'],sort=True).sort_index()
c.plot(kind='bar')
plt.title("Ownership")

#시도
plt.subplot(2,1,2)
c = pd.value_counts(krdata['State/Province'],sort=True).sort_index()
c.plot(kind='bar')
plt.title("State/Province") 
from mpl_toolkits.basemap import Basemap

plt.figure(figsize=(11,13))
m = Basemap(llcrnrlon=124.455502, llcrnrlat=32.847638, urcrnrlon=132.111760, urcrnrlat=38.789297,
             resolution='h', projection='lcc', lat_0 = 35.95,lon_0=128.25)
x, y = m(krdata['Longitude'].tolist(),krdata['Latitude'].tolist())
m.shadedrelief()
m.drawcountries()
m.scatter(x, y, 4, marker='o', color='r')
plt.title("Starbuck's locations in Korea")
plt.show()
