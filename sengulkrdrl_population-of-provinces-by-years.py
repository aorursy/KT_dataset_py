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
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/cihadturhan/tr-geojson/master/geo/tr-cities-utf8.json') as response:
    cities = json.load(response)
cities["features"][0]
df = pd.read_csv('../input/populationcsv/yillaragoreiller.csv') 
df.head(7)
df.rename(columns = {'Unnamed: 0':'city'},inplace =True)
df = df.melt( id_vars=["city"],
                     value_vars = [ "2000","2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019"])
df.rename(columns = {'city':'cities','variable':'year', 'value':'population'},inplace =True)
indexlist = [0]
for i in range(1,82,1):
    indexlist.append(i)
indexlist2 = []
for i in range(1,82,1):
    indexlist2.append(i)
# indexlist2
indexlist.extend(indexlist2 +indexlist2 ) 
indexlist # for 4 years 
indexlist.extend(indexlist2+indexlist2+indexlist2+indexlist2+indexlist2+indexlist2+indexlist2+indexlist2+indexlist2+indexlist2)
len(indexlist)
test_list = indexlist[: len(indexlist) - 243]
len(test_list)   #(1621-1)/81= 20 yılı sağlar.
df
len(df)
df.head()
df['id'] = test_list 
print(df.head(10))