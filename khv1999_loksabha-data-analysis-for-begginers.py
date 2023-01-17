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
data_2004 = pd.read_csv('/kaggle/input/lok-sabha-election-candidate-list-2004-to-2019/LokSabha2004.csv')

data_2004.head()
data_2009 = pd.read_csv('/kaggle/input/lok-sabha-election-candidate-list-2004-to-2019/LokSabha2009.csv')

data_2014 = pd.read_csv('/kaggle/input/lok-sabha-election-candidate-list-2004-to-2019/LokSabha2014.csv')

data_2019 = pd.read_csv('/kaggle/input/lok-sabha-election-candidate-list-2004-to-2019/LokSabha2019.csv')
data_2004['year'] = 2004

data_2009['year'] = 2009

data_2014['year'] = 2014

data_2019['year'] = 2019
data = pd.concat([data_2004, data_2009, data_2014, data_2019])
data.shape
data.head()
data.columns
data.sort_values(by = 'Criminal Cases', ascending=False)
crimes_city = pd.DataFrame()

crimes_city['Criminal Cases'] = data['Criminal Cases']

crimes_city['City'] = data['City']
crimes_city.head()
top_cities = crimes_city.groupby('City').sum().sort_values(by='Criminal Cases', ascending=False)[:10]
top_cities.plot(kind='bar')
data['Education'].value_counts().plot(kind='bar')
(data['Education'].value_counts()/data['Education'].count()*100).plot(kind='bar')
(data['Education'].value_counts()/data['Education'].count()*100).plot(kind='pie', figsize=(25, 10), autopct='%1.2f%%')
data.loc[data['Education']=='Doctorate']
data.loc[data['Education']=='Illiterate']