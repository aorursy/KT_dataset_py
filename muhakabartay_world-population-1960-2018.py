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
import matplotlib.pyplot as plt
import os

print(os.listdir("../input"))
# import warnings

import warnings



# ignore warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/world-population-19602018/Use_API_SP.POP.TOTL_DS2_en_csv_v2_1120881.csv')
data.head()
data.columns
data['Country Name']
data['1960']
data['1960'][1]
len(data['1960'])
len(data['Country Name'])
pop_by_countries = []
data['1960']
for i in range(1960,2018+1):

    year = str(i)

    pop_by_countries.append(list(data[year]))
len(pop_by_countries)
len(pop_by_countries)
len(pop_by_countries[0])
countries_list = data['Country Name']

countries_list
years = []

for year in range(1960,2018+1):

    years.append(year)
years
len(pop_by_countries[0])
pop_by_countries[0]