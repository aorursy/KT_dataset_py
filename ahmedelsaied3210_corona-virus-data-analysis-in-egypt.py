# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data =pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

data.head()
data.shape
data.isnull().sum()
data=data.rename(columns={"Province/State": "State", "Country/Region": "Country","Last Update":"Update"})

data.head()
print(data[data['Country']=='Egypt'])
data['Country'] = data['Country'].replace('Mainland China', 'China')

data.head()
data['Active_'] = data['Confirmed'] - data['Deaths'] - data['Recovered']

data.head()
egypt=data[data['Country']=='Egypt'].drop(['State','Update'],axis=1)

egypt.plot(kind='bar',figsize=(15,8))

egypt.head()
egypt.plot(x="ObservationDate", y=["Confirmed"], kind="bar",figsize=(15,8))
egypt.plot(x="ObservationDate", y=["Deaths"], kind="bar",figsize=(15,8))
egypt.plot(x="ObservationDate", y=["Confirmed", "Deaths", "Recovered","Active_"], kind="bar",figsize=(15,8))
egypt.plot(x="ObservationDate", y=["Confirmed", "Deaths", "Recovered","Active_"], kind="line",figsize=(15,8))


egypt.plot(x="ObservationDate", y=["Active_"], kind="line",figsize=(10,8))

plt.show()
egypt.sort_values(by=['ObservationDate'])
last_updata_egypt = egypt[egypt["ObservationDate"]=='03/23/2020']

last_updata_egypt