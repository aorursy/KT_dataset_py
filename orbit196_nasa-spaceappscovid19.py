# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.chdir('/kaggle/input/')

df = pd.read_csv('new-york usa-air-quality.csv', parse_dates = ['date'])

df.drop([' o3', ' no2', ' co'], axis = 1, inplace = True)

df.head()
df2019 = df[(df['date'] >= '2019-01-01') & (df['date'] <= '2019-12-31')]

df2020 = df[df['date'] >= '2020-01-01']
df2020 = df2020.sort_values('date', ascending=True)

plt.plot(df2020['date'], df2020[' pm25'])

plt.xticks(rotation='vertical')
df2019 = df2019.sort_values('date', ascending=True)

plt.plot(df2019['date'], df2019[' pm25'])

plt.xticks(rotation='vertical')
import pandas as pd

import os



data2020 = pd.read_csv(

    os.path.join(

        "/kaggle/input/aqiindices", "waqi-covid19-airqualitydata-2020.csv"), 

        skiprows = 4

    )



data2019 = pd.read_csv(

    os.path.join(

        "/kaggle/input/aqiindices", "waqi-covid19-airqualitydata-2019Q1.csv"), 

        skiprows = 4)



data2019 = data2019.append(

    pd.read_csv(

        os.path.join("/kaggle/input/aqiindices", 

            "waqi-covid19-airqualitydata-2019Q2.csv"

        ),  skiprows = 4)

    )





cities = [

    "New York",

    "Los Angeles",

    "Chicago",

    "Washington",

    "San Francisco"

]



# what did did these cities pass 3 daily deaths?

# https://ig.ft.com/coronavirus-chart/?areas=usa&areas=gbr&areasRegional=usny&areasRegional=usnj&cumulative=0&logScale=1&perMillion=0&values=deaths



start_dates = {

    "New York" : "2020-03-13",

    "Los Angeles": "2020-03-02"

}