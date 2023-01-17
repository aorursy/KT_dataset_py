# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')
bulletin_dir = "../input/cityofla/CityofLA/Job Bulletins"

data_list = []

for filename in os.listdir(bulletin_dir):

    with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:

        for line in f.readlines():

            #Insert code to parse job bulletins

            if "Open Date:" in line:

                job_bulletin_date = line.split("Open Date:")[1].split("(")[0].strip()

        data_list.append([filename, job_bulletin_date])
df = pd.DataFrame(data_list)

df.columns = ["FILE_NAME", "OPEN_DATE"]

df["OPEN_DATE"] = df["OPEN_DATE"].astype('datetime64[ns]')

df.info()

data = df.groupby('OPEN_DATE').count()
from fbprophet import Prophet
data.index
data['FILE_NAME'].values.astype(int)
df = pd.DataFrame()

df['ds'] = data.index

df['y'] = data['FILE_NAME'].values.astype(int)

df = df.dropna()
m = Prophet(changepoint_prior_scale=0.01).fit(df)

future = m.make_future_dataframe(periods=300, freq='H')

fcst = m.predict(future)

fig = m.plot(fcst)
fig = m.plot_components(fcst)
m = Prophet(seasonality_mode='multiplicative').fit(df)

future = m.make_future_dataframe(periods=3652)

fcst = m.predict(future)

fig = m.plot(fcst)
m = Prophet(seasonality_mode='multiplicative', mcmc_samples=300).fit(df)

fcst = m.predict(future)

fig = m.plot_components(fcst)