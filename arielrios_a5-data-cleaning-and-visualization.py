# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plotting library for data viz



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
cv = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

cv_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_confirmed.csv")

cv_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_deaths.csv")

cv_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_recovered.csv")
cv.head()
cv.columns

cv.dtypes
cv.groupby("Country")['Confirmed'].sum()
cv.groupby("Country")['Confirmed'].sum().plot(kind="bar")
cv.groupby("Country")['Deaths'].count()
cv.groupby("Country")['Deaths'].count().plot(kind="bar")
cv.groupby("Country")['Recovered'].count()
cv.groupby("Country")['Recovered'].count().plot(kind="bar")
cv.groupby("Country")['Confirmed','Deaths', 'Recovered'].sum().plot(kind="bar")
#cv.plot.pie(y=["Confirmed","Deaths", "Recovered"])