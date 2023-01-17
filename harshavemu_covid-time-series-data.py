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
confirmed_df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

recovered_df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

deaths_df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")




def filter_cntry(country="China"):

    c_df = confirmed_df[confirmed_df["Country/Region"]==country]

    r_df = recovered_df[confirmed_df["Country/Region"]==country]

    d_df = deaths_df[confirmed_df["Country/Region"]==country]

    c_df = c_df.drop(['Province/State','Country/Region','Lat','Long'],axis=1)

    r_df = r_df.drop(['Province/State','Country/Region','Lat','Long'],axis=1)

    d_df = d_df.drop(['Province/State','Country/Region','Lat','Long'],axis=1)

    sum_c = c_df.sum(axis=0)

    sum_r = r_df.sum(axis=0)

    sum_d = d_df.sum(axis=0)

    

    ax = sum_c.plot.line(x="Date",y = "count",label="Confirmed",title=country,figsize=(10,5),legend=1)

    ax = sum_r.plot.line(x="Date",y = "count",label="Recovered",figsize=(10,5),legend=1)

    ax = sum_d.plot.line(x="Date",y = "count",label="Deaths",figsize=(10,5),legend=1)

    return sum_c,sum_r,sum_d



sum_c,sum_r,sum_d  = filter_cntry()
n = sum_c.pct_change(periods=1).plot.line()

n1 = sum_r.pct_change(periods=1).plot.line()

n2 = sum_d.pct_change(periods=1).plot.line()