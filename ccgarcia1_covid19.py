# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



df_1=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

df_2=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

df_3=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

df_4=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')

df_5=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')

df_6=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')





df_1_melt=df_1.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], 

        var_name="Date", 

        value_name="Value")



df_2_melt=df_2.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], 

        var_name="Date", 

        value_name="Value")



df_3_melt=df_3.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], 

        var_name="Date", 

        value_name="Value")



df = pd.DataFrame(columns = ['Province/State', 'Country/Region', 'Lat', 'Long', 'confirmed','recovered', 'deaths']) 



df = pd.DataFrame(columns = ['Province/State', 'Country/Region', 'Lat', 'Long', 'confirmed','recovered', 'deaths'])

for i in ['Province/State', 'Country/Region', 'Lat', 'Long','Date']:

    df[i]=df_1_melt[i]

df['confirmed']=df_3_melt['Value']

df['recovered']=df_1_melt['Value']

df['deaths']=df_2_melt['Value']



df.to_csv('mycsvfile.csv',index=False)