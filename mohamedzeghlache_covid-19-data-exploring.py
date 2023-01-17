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
df_cCov_data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

df_cCov_data
df_cCov_data["Country/Region"].unique()
df_cCov_data.loc[df_cCov_data['Country/Region'] == 'Mainland China', 'Country/Region'] = 'China'

df_cCov_data.loc[df_cCov_data['Country/Region'] == 'Hong Kong', 'Country/Region'] = 'China'

df_cCov_data.loc[df_cCov_data['Country/Region'] == 'UK', 'Country/Region'] = 'United Kingdom'
df_cCov_data.info()
import matplotlib.pyplot as plt

#plt.bar(x = df_cCov_data[df_cCov_data['Country']], height = df_cCov_data[df_cCov_data['Confirmed']])

df_cCov_data[['Confirmed']].plot(kind='line', title ="Confirmed cases line Chart", figsize=(15, 10))

df_cCov_data.loc[df_cCov_data['Country/Region'] == 'China', 'Rest'] = 'China'

df_cCov_data.loc[df_cCov_data['Country/Region'] != 'China', 'Rest'] = 'Rest of the world'
#df_cCov_data.drop(df_cCov_data.columns[0],axis=1,inplace=True)

csv_corona = df_cCov_data.to_csv('COVID.csv', index = False)
df = pd.read_csv('COVID.csv')

df