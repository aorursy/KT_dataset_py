# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from datetime import datetime
corona_virus_brazil = pd.read_csv('../input/corona-virus-brazil/brazil_covid19.csv');

corona_virus_brazil_temp = pd.read_csv('../input/corona-virus-brazil/brazil_covid19.csv');

corona_virus_italy = pd.read_csv('../input/covid-19-italy-updated-regularly/national_data.csv');

corona_virus_usa = pd.read_csv('../input/covid19-in-usa/us_covid19_daily.csv');

print('Loaded')
plt.rcParams['figure.figsize'] = (20,7)

corona_virus_brazil_temp_state = corona_virus_brazil_temp.groupby('state').count()

corona_virus_brazil_temp_state = len(corona_virus_brazil_temp_state.index)

corona_virus_brazil_temp_total = corona_virus_brazil_temp.loc[ (len(corona_virus_brazil_temp) - corona_virus_brazil_temp_state) : len(corona_virus_brazil_temp) ]

corona_virus_brazil_temp_total = corona_virus_brazil_temp_total.reset_index()

corona_virus_brazil_temp_total.groupby('region').count()

corona_virus_brazil_temp_total_pie =  corona_virus_brazil_temp_total.loc[:,['region','cases']].groupby('region').sum()

corona_virus_brazil_temp_total_barh = corona_virus_brazil_temp_total.loc[:,['state','cases']].groupby('state').sum()

plt.subplot(121)

plt.barh(corona_virus_brazil_temp_total_barh.index, corona_virus_brazil_temp_total_barh['cases'])

plt.subplot(122)

plt.pie(corona_virus_brazil_temp_total_pie['cases'], autopct="%1.1f%%", labels=corona_virus_brazil_temp_total_pie.index)

plt.axis('equal') 

plt.show()
corona_virus_italy_cases = pd.DataFrame(corona_virus_italy['data'],corona_virus_italy['totale_positivi']).rename(columns={'index': 'date', 'total_positive_cases': 'cases'})

corona_virus_italy_cases.tail()

print('Finished')
corona_virus_brazil_national_temp = corona_virus_brazil_temp.groupby('date').sum().reset_index()

corona_virus_italy_national_temp = corona_virus_italy.groupby('data').sum().reset_index()

corona_virus_usa_national_temp = corona_virus_usa.groupby('date').sum().reset_index()

corona_virus_usa[0:len(corona_virus_usa)]

corona_virus_brazil_national_temp.count()

corona_virus_italy_national_temp.count()

corona_virus_usa_national_temp.count()



print('Finished')
plt.figure(figsize=(15,10))



g = sns.lineplot(x=corona_virus_brazil_national_temp.index, y=corona_virus_brazil_national_temp['cases'], label="Brazil")

g = sns.lineplot(x=corona_virus_italy_national_temp.index, y=corona_virus_italy_national_temp['totale_positivi'], label="Italy")

g = sns.lineplot(x=corona_virus_usa_national_temp.index, y=corona_virus_usa_national_temp['positive'], label="USA")



plt.ylabel('Evolution of cases')

plt.xlabel('period( days )')

plt.title('Evolution of cases per day')

plt.show()



print('Finished')
plt.figure(figsize=(15,10))



sns.lineplot(x=corona_virus_brazil_national_temp.index, y=corona_virus_brazil_national_temp["deaths"], label='Brazil')

sns.lineplot(x=corona_virus_italy_national_temp.index, y=corona_virus_italy_national_temp["deceduti"], label='Italy')

sns.lineplot(x=corona_virus_usa_national_temp.index, y=corona_virus_usa_national_temp["death"], label='USA')

plt.xlabel("Daily Evolution index of deaths confirmed by the new Corona Virus")

plt.title('Evolution of deaths per day')

plt.show()



print('Finished')
plt.figure(figsize=(15,10))



sns.lineplot(x='cases', y='deaths' , data=corona_virus_brazil_national_temp, label='Brazil' )

sns.lineplot(x='totale_positivi', y='deceduti' , data=corona_virus_italy_national_temp, label='Italy' )

sns.lineplot(x='positive', y='death' , data=corona_virus_usa_national_temp, label='USA' )

plt.ylabel('Number of deaths')

plt.xlabel('Positive cases evolution')

plt.title('Correlation between positive cases evolution and death numbers evolution')

plt.show()



print('Finished')