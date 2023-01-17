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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import numpy as np

%matplotlib inline
df= pd.read_csv('/kaggle/input/coronavirusdataset/PatientInfo.csv') #_patientinfo

df_cases= pd.read_csv('/kaggle/input/coronavirusdataset/Case.csv') #Data of COVID-19 infection cases in South Korea
df.columns
dfK= df.query("country=='Korea'")
casesPercity=dfK.groupby('city')['patient_id'].nunique().sort_values(ascending=False).reset_index().head(10)
dfplot2=dfK[dfK.city.isin(casesPercity['city'].head(10))]

fig, ax= plt.subplots(figsize=(15,7))

dfplot2.groupby(['confirmed_date' ,'city'])['patient_id'].nunique().sort_values(ascending=False).unstack().plot(ax=ax)

# dfK.groupby(['confirmed_date' ,'city_x'])['patient_id_x'].nunique().unstack().plot(ax=ax)

plt.show()
dfplot3=dfK[dfK.infection_case=='Shincheonji Church']

#Distribution of where the 93 people who got affected at Shincheonji Church:

fig, ax= plt.subplots(figsize=(10,7))



dfplot3.groupby(['province'])['patient_id'].nunique().sort_values(ascending= False).plot(kind='bar', ax=ax)

plt.title('Shincheonji Church case- Most people belonged to Daegu')

plt.xlabel('Cities they belonged to')

plt.ylabel('# of people affected')
df_cases.query("infection_case=='Shincheonji Church'").sort_values(by='confirmed', ascending=False)
# Confirmation date vs Age of cases affected at Shincheonji Church:

fig, ax= plt.subplots(figsize=(15,7))

dfplot3.groupby(['confirmed_date','age'])['patient_id'].nunique().unstack().plot(kind= 'bar', stacked=True, ax=ax)

plt.title('Confirmation date vs Age of cases affected at Shincheonji Church')

plt.show()
fig, ax= plt.subplots(figsize=(10,7))



dfplot3.groupby(['province'])['patient_id'].nunique().sort_values().plot(kind='barh', ax=ax)
dfplot5= dfK[dfK.infection_case=='overseas inflow']

fig, ax= plt.subplots(figsize=(10,7))



dfplot5.groupby(['province'])['patient_id'].nunique().sort_values(ascending= False).plot(kind='bar', ax=ax)

plt.title('Distribution of cases spread via Overseas Inflow- Most people hailed from Seoul/Gyeonggi-do')

plt.xlabel('Cities they belonged to')

plt.ylabel('# of people affected')
df_cases.query("infection_case=='overseas inflow'").sort_values(by='confirmed', ascending=False)
dfplot4= dfK[dfK.infection_case=='Guro-gu Call Center']

fig, ax= plt.subplots(figsize=(10,7))



dfplot3.groupby(['province'])['patient_id'].nunique().sort_values(ascending= False).plot(kind='bar', ax=ax)

plt.title('Distribution of Guro-gu call center case- Most people hailed from Seoul')

plt.xlabel('Cities they belonged to')

plt.ylabel('# of people affected')
df_cases.query("infection_case=='Guro-gu Call Center'").sort_values(by='confirmed', ascending=False)
dfplot6= dfK[dfK.infection_case=='Onchun Church']

fig, ax= plt.subplots(figsize=(10,5))



dfplot6.groupby(['province'])['patient_id'].nunique().sort_values(ascending= False).plot(kind='bar', ax=ax)

plt.title('Onchun Church case- Most people hailed from Busan')

plt.xlabel('Cities they hailed from')

plt.ylabel('# of people affected')
df_cases.query("infection_case=='Onchun Church'").sort_values(by='confirmed', ascending=False)