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
df = pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv')

values=[]

for val in df.columns:

    values.append(val.replace (' ','_').lower())

df.columns= values
Total = df.isnull().sum().sort_values(ascending=False)

percent = ((df.isnull().sum()/df.isnull().count())*100).sort_values(ascending=False)

missing_data = pd.concat([Total,percent],axis=1,keys=['Total','percent'])

missing_data

del df['id'], df['unique_id'],df['detected_city_pt'], df['current_location'],df['current_location_pt'], df['government_id'], df ['contacts']
df.head()
df['detected_state'] = df['detected_state'].apply(lambda x: x.replace(" ","-"))

#df['temp'] = df['detected_state'].apply(lambda x: x.split()[0].strip())

#df['detected_state'].isna().value_counts()

df.head()
df.info()
df['diagnosed_date'] = pd.to_datetime(df['diagnosed_date'])

df ['created_on'] = pd.to_datetime (df ['created_on'])

df ['updated_on'] = pd.to_datetime (df ['updated_on'])

df['status_change_date'] = pd.to_datetime (df['status_change_date'])
df.head()
df['age'].replace(np.nan,0,inplace= True)
df.describe()
from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

temp = df.groupby ('age').size()

plt.figure (figsize=(15,5))

temp.plot.bar()
plt.figure(figsize=(20,5))

chart = sns.countplot(

    data=df[df['detected_state'] == 'Karnataka'],

    x='diagnosed_date',

    palette='Set1'

)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.figure(figsize=(20,5))

chart = sns.countplot(

    data=df[df['current_status'] == 'Recovered'] ,

    x='diagnosed_date',

    palette='Set1'

)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.figure(figsize=(25,5))

maskIndiaKerala = df [(df['nationality'] == 'India') &  (df['detected_state'] == 'Kerala')]

chart = sns.countplot(

    data=maskIndiaKerala, hue = 'current_status',

    x='diagnosed_date',

    palette='Set1'

)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.figure(figsize=(25,5))

maskIndiaKA = df [(df['nationality'] == 'India') &  (df['detected_state'] == 'Karnataka')]

chart = sns.countplot(

    data=maskIndiaKA, hue = 'current_status',

    x='diagnosed_date',

    palette='Set1'

)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
df [(df['detected_state'] == 'Karnataka') & (df['nationality'] == 'India')]



plt.figure(figsize=(25,5))

maskIndiaKA = df [(df['detected_state'] == 'Karnataka') & (df['nationality'] == 'India')]

chart = sns.countplot(

    data=maskIndiaKA, hue = 'current_status',

    x='detected_district',

    palette='Set1'

)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.figure(figsize=(25,5))

maskIndiaKA = df [(df['detected_state'] == 'Kerala') & (df['nationality'] == 'India')]

chart = sns.countplot(

    data=maskIndiaKA, hue = 'current_status',

    x='detected_district',

    palette='Set1'

)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.figure(figsize=(25,5))

maskIndiaKA = df [(df['detected_state'] == 'Karnataka') & (df['diagnosed_date'] > '2020-01-21')]

chart = sns.countplot(

    data=maskIndiaKA, hue = 'current_status',

    x='gender',

    palette='Set1'

)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
covid19 = pd.read_csv ('/kaggle/input/covid19-in-india/covid_19_india.csv')

values=[]

for val in covid19.columns:

    values.append(val.replace (' ','_').lower())

covid19.columns= values

covid19.head()

covid19.rename (columns = {'state/unionterritory':'state'}, inplace= True)

covid19.head()
covid19['date'] = pd.to_datetime (covid19['date'])

covid19.info()
covid19.describe()
data = pd.read_csv ('/kaggle/input/covid19-in-india/population_india_census2011.csv')

data.head()