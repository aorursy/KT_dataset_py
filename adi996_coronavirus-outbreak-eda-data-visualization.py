# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
patient = pd.read_csv('/kaggle/input/coronavirusdataset/patient.csv')
patient.info()
patient.isnull().sum()
patient.head(10)
patient.tail(10)
sns.set(rc={'figure.figsize':(15,15)})

sns.countplot(

    y=patient['region'],



).set_title('Regions affected Overall')
reason = [x for x in patient['infection_reason'].unique()]

size = [len((patient['infection_reason'].loc[patient['infection_reason']==reason])) for reason in reason]
fig1, ax1 = plt.subplots(figsize=(10,10))

ax1.pie(size,labels=reason, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax1.legend()

ax1.set_title('Reasons CoronaVirus\n\n')

plt.show()
sns.set(rc={'figure.figsize':(5,5)})

sns.countplot(x=patient['state'].loc[

    (patient['infection_reason']=='contact with patient')

])
patient['country'].unique()
patient['age'] = 2020-patient['birth_year']
sns.set(rc={'figure.figsize':(5,5)})

sns.countplot(x=patient['sex'].loc[(patient['country']=="China")]).set_title('Affected population , By gender')
sns.set(rc={'figure.figsize':(5,5)})

sns.countplot(x=patient['state'].loc[(patient['country']=="China") &

                                    (patient['sex']=="female")]).set_title('Female state in china')
sns.set(rc={'figure.figsize':(5,5)})

sns.countplot(x=patient['state'].loc[(patient['country']=="China") &

                                    (patient['sex']=="male")]).set_title('Male state in china')
sns.distplot(patient['birth_year'].loc[

    (patient['country']=="China") &

    (patient['sex']=="female")

    

]).set_title("Distribution plot for year , Females in China")
sns.distplot(patient['birth_year'].loc[

    (patient['country']=="China") &

    (patient['sex']=="male")

    

]).set_title('Distribution plot for birth year , Males in China')
sns.set(rc={'figure.figsize':(5,5)})

sns.distplot(patient['age'].loc[

    (patient['country']=="China") &

    (patient['sex']=="male")

    

]).set_title('Distribution plot for age , Males in China')
sns.set(rc={'figure.figsize':(5,5)})

sns.distplot(patient['age'].loc[

    (patient['country']=="China") &

    (patient['sex']=="female")

    

]).set_title('Distribution plot for age , Females in China')
sns.countplot(

    patient['region'].loc[

        (patient['country']=="China") 

    ]

).set_title('Regions in china where the patient got affected')
sns.set(rc={'figure.figsize':(10,10)})

sns.countplot(

    y = patient['confirmed_date'].loc[

        (patient['country']=="China")

    ]



).set_title('Confirmed dates in China')
sns.set(rc={'figure.figsize':(5,5)})

sns.countplot(x=patient['sex'].loc[(patient['country']=="Korea")]).set_title('Affected population , By gender in Korea')
sns.set(rc={'figure.figsize':(5,5)})

sns.countplot(x=patient['state'].loc[(patient['country']=="Korea") &

                                    (patient['sex']=="female")]).set_title('Female state in Korea')
sns.set(rc={'figure.figsize':(5,5)})

sns.countplot(x=patient['state'].loc[(patient['country']=="Korea") &

                                    (patient['sex']=="male")]).set_title('Male state in Korea')
sns.distplot(patient['age'].loc[

    (patient['country']=="Korea") &

    (patient['sex']=="female")

    

]).set_title("Distribution plot for age , Females in Korea")
sns.distplot(patient['birth_year'].loc[

    (patient['country']=="Korea") &

    (patient['sex']=="female")

    

]).set_title("Distribution plot for year , Females in Korea")
sns.distplot(patient['age'].loc[

    (patient['country']=="Korea") &

    (patient['sex']=="male")

    

]).set_title("Distribution plot for age , Males in Korea")
sns.distplot(patient['birth_year'].loc[

    (patient['country']=="Korea") &

    (patient['sex']=="male")

    

]).set_title('Distribution plot for birth year , Males in Korea')
sns.set(rc={'figure.figsize':(15,15)})

sns.countplot(

    y=patient['region'].loc[

        (patient['country']=="Korea")],



).set_title('Regions affected in Korea')
region_korea = [x for x in patient['region'].loc[patient['country']=="Korea"].unique()]

size_region_korea = [len(patient['region'].loc[(patient['region']==region)])

                     for region in region_korea]
fig1, ax1 = plt.subplots(figsize=(10,10))

ax1.pie(size_region_korea,labels=region_korea, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax1.legend()

ax1.set_title('Regions in Korea\n\n')

plt.show()
sns.set(rc={'figure.figsize':(15,15)})

sns.countplot(

    y=patient['infection_reason'].loc[

        (patient['country']=="Korea")],



).set_title('Infection reason in Korea')
infection_reason = [x for x in patient['infection_reason'].loc[patient['country']=="Korea"].unique()]

size_infection_korea = [len(patient['infection_reason'].loc[(patient['infection_reason']==infection_reason)])

                     for infection_reason in infection_reason]
fig1, ax1 = plt.subplots(figsize=(10,10))

ax1.pie(size_infection_korea,labels=infection_reason, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax1.legend()

ax1.set_title('Reason for getting infection in Korea\n\n')

plt.show()
sns.set(rc={'figure.figsize':(15,15)})

sns.countplot(

    y=patient['confirmed_date'].loc[

        (patient['country']=="Korea")],



).set_title('Confirmed dates in Korea')