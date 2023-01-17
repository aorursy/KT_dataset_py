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
route = pd.read_csv("/kaggle/input/coronavirusdataset/route.csv")

trend = pd.read_csv("/kaggle/input/coronavirusdataset/trend.csv")

time = pd.read_csv("/kaggle/input/coronavirusdataset/time.csv")

patient = pd.read_csv("/kaggle/input/coronavirusdataset/patient.csv")
route['date'] = route['date'].str.split('2020-').str.join('')

route = route.sort_values(by=['date'])

route.head()
route_date = route.groupby('date').agg('count')

route_date.head()
plt.figure(figsize=(20, 5))

sns.barplot(route_date.index, route_date['id'])

plt.show()
route_prov = route.groupby('province').agg('count')

route_prov.head()
plt.figure(figsize=(20, 5))

sns.barplot(route_prov.index, route_prov['id'])

plt.show()
density = pd.DataFrame()

density['province'] = route_prov.index

density['pop_density'] = [2773, 90, 2980, 1279, 141, 2764, 226, 145, 16034]



density
plt.figure(figsize=(20, 5))

sns.barplot(route_prov.index, route_prov['id'])

plt.figure(figsize=(20, 5))

sns.barplot(density['province'], density['pop_density'])

plt.show()
def normalize(v):

    norm = np.sum(v)

    if norm == 0: 

       return v

    return v / norm
corr = pd.DataFrame()

corr['infected'] = normalize(np.array(route_prov['id']))

corr['density'] = normalize(np.array(density['pop_density']))



corr.index = route_prov.index



corr_coef = pd.DataFrame(np.corrcoef(np.array(corr[['infected', 'density']])))

corr_coef.columns = corr.index

corr_coef.index = corr.index

corr_coef
def return_corr_reason(df, col):

    return df[df[col] == 1.0].index
sns.heatmap(corr_coef)

plt.show()
print("first correlation group :", return_corr_reason(corr_coef, 'Daegu').tolist())

print("second correlation group :", return_corr_reason(corr_coef, 'Seoul').tolist())
col1 = return_corr_reason(corr_coef, 'Daegu').tolist()

col2 = return_corr_reason(corr_coef, 'Seoul').tolist()



print("first correlation group's impact :", sum(list(corr.loc[col1, 'infected'])))

print()

print("maximum of first correlation group's impact")

print(corr[corr['infected'] == max(list(corr.loc[col1, 'infected']))]['infected'])

print()

print()

print("second correlation group's impact :", sum(list(corr.loc[col2, 'infected'])))

print()

print("maximum of second correlation group's impact")

print(corr[corr['infected'] == max(list(corr.loc[col2, 'infected']))]['infected'])
print("first observation date :", route.head(1)['date'])

print("final observation date :", route.tail(1)['date'])
trend.head()
sns.lmplot(x='cold', y='coronavirus', data=trend)

sns.lmplot(x='flu', y='coronavirus', data=trend)

sns.lmplot(x='pneumonia', y='coronavirus', data=trend)

plt.show()
plt.figure(figsize=(20, 5))

plt.plot(trend['date'], trend.drop('date', axis=1))

plt.xlabel('date')

plt.ylabel('volume')

plt.xticks('')

plt.legend(['cold', 'flu', 'pneumonia', 'coronavirus'])

plt.show()
time['date'] = time['date'].str.split('2020-').str.join('')

time.head()


plt.figure(figsize=(20, 5))

plt.title("New Confirmed by Date")

sns.barplot(time['date'], time['new_confirmed'])

plt.xticks([])





plt.figure(figsize=(20, 5))

plt.title("New Released by Date")

sns.barplot(time['date'], time['new_released'])

plt.xticks([])





plt.figure(figsize=(20, 5))

plt.title("New Deceased by Date")

sns.barplot(time['date'], time['new_deceased'])

plt.xticks([])

plt.show()
time['acc_infected'] = np.array(time['acc_confirmed']) - np.array(time['acc_released'])



plt.figure(figsize=(20, 5))

plt.title("Infected People by Date")

sns.barplot(time['date'], time['acc_infected'])

plt.xticks([])

plt.show()
patient.head()
patient_birth = patient.groupby('birth_year').agg('count')

patient_birth.head()
patient_birth.index = list(map(int, np.array(2020) - np.array(patient_birth.index)))

                           

plt.figure(figsize=(30, 5))

sns.barplot(patient_birth.index, patient_birth['id'])

plt.show()
patient_reason = patient.groupby('infection_reason').agg('count')

patient_reason.head()
plt.figure(figsize=(40, 5))

sns.barplot(patient_reason.index, patient_reason['id'])

plt.show()
patient_contact = patient.groupby('infection_reason').agg('sum')

patient_contact.head()
plt.figure(figsize=(40, 5))

sns.barplot(patient_contact.index, patient_contact['contact_number'])

plt.show()
corr = pd.DataFrame()

corr['id'] = normalize(patient_reason['id'])

corr['contact_number'] = normalize(patient_contact['contact_number'])

corr_coef = pd.DataFrame(np.corrcoef(np.array(corr[['id', 'contact_number']])))

corr_coef.columns = corr.index

corr_coef.index = corr.index

corr_coef
sns.heatmap(corr_coef)

plt.show()
print("first correlation group :", return_corr_reason(corr_coef, 'visit to Wuhan').tolist())

print("second correlation group :", return_corr_reason(corr_coef, 'contact with patient').tolist())
col1 = return_corr_reason(corr_coef, 'visit to Wuhan').tolist()

col2 = return_corr_reason(corr_coef, 'contact with patient').tolist()



print("first correlation group's impact :", sum(list(corr.loc[col1, 'id'])))

print()

print("maximum of first correlation group's impact")

print(corr[corr['id'] == max(list(corr.loc[col1, 'id']))]['id'])

print()

print()

print("second correlation group's impact :", sum(list(corr.loc[col2, 'id'])))

print()

print("maximum of second correlation group's impact")

print(corr[corr['id'] == max(list(corr.loc[col2, 'id']))]['id'])