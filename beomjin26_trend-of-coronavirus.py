import numpy as np

import pandas as pd 



df = pd.read_csv('../input/coronavirusdataset/patient.csv')

df.head(5)
# How many NaN?

print('How many NaN?')

df.isna().sum().to_frame().sort_values(0).style.background_gradient(cmap='summer_r')
df2 = df.country.value_counts() / len(df) *100

df2.plot.bar(title='The percentage of hometowns of paitents in Korea', color='r')
import seaborn as sns; sns.set()

import matplotlib.pyplot as plt



conf = df.iloc[:,:] # Confirmed

# Cut year.

conf['time'] = conf['confirmed_date'].apply(lambda x : str(x)[5:])

# Value count

counts = conf['time'].value_counts().sort_index().reset_index()

counts.columns = ['time', 'new_patient']

# Plot

fig = plt.figure(figsize=(15,8))

plt.xticks(rotation=45)

ax = sns.lineplot(x="time", y="new_patient", data=counts)

import datetime # Change date string to date object



patient = df.loc[:,['confirmed_date', 'state']]

patient.dropna(inplace=True)

patient['confirmed_date'] = patient['confirmed_date'].apply(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d'))



# New columns 

patient['time'] = 0 

patient['accum'] = 0
# Calculate

for i in range(1, len(patient)):

    days = (patient.iloc[i, 0] - patient.iloc[i-1, 0])

    if days == 0:

        patient.iloc[i, 2] = patient.iloc[i-1, 2]

    else:

        patient.iloc[i, 2] = patient.iloc[i-1, 2] + days.days





Released = 0

Isolated = 0

Deceased = 0



for i in range(0, len(patient)):

    if patient.iloc[i, 1] == 'released':

        Released +=1 

        patient.iloc[i, 3] = Released

    if patient.iloc[i, 1] == 'isolated':

        Isolated +=1

        patient.iloc[i, 3] = Isolated

    else:

        Deceased +=1

        patient.iloc[i, 3] = Deceased

    

patient
sns.factorplot('time','accum', data=patient, hue='state',size=10, kind='point')
sns.factorplot('time','accum', data=patient, hue='state',size=8, kind='point').set(ylim=(0, 80))