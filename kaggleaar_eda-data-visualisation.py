import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

import pandas as pd

data_main = pd.read_csv('/kaggle/input/covid19-symptoms-checker/Raw-Data.csv')

data_main.shape
data_main.head(10)
data_main.dtypes
pd.isnull(data_main)
data_main.describe(include='all')
country = len(data_main.Country.dropna().unique())

age = len(data_main.Age.dropna().unique())

gender = len(data_main.Gender.dropna().unique())

symptoms = len(data_main.Symptoms.dropna().unique())

esymptoms = len(data_main.Experiencing_Symptoms.dropna().unique())

severity = len(data_main.Severity.dropna().unique())

contact = len(data_main.Contact.dropna().unique())



print("Total Combination Possible: ",country * age * gender * symptoms * esymptoms * severity * contact)
import itertools

columns = [data_main.Country.dropna().unique().tolist(),

          data_main.Age.dropna().unique().tolist(),

          data_main.Gender.dropna().unique().tolist(),

          data_main.Symptoms.dropna().unique().tolist(),

          data_main.Experiencing_Symptoms.dropna().unique().tolist(),

          data_main.Severity.dropna().unique().tolist(),

          data_main.Contact.dropna().unique().tolist()]



final_data = pd.DataFrame(list(itertools.product(*columns)), columns=data_main.columns)
final_data.shape
final_data.head(5)
symptoms_list = final_data['Symptoms'].str.split(',')



from collections import Counter

symptoms_counter = Counter(([a for b in symptoms_list.tolist() for a in b]))



for symptom in symptoms_counter.keys():

    final_data[symptom] = 0

    final_data.loc[final_data['Symptoms'].str.contains(symptom), symptom] = 1



final_data.head()
esymptoms_list = final_data['Experiencing_Symptoms'].str.split(',')



from collections import Counter

esymptoms_counter = Counter(([a for b in esymptoms_list.tolist() for a in b]))



for esymptom in esymptoms_counter.keys():

    final_data[esymptom] = 0

    final_data.loc[final_data['Experiencing_Symptoms'].str.contains(esymptom), esymptom] = 1



final_data.head()
final_data = final_data.drop(['Symptoms','Experiencing_Symptoms'],axis=1)

dummies = pd.get_dummies(final_data.drop('Country',axis=1))

dummies['Country'] = final_data['Country']

final_data = dummies

final_data.head(10)
final_data.dtypes
data=pd.read_csv('../input/covid19-symptoms-checker/Cleaned-Data.csv')

data.head(10)
data.dtypes
data.groupby(['Severity_Severe'])

data
plt.figure(figsize=(8,6))

data.groupby('Fever').size().plot(color='green',kind='bar')

plt.show()
plt.figure(figsize=(15,10))

data.groupby('Dry-Cough').sum().plot(kind='hist')

plt.show()