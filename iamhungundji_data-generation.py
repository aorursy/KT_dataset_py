import pandas as pd

data = pd.read_csv('/kaggle/input/covid19-symptoms-checker/Raw-Data.csv')

data.shape
data.head()
country = len(data.Country.dropna().unique())

age = len(data.Age.dropna().unique())

gender = len(data.Gender.dropna().unique())

symptoms = len(data.Symptoms.dropna().unique())

esymptoms = len(data.Experiencing_Symptoms.dropna().unique())

severity = len(data.Severity.dropna().unique())

contact = len(data.Contact.dropna().unique())



print("Total Combination Possible: ",country * age * gender * symptoms * esymptoms * severity * contact)
import itertools

columns = [data.Country.dropna().unique().tolist(),

          data.Age.dropna().unique().tolist(),

          data.Gender.dropna().unique().tolist(),

          data.Symptoms.dropna().unique().tolist(),

          data.Experiencing_Symptoms.dropna().unique().tolist(),

          data.Severity.dropna().unique().tolist(),

          data.Contact.dropna().unique().tolist()]



final_data = pd.DataFrame(list(itertools.product(*columns)), columns=data.columns)
final_data.shape
final_data.head()
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

final_data.head()
final_data.to_csv('Cleaned-Data.csv', index=False, header=True)