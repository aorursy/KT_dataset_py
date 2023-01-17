import pydicom

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns



matplotlib.style.use('ggplot')
DIR_ROOT = '../input/osic-pulmonary-fibrosis-progression'



train_csv = pd.read_csv(f"{DIR_ROOT}/train.csv")
train_csv.head()
print(f"Total number of patient IDs: {len(train_csv)}")
train_csv.info()
train_csv.describe()
train_csv.isnull().values.any()
print(train_csv.Patient.value_counts())

print(train_csv.Patient.value_counts().keys())

patient_id_keys = [train_csv.Patient.value_counts()]

# print(patient_id_keys)
patient_id_dict = {'id': [], 'num': []}

patient_keys = train_csv.Patient.value_counts().keys()

for i, data in enumerate(train_csv.Patient.value_counts()):

    patient_id_dict['id'].append(patient_keys[i])

    patient_id_dict['num'].append(data)
plt.figure(figsize=(20, 17))

plt.bar(patient_id_dict['id'], patient_id_dict['num'], color='orange')

plt.tick_params(

    axis='x',         

    which='both',     

    bottom=False,      

    top=False,        

    labelbottom=False) 

plt.show()
num_patients_list = []

num_visits = []

for i in range(10, 5, -1):

    num_visits.append(i)

    num_patients_counter = 0

    for j in range(len(train_csv.Patient.value_counts())):

        if i == patient_id_dict['num'][j]:

            num_patients_counter += 1

    num_patients_list.append(num_patients_counter)
plt.figure(figsize=(10, 7))

plt.bar(num_patients_list, num_visits, color='orange', width=1)

plt.xlabel('Number of patients')

plt.ylabel('Number of times visited')

plt.show()
mean_csv = train_csv.groupby(['Patient']).mean()

print(mean_csv.head())

print(mean_csv.columns)
plt.figure(figsize=(15, 12))

sns.scatterplot(patient_id_dict['id'], mean_csv['Weeks'], 

                hue=mean_csv['Weeks'], size=mean_csv['Weeks'], 

                sizes=(10, 200))

plt.tick_params(

    axis='x',         

    which='both',     

    bottom=False,      

    top=False,        

    labelbottom=False) 
plt.figure(figsize=(18, 15))

sns.barplot(patient_id_dict['id'], mean_csv['FVC'])

plt.ylabel('Mean FVC (in ml)')

plt.tick_params(

    axis='x',         

    which='both',     

    bottom=False,      

    top=False,        

    labelbottom=False) 
plt.figure(figsize=(18, 15))

sns.barplot(patient_id_dict['id'], mean_csv['Percent'], 

            palette="rocket")

plt.ylabel('FVC Perncentage')

plt.tick_params(

    axis='x',         

    which='both',     

    bottom=False,      

    top=False,        

    labelbottom=False) 
plt.figure(figsize=(15, 12))

sns.swarmplot(patient_id_dict['id'], mean_csv['Age'])

plt.ylabel('Age')

plt.tick_params(

    axis='x',         

    which='both',     

    bottom=False,      

    top=False,        

    labelbottom=False) 