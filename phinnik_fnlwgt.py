import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

df = pd.read_csv('/kaggle/input/adult-census-income/adult.csv')
def highlight_cols(s):
    color = '#a8f8ff'
    return 'background-color: %s' % color

df.head().style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['fnlwgt']])
df['fnlwgt'].sum()
f, axs = plt.subplots(1,2,figsize=(15,5))

male_count = df[df['sex'] == 'Male']['fnlwgt'].sum()
female_count = df[df['sex'] == 'Female']['fnlwgt'].sum()


axs[0].bar(x=['male', 'female'], height=[male_count, female_count], color=['red', 'blue'])
axs[0].set_title('Численность людей обоих полов учитывая fnlwgt')
axs[0].set_xlabel('Пол')
axs[0].set_ylabel('Численность')



male_count = len(df[df['sex'] == 'Male'])
female_count = len(df[df['sex'] == 'Female'])


axs[1].bar(x=['male', 'female'], height=[male_count, female_count], color=['red', 'blue'])
axs[1].set_title('Численность людей обоих полов НЕ учитывая fnlwgt')
axs[1].set_xlabel('Пол')
axs[1].set_ylabel('Численность')

plt.show()
f, axs = plt.subplots(1,2,figsize=(15,5))

gs_people = df[df['income'] == '>50K']['fnlwgt'].sum()
not_gs_people = df[df['income'] != '>50k']['fnlwgt'].sum()


axs[0].bar(x=['good salary', 'bad salary'], height=[gs_people, not_gs_people], color=['red', 'blue'])
axs[0].set_title('Соотношение людей с хорошей и плохой зарплатой, учитывая fnlwgt')
axs[0].set_xlabel('Зарплата')
axs[0].set_ylabel('Численность')


gs_people = len(df[df['income'] == '>50K'])
not_gs_people = len(df[df['income'] != '>50k'])




axs[1].bar(x=['good salary', 'bad salary'], height=[gs_people, not_gs_people], color=['red', 'blue'])
axs[1].set_title('Соотношение людей с хорошей и плохой зарплатой, НЕ учитывая fnlwgt')
axs[1].set_xlabel('Зарплата')
axs[1].set_ylabel('Численность')

plt.tight_layout()
marital_status_values = list(df['marital.status'].value_counts().keys())



marital_status_counts = {ms: len(df[df['marital.status'] == ms]) for ms in marital_status_values}
marital_status_counts_fnlwgt = {ms: df[df['marital.status'] == ms]['fnlwgt'].sum() for ms in marital_status_values}


f, axs = plt.subplots(2,1,figsize=(14,5))

axs[0].bar(x = list(marital_status_counts.keys()), height=list(marital_status_counts.values()))
axs[0].set_title('Семейное положение НЕ учитывая fnlwgt')
axs[0].set_xlabel('Семейное положение')
axs[0].set_ylabel('Численность')



axs[1].bar(x = list(marital_status_counts_fnlwgt.keys()), height=list(marital_status_counts_fnlwgt.values()))
axs[1].set_title('Семейное положение учитывая fnlwgt')
axs[1].set_xlabel('Семейное положение')
axs[1].set_ylabel('Численность')


plt.tight_layout()
