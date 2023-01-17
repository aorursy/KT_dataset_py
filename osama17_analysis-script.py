import pandas as pd

import numpy as np

import matplotlib as plt

import os
csv_data = pd.read_csv('../input/carona-paient-record/COVID19.csv')
average_age = csv_data['age'].mean()

print('Average Age of Patient: ',average_age)
gender = ['male','female']

for value in gender:

    gender_no = len(csv_data[csv_data['gender'] == value].index)

    percentage = (gender_no/len(csv_data['gender'].dropna().index)*100)

    check_avg = csv_data[csv_data['gender']==value]['age'].mean()

    print(value+':','Total:',gender_no,'Percentage: ',percentage,'Average Age: ',check_avg)



all_countries = []

countries = csv_data['country'].drop_duplicates()

for value in countries:

    all_countries.insert(0,value)

all_countries = all_countries[::-1]
for value in all_countries:

    print(value)

    select_country = csv_data[csv_data['country'] == value]

    for sex in gender:

        sex_ratio = len(select_country[select_country['gender'] == sex].index)

        sex_percentage = (sex_ratio/len(select_country['gender'].index)*100)

        sex_avg_age = select_country[select_country['gender']==sex]['age'].mean()

        print(sex,sex_ratio,'Percentage:',sex_percentage,'Average Age: ',sex_avg_age)
all_symptom = []

symptom = csv_data['symptom'].dropna().drop_duplicates()

for value in symptom:

    all_symptom.insert(0,value)

all_symptom = all_symptom[::-1]

print(all_symptom)

    