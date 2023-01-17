# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
drug_data = pd.read_csv('../input/adverse_drug_35.csv')
columns_of_interest = ['receiptdate',

                      'patient.patientonsetage',

                      'primarysource.reportercountry',

                     'patient.patientsex',

                     'patient.patientweight',

                     'patient.drug',

                     'patient.reaction',

                     'serious']



sample = drug_data[columns_of_interest]
import datetime as dt



sample['receiptdate'] = pd.to_datetime(sample['receiptdate'].astype(str), format='%Y%m%d')

i = 0



while i < len(sample):

    sample.loc[sample.index[i], 'Patient Name'] = "Patient " + str(i)

    i += 1

    

cols = sample.columns.tolist()

new_cols = [cols[-1]] + cols[0:-1]



sample = sample.reindex(columns = new_cols)
sample.reset_index()
c = 0

y = []

for pat in sample['patient.drug']:

    d = eval(pat)

    y.append(len(d))

    

    

while c < len(sample):

    sample.loc[sample.index[c], 'numdrugs'] = y[c]

    c += 1

    

sample = sample.loc[sample.index.repeat(sample.numdrugs.astype(int))].reset_index(drop=True)
sample
for i, patient in enumerate(sample['patient.drug']):

    

    drugs = eval(patient)

    

    for drug in drugs:

        col_names = list(drug.keys())

        for col in col_names:

            if col not in sample.columns:

                sample.insert(len(sample.columns), col, value = None, allow_duplicates = False)

        
sample
enter_drug = 0



for i, patient in enumerate(sample['patient.drug']):

    

    drugs = eval(patient)

    

    for k, v in drugs[enter_drug].items():

        if type(v)  == dict:

            v = str(v)

        if type(v) == list:

            v = str(v)

        sample.loc[i, k] = v

            

    if enter_drug == (sample['numdrugs'][i] - 1):

        enter_drug = 0

    else:

        enter_drug += 1
sample = sample.drop('numdrugs', axis = 1)

sample = sample.drop('patient.drug', axis = 1)

sample = sample.set_index('Patient Name')
sample