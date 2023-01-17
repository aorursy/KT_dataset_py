import numpy as np

import seaborn as sns

import pandas as pd



import time
# fetch data and convert to dataframe

beginning = time.time()

data = pd.read_json('../input/prescriptionbasedprediction/roam_prescription_based_prediction.jsonl', lines=True)

print(time.time() - beginning, 'seconds')
data.head()
data.shape
# merge json data along axis

t = time.time()

prescription_counts = pd.DataFrame(data=[row for row in data['cms_prescription_counts']])

provider_variables = pd.DataFrame(data=[row for row in data['provider_variables']])

npi = pd.DataFrame(data=[row for row in data['npi']])

prescription_data = pd.concat([prescription_counts, npi, provider_variables], axis=1)

print(prescription_data.shape)

print(time.time() - t)

del prescription_counts

del provider_variables

del npi
prescription_data.head()
prescription_data = prescription_data.fillna(0)

prescription_data.memory_usage().sum()
prescription_data = prescription_data.to_sparse(fill_value=0)

prescription_data.memory_usage().sum()
# List prescription counts for each drug sorted by most prescribed.

# Batching to deal with limited memory available

t = time.time()

prescription_counts_1 = prescription_data.iloc[:, 0:500].sum()

prescription_counts_2 = prescription_data.iloc[:, 501:1000].sum()

prescription_counts_3 = prescription_data.iloc[:, 1001:1500].sum()

prescription_counts_4 = prescription_data.iloc[:, 1501: 2000].sum()

prescription_counts_5 = prescription_data.iloc[:, 2001:].sum()

print(time.time() - t)
prescription_counts = pd.concat([prescription_counts_1, prescription_counts_2, prescription_counts_3, prescription_counts_4, prescription_counts_5])

del prescription_counts_1

del prescription_counts_2

del prescription_counts_3

del prescription_counts_4

del prescription_counts_5
prescription_counts.head()
specialties = sorted(prescription_data.specialty.unique())
print(len(specialties))

specialties[0:10]
prescription_data_groupedby_specialty = prescription_data.groupby('specialty')
prescription_data_specialty_means = {}

t = time.time()

for specialty in prescription_data_groupedby_specialty.groups:

    counts = prescription_data_groupedby_specialty.get_group(specialty).sum()

    counts = pd.to_numeric(counts, downcast='float', errors='coerce').fillna(0).astype('float64')[:-8]

    n = (prescription_data.specialty == specialty).sum()

    prescription_data_specialty_means[specialty] = counts / n

    print(specialty, "{} instances of specialty grouped.".format(n), time.time() - t)
prescription_means_by_specialty = pd.DataFrame()
for specialty, means in prescription_data_specialty_means.items():

    prescription_means_by_specialty.loc[:, specialty] = means
prescription_means_by_specialty = prescription_means_by_specialty.fillna(0)

del prescription_data_specialty_means

del prescription_data_groupedby_specialty

prescription_means_by_specialty.memory_usage().sum()
prescription_means_by_specialty.shape
prescription_means_by_specialty.head()
# save created dataframe

# prescription_means_by_specialty.to_csv('prescription_means_by_specialty', header=True)
