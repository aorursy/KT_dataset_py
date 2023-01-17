import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt

print(os.listdir("../input"))

# Reading the file from January 15 to June 15
df = pd.read_csv('../input/uber-raw-data-janjune-15.csv')

# Reading other FHV service Area Codes
ofhvser = pd.read_csv('../input/other-FHV-services_jan-aug-2015.csv')
# ofhvser['Base Number'].unique().tolist()
uber_used_set = set(df.Affiliated_base_num.unique())
otherfhv_set = set(ofhvser['Base Number'])
# Getting names of bases
ofhvser['Base Number'] = ofhvser['Base Number'].str.strip()
base_name_pairs = ofhvser['Base Name']
base_name_pairs.index = ofhvser['Base Number']
base_name_pairs = base_name_pairs.drop_duplicates()
# Finding Bases that Uber has been Using
uber_affiliation_bases = pd.Series(df.groupby('Affiliated_base_num').count()['Dispatching_base_num'])

# Finding overlapping areas that Uber called affiliated rides with
uber_business_overlap = uber_affiliation_bases.loc[base_name_pairs.index].dropna()

# Plotting businesses that were lost to Uber by more than 5000 rides
uber_business_overlap[uber_business_overlap>5000].sort_values().plot(kind='bar')
plt.ylabel('Number of Rides')

# The list of businesses that overlapped can be found with base_name_pairs


# Changing index names to company names
uber_business_overlap_names = uber_business_overlap[uber_business_overlap>5000].sort_values()

indexes = list()
for row in uber_business_overlap_names.items():
    indexes.append(base_name_pairs.to_dict()[row[0]])
uber_business_overlap_names.index = indexes

# Plotting
uber_business_overlap_names.plot(kind='bar')