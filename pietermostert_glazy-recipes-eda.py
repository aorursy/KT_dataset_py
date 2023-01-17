import numpy as np

import pandas as pd

import json

from matplotlib import pyplot as plt

import seaborn as sns

import os

print(os.listdir('../input/glazy-data-june-2019'))
# Find out what the columns are

data0 = pd.read_csv("../input/glazy-data-june-2019/glazy_data_june_2019.csv", nrows=0)

for i, col in enumerate(list(data0.columns)):

    print(i, col)
percent_col_indices = list(range(18, 79))

umf_col_indices = list(range(79, 140))

xumf_col_indices = list(range(140, 201))

percent_mol_col_indices = list(range(262, 323))

ox_percent_mols = data0.columns[percent_mol_col_indices].tolist()
data = pd.read_csv("../input/glazy-data-june-2019/glazy_data_june_2019.csv", 

                   usecols=[0,1,3,12,13,14,15] + percent_mol_col_indices)  # Only selecting percent mol

display(data.head())   # Show the first 5 rows
# Drop analyses and primitive materials:

data = data[(~(data["is_analysis"]==1)) & ~(data["is_primitive"]==1)]    

data.drop(columns=["is_analysis","is_primitive"], inplace=True)



# Create a dataframe consisting of those rows in data for which the entries under ox_percent_mols occur 

# elsewhere:

duplicates = data[data.duplicated(subset=ox_percent_mols, keep=False)] 



data_dup = duplicates.groupby(ox_percent_mols)



# Concatenate the indices of recipes with the same oxide compositions

dup_list = data_dup.apply(lambda s: s['id'].astype(np.str).str.cat(sep=', ')).values.tolist()



for s in dup_list:

    print(s)
n1 = data.shape[0]

data = data.drop_duplicates(subset=ox_percent_mols)  # Should take into account duplicate glazes with \

                                                     # different cone ranges

n2 = data.shape[0]

print("Number of duplicates dropped:", n1 - n2)
data.set_index("id", inplace=True)

data = data.loc[data["material_type_id"].between(460, 1170)]   # Select glaze recipes
threshold = 0.5

for ox in ['SiO2', 'Al2O3', 'K2O', 'Na2O', 'CaO', 'MgO', 'B2O3']:

    oxide_mp = data[ox+'_percent_mol']

    hist = sns.distplot(oxide_mp[oxide_mp >= threshold], bins=30)

    plt.show()