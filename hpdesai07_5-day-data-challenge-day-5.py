import numpy as np

import pandas as pd

import scipy.stats as st

import seaborn as sns



data = pd.read_csv('../input/DigiDB_digimonlist.csv')

dataframe = data.describe()

# print (dataframe)

arr1 = st.chi2_contingency(dataframe['Memory'])

# arr2 = st.chi2_contingency(dataframe['Equip Slots'])

# arr3 = st.chi2_contingency(dataframe['Number'])

print (arr1)

sns.barplot(arr1[3])

# print (arr2)

# print (arr3)

# print (data['Type'])

# print (data['Attribute'])