import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

from pandas import Series
import os
df_inspection = pd.read_csv ("../input/inspections.csv")

df_inspection.head()
print(df_inspection.describe())
fig, ax = plt.subplots(1,1)
df_inspection.hist(ax=ax, column='score')
df_inspection['facility_city'].value_counts()
df_inspection['facility_zip'].value_counts()
df_inspection['grade'].value_counts()
df_inspection['owner_name'].value_counts()
