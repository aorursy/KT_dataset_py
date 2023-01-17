import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Path of the file to read

filepath = "/kaggle/input/uk-hospitals/Hospital.csv"



# Read the file into df

df = pd.read_csv(filepath)

df.head()
df.isnull().sum()
predrop=df.shape

droppeddf=df.dropna()

postdrop=droppeddf.shape

print(f"Out of an original {predrop[0]} rows, a total of {predrop[0]-postdrop[0]} will be dropped.")
df.describe().T
sns.pairplot(df, vars=['OrganisationID','Latitude','Longitude'], diag_kind = 'kde', 

             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},

             height = 4)