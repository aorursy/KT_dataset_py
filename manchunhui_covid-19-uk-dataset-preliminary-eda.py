import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Path of the file to read

filepath = "/kaggle/input/covid19-uk-dataset/PCRTesting.csv"



# Read the file into df

df = pd.read_csv(filepath, parse_dates=['date'], dayfirst=True)

df.head().T
df.isnull().sum()
df.fillna('0')
df.describe().T
plt.figure(figsize=(18,6))

sns.set_style("ticks")

sns.boxplot(data=df[['plannedPCRCapacityByPublishDate',

                     'plannedAntibodyCapacityByPublishDate',

                     'newPCRTestsByPublishDate',

                     'newAntibodyTestsByPublishDate']], palette="deep")

sns.despine(offset=10, trim=True)