import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
master = pd.read_csv('/kaggle/input/coronavirus-covid19-drug-discovery/master_results_table.csv')

sars_med = pd.read_csv('/kaggle/input/coronavirus-covid19-drug-discovery/sars_results.csv')

mers_med = pd.read_csv('/kaggle/input/coronavirus-covid19-drug-discovery/mers_results.csv')
master
sars_med
mers_med
train_X = sars_med.merge(mers_med, on='Unnamed: 0')
train_X
train_X['Unnamed: 0']