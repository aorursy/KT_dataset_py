import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv('../input/DigiDB_digimonlist.csv')

dataframe = data.describe()

sns.barplot(dataframe["Lv50 Int"],dataframe["Lv50 Def"],dataframe["Number"])