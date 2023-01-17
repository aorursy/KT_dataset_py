import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from scipy.stats import ttest_ind

from subprocess import check_output





cereal_df = pd.read_csv('../input/cereal.csv')





df1 = cereal_df.sodium[cereal_df['type']=='C']

df2 = cereal_df.sodium[cereal_df['type']=='H']



df3 = cereal_df.sodium[(cereal_df['mfr'] == 'K')]

df4 = cereal_df.sodium[(cereal_df['mfr'] == 'G')]





ttest_ind(df3 , df4 ,equal_var=False)

ax = plt.subplots(figsize=(12,8))

sns.distplot(df3  ,color="gold" , bins=30 , axlabel='Sodium',label='Kelloggs').set_title("Frequency of Sodium")

sns.distplot(df4, color="teal" , bins=10 , axlabel='Sodium',label='General Mills')

plt.legend()
ttest_ind(df1 , df2 ,equal_var=False)

ax = plt.subplots(figsize=(12,8))

sns.distplot(df1  ,color="skyblue" , bins=30 , axlabel='Sodium',label='Cold').set_title("Frequency of Sodium in Hot and Cold cereals")

sns.distplot(df2, color="olive" , bins=10 , axlabel='Sodium',label='Hot')

plt.legend()