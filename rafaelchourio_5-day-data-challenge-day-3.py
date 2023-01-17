# Import Required Libraries

import numpy as np 

import pandas as pd 

from scipy import stats

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Check for availability of datasets

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Load Dataset in dataframe

df=pd.read_csv('../input/cereal.csv')

# Look at the features

df.describe()
# See the first 5 data points

df.head().round(2)
df['type'].unique()
typeC=df[df['type']=='C']['sugars']
typeH=df[df['type']=='H']['sugars']
typeC.var()
typeH.var()
stats.ttest_ind(typeC,typeH,equal_var=False)
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(12,6))

sns.distplot(typeC,ax=ax1).set_title('Cereal Type C')

sns.distplot(typeH,ax=ax2).set_title('Cereal Type H')
