# Import Required Libraries

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt
# Check for availability of dataset

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Load Dataset in dataframe

df=pd.read_csv('../input/2017.csv')

df.head()
df[df['Country']=='Venezuela']
# Look the data summary

df.describe()
# Choose any numerical column

freedom = df['Freedom']
# Graph Distribution Plot of Column

sns.distplot(freedom,kde=False,bins=10,hist_kws={'lw':1,'edgecolor':'gray'}).set_title('Freedom Distribution Plot')

plt.grid()
# Choose other numerical column

generosity = df['Generosity']
# Graph Distribution Plot of Column

sns.distplot(generosity,bins=40,hist_kws={'lw':1,'edgecolor':'gray'}).set_title('Generosity Distribution Plot')

plt.grid()
df[df['Country']=='Venezuela']