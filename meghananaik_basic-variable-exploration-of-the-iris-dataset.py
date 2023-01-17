# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import scipy

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv("../input/Iris.csv") 

df.head(2)

df.info()
df.describe()
# Displaying the number of Nan's in each column

labels= []

value = []

for col in df.columns:

    labels.append(col)

    value.append(df[col].isnull().sum())

    print(col,value[-1])

# Counting the number of species in the dataset

df['Species'].value_counts()
rel_df = pd.read_csv("../input/Iris.csv") 

del rel_df['Id']

rel = rel_df.corr()

sns.heatmap(rel,square= True)

plt.yticks(rotation=0)

plt.xticks(rotation=90)
