# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

%matplotlib inline



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/responses.csv")

df = df.dropna()

df.head(10)



ages = df.Age

metal_or_hardrock = df["Metal or Hardrock"]

unique_ages = df.Age.unique()

unique_ages
sns.barplot(df.Age, df["Metal or Hardrock"])
df = df[["Age", "Metal or Hardrock", "Only child"]]

correlation = df.corr()

plt.figure(figsize=(10,10))

sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')



plt.title('Correlation between different features')