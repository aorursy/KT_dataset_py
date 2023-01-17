# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/movie_metadata.csv")
columns_names=df.columns.tolist()

print("columns_names:")

print(columns_names)
df.shape

df.head(10)
df.corr()
correlation=df.corr()

plt.figure(figsize=(10,10))

sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')

plt.title=('Correlation between gross and directors')
df['gross'].mean()
df['gross'].median()
df["director_name"].value_counts().head(10)
df["genres"].value_counts().head(15)
dir_df=df[['director_name','gross']]

print(dir_df.head(10))