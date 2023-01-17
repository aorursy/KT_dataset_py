# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import ggplot

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



menu = pd.read_csv('../input/menu.csv')

menu.head(5)



df = menu
print(menu.describe())
pd.pivot_table(df,index=['Category'],values=['Protein'],aggfunc=np.max).plot(kind='bar')
df.sort_values(by='Protein', ascending=False).head(10)
df['Protein/Sugar'] = np.where(df['Sugars'] < 1, df['Sugars'], df['Protein']/df['Sugars'])
df.sort_values(by='Protein/Sugar', ascending=False).head(10)