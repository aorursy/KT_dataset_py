# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn

from scipy import stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



df=pd.read_csv("/kaggle/input/survey_results_public.csv")

print(df["Salary"].describe())

print(df["Salary"].median())

print(df["Salary"].mode())

print(df["Salary"].std())

alpha = 0.05

pv = stats.shapiro(df["Salary"].dropna())

pv = stats.kstest(df["Salary"].dropna(), 'norm')

if pv[0] > alpha: print("distributie normala")

df["Salary"].plot(kind="hist")

#df["Salary"].plot(kind="line")



# Any results you write to the current directory are saved as output.