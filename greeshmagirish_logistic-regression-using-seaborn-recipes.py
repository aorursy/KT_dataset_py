import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import os



%matplotlib inline
recipes = pd.read_csv("../input/epi_r.csv")
recipes = recipes[recipes['calories'] < 10000].dropna()
sns.set(style="whitegrid")

g = sns.regplot(x="calories", y="dessert", data=recipes, fit_reg=False)

g.figure.set_size_inches(10, 10)
sns.set(style="whitegrid")

g = sns.regplot(x="calories", y="dessert", data=recipes, logistic=True)

g.figure.set_size_inches(10,10)