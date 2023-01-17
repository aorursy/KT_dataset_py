# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df = pd.read_csv("../input/starbucks_drinkMenu_expanded.csv")

# Any results you write to the current directory are saved as output.
calories = df.Calories
sns.set_style("white")
plt.hist(calories)
plt.title("Calories histogram")
sugar = df[" Sugars (g)"]
plt.hist(sugar)
plt.title("Sugar histogram")
x = calories
y = sugar
grid = sns.JointGrid(x, y, height=8, ratio=50)
grid.plot_joint(plt.scatter, color="g")
grid.plot_marginals(sns.rugplot, height=2, color="r")
plt.figure(figsize=(10,8))
bar_cal = sns.barplot(x="Calories", y="Beverage_category", data=df)
plt.ylabel("")
plt.figure(figsize=(10,4))
sizes = df[(df.Beverage_prep == 'Short') | (df.Beverage_prep == 'Venti') | (df.Beverage_prep == 'Grande') | (df.Beverage_prep == 'Tall')]
bar_sizes = sns.barplot(x="Calories", y="Beverage_prep", data=sizes)
plt.ylabel("")
plt.figure(figsize=(10,3))
milk = df[(df.Beverage_prep.str.contains('milk')) | (df.Beverage_prep.str.contains('Milk')) & ~((df.Beverage_prep.str.contains('Short')) | (df.Beverage_prep.str.contains('Grande')) | (df.Beverage_prep.str.contains('Venti')) | (df.Beverage_prep.str.contains('Tall')))]
bar_milk = sns.barplot(x="Calories", y="Beverage_prep", data=milk)
plt.ylabel("")