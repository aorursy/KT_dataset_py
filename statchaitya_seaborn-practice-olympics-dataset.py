# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
olympics = pd.read_csv("../input/athlete_events.csv")
olympics.head()
olympics.shape
sns.distplot(olympics['Height'].dropna())
# Changing the figure size using pyplot
f, ax = plt.subplots(figsize=(15,5))
sns.distplot(olympics['Weight'].dropna())
# Changing the figure size using pyplot
f, ax = plt.subplots(figsize=(15,5))
# specifying bins and removing KDE
sns.distplot(olympics['Weight'].dropna(), bins=75, kde=False)
# Just showing KDE and not the histogram
f, ax = plt.subplots(figsize=(15,5))
sns.distplot(olympics['Weight'].dropna(), hist=False, kde=True)
f, ax = plt.subplots(figsize=(15,5))
# Setting a color for plot
sns.distplot(olympics['Weight'].dropna(), bins=50, kde=False, color="g")
f, ax = plt.subplots(figsize=(15,5))
# Setting a color for KDE plot
sns.kdeplot(olympics['Weight'].dropna(), shade=True, color="r")
f, ax = plt.subplots(figsize=(15,5))
# Multiple KDE plots on one graph
sns.kdeplot(olympics['Weight'].dropna(), color="r", label="Weight")
sns.kdeplot(olympics['Height'].dropna(), color="g", label="Height")
# A different kind of plot to compare two continuous variables
sns.jointplot(x="Weight", y="Height", data=olympics)
# Pairplot
sns.pairplot(olympics.dropna(), size=4)
# Joint plot with controlled limits
sns.jointplot(x="Weight", y="Height", data=olympics, xlim=(25,175), ylim=(140,200))
# Representing a third dimension color in a pairplot
sns.pairplot(olympics.dropna(), hue="Medal")
# Representing a regression line in the bivariate relationships in a pairplot
sns.pairplot(olympics[['Height', 'Weight', 'Age']].dropna(), kind="reg")
# Representing KDE plots instead of histograms on the diagonal
sns.pairplot(olympics[['Height', 'Weight', 'Age']].dropna(), diag_kind="kde")
# Representing correlations between various features in the data as a heatmap
corrmat = olympics.dropna().corr()
f, ax = plt.subplots(figsize=(10,10))
# annot controls annotations, square=True outputs squares as correlation representing figures, cmap represents color map
sns.heatmap(corrmat, vmax=.8, square=True, annot=True, fmt=".2f", cmap="summer")