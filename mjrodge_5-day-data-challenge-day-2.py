# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Import tools

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Read data

drinks_Dataset = pd.read_csv("../input/starbucks_drinkMenu_expanded.csv")

drinks_Dataset.describe()



# Histogram of numeric value

calories = drinks_Dataset["Calories"]

plt.hist(calories)

plt.title("Starbucks Calories")