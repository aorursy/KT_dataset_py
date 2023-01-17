# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv("../input/abalone/abalone.csv")[:150]
list(dataset.columns)
dataset.shape
dataset.head(5)
dataset.tail(5)
pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

import seaborn as sns
dataset.describe()
plt.figure(figsize = (14, 14)) # Set width to 7 inches and height to 5 inches

plt.title("Line plot of attributes in abalone") # Set title

sns.lineplot(data = dataset.iloc[:,1:6])
plt.figure(figsize = (7, 5))

plt.title("Line plot of attributes in abalone")

sns.lineplot(data = dataset["Length"], label = "Length")

sns.lineplot(data = dataset["Rings"], label = "Rings")

plt.xlabel("Length")

plt.ylabel("Number of crimes")
plt.figure(figsize = (7, 5))

plt.title("Line plot of attributes in abalone")

sns.lineplot(data = dataset["Diameter"], label = "Diameter")

sns.lineplot(data = dataset["Rings"], label = "Rings")

plt.xlabel("Diameter")

plt.ylabel("Number of crimes")
plt.figure(figsize = (7, 5))

plt.title("Line plot of attributes in abalone")

sns.lineplot(data = dataset["Height"], label = "Height")

sns.lineplot(data = dataset["Rings"], label = "Rings")

plt.xlabel("Height")

plt.ylabel("Number of crimes")
plt.figure(figsize = (9, 5))

plt.title("histogram of attributes in abalone")

sns.barplot(x = dataset.Rings, y = dataset.Robbery)

plt.xlabel("Attributes")
# code for task 3