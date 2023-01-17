# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv")
df.drop(labels=["Unnamed: 0"],axis=1,inplace=True)

df.head(10)
sns.factorplot(x = "brand",y="price",data=df,kind="bar",size=6)

plt.xticks(rotation=90)

plt.show()
df["model"] = [1 if i == "door" else 2 if i == "f-150" else 3 for i in df["model"]]
sns.factorplot(x = "model",y="price",data=df,kind="bar",size=5)

plt.xticks(rotation=90)

plt.show()
sns.factorplot(x = "year",y="price",data=df,kind="bar")

plt.xticks(rotation=90)

plt.show()
sns.factorplot(x = "color",y="price",data=df,kind="bar",size=7)

plt.xticks(rotation=90)

plt.show()
df["color_class"] = [1 if i == "white" else 2 if i == "black" else 3 for i in df["color"]]

df.head()
sns.factorplot(x = "color_class",y="price",data=df,kind="bar",size=7)

plt.xticks(rotation=90)

plt.show()
sns.factorplot(x = "state",y="price",data=df,kind="bar",size=7)

plt.xticks(rotation=90)

plt.show()