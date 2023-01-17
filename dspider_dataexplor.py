# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv") 
df_train.shape
df_train.head()
molecule_name_counts = df_train["molecule_name"].value_counts()

molecule_name_counts
print("Max: " + molecule_name_counts.idxmax() + ": " + str(molecule_name_counts[molecule_name_counts.idxmax]))

print("Min: " + molecule_name_counts.idxmin() + ": " + str(molecule_name_counts[molecule_name_counts.idxmin]))
molecule_str = pd.DataFrame(molecule_name_counts.keys().str.split("_").tolist(), columns=["str", "digit"])
molecule_str.head()
molecule_str["str"].value_counts()
molecule_str_train = pd.DataFrame(df_train["molecule_name"].str.split("_").tolist(), columns=["str", "digit"])

df_train["molecule_digit"] = molecule_str_train["digit"]
df_train.head()
print("Max: " + str(df_train["scalar_coupling_constant"].loc[df_train["scalar_coupling_constant"].idxmax()]))

print("Min: " + str(df_train["scalar_coupling_constant"].loc[df_train["scalar_coupling_constant"].idxmin()]))
plt.figure(figsize=(15,20))

df_train["scalar_coupling_constant"].plot.hist(bins=range(-50, 240, 10))
# Check values between 20 & 70. 

plt.figure(figsize=(15,20))

df_train.loc[ (df_train["scalar_coupling_constant"] > 20) & (df_train["scalar_coupling_constant"] < 70)]["scalar_coupling_constant"].plot.hist(bins=range(20, 70, 10))
# Check values between 20 & 70. 

plt.figure(figsize=(15,20))

df_train.loc[ (df_train["scalar_coupling_constant"] > 150) & (df_train["scalar_coupling_constant"] < 190)]["scalar_coupling_constant"].plot.hist(bins=range(150, 190, 10))
from matplotlib import pyplot as plt

plt.figure(figsize=(15,20))

df_train["type"].value_counts().plot.bar()
struct_df = pd.read_csv("../input/structures.csv")
struct_df.shape
display(struct_df.head())