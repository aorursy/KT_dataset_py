import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("darkgrid")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/all-space-missions-from-1957/Space_Corrected.csv")

df.head()
df.info()
df.shape
# Extract the launch year

df['DateTime'] = pd.to_datetime(df['Datum'])

df['Year'] = df['DateTime'].apply(lambda datetime: datetime.year)



# Extract the country of launch

df["Country"] = df["Location"].apply(lambda location: location.split(", ")[-1])



df.head(10)
# Company vs Number of launches

plt.figure(figsize=(8,18))

ax = sns.countplot(y="Company Name", data=df, order=df["Company Name"].value_counts().index)

ax.set_xscale("log")

ax.axes.set_title("Company vs. # Launches (Log Scale)",fontsize=18)

ax.set_xlabel("Number of Launches (Log Scale)",fontsize=16)

ax.set_ylabel("Company Name",fontsize=16)

ax.tick_params(labelsize=12)

plt.tight_layout()

plt.show()
# Country vs Number of launches

plt.figure(figsize=(8,8))

ax = sns.countplot(y="Country", data=df, order=df["Country"].value_counts().index)

ax.set_xscale("log")

ax.axes.set_title("Country vs. # Launches (Log Scale)",fontsize=18)

ax.set_xlabel("Number of Launches (Log Scale)",fontsize=16)

ax.set_ylabel("Country",fontsize=16)

ax.tick_params(labelsize=12)

plt.tight_layout()

plt.show()
# Location vs Number of launches

import warnings

warnings.filterwarnings("ignore")

plt.figure(figsize=(12,36))

ax = sns.countplot(y="Location", data=df, order=df["Location"].value_counts().index)

ax.set_xscale("log")

ax.axes.set_title("Location vs. # Launches (Log Scale)",fontsize=18)

ax.set_xlabel("Number of Launches (Log Scale)",fontsize=16)

ax.set_ylabel("Location",fontsize=16)

ax.tick_params(labelsize=12)

plt.tight_layout()

plt.show()
# Year vs Number of launches

plt.figure(figsize=(8,18))

ax = sns.countplot(y=df['Year'])

ax.axes.set_title("Year vs. # Launches",fontsize=18)

ax.set_xlabel("Year",fontsize=16)

ax.set_ylabel("# Launches",fontsize=16)

ax.tick_params(labelsize=12)

plt.tight_layout()

plt.show()
# Rocket status vs Count

plt.figure(figsize=(6,6))

ax = sns.countplot(x="Status Rocket", data=df, order=df["Status Rocket"].value_counts().index, palette="pastel")

ax.axes.set_title("Rocket Status vs. Count",fontsize=18)

ax.set_xlabel("Count",fontsize=16)

ax.set_ylabel("Rocket Status",fontsize=16)

ax.tick_params(labelsize=12)

plt.tight_layout()

plt.show()
# Mission status vs Count

plt.figure(figsize=(6,6))

ax = sns.countplot(y="Status Mission", data=df, order=df["Status Mission"].value_counts().index, palette="Set2")

ax.set_xscale("log")

ax.axes.set_title("Mission Status vs. Count",fontsize=18)

ax.set_xlabel("Count",fontsize=16)

ax.set_ylabel("Mission Status",fontsize=16)

ax.tick_params(labelsize=12)

plt.tight_layout()

plt.show()