# Regular Imports

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("darkgrid")



import warnings

warnings.filterwarnings("ignore")
space_missions = pd.read_csv("../input/all-space-missions-from-1957/Space_Corrected.csv")

space_missions.head(10)
space_missions['DateTime'] = pd.to_datetime(space_missions['Datum'])



# Extract the launch year

space_missions['Year'] = space_missions['DateTime'].apply(lambda datetime: datetime.year)



# Extract the country of launch

space_missions["Country"] = space_missions["Location"].apply(lambda location: location.split(", ")[-1])



space_missions.head(10)
plt.figure(figsize=(8,18))

ax = sns.countplot(y="Company Name", data=space_missions, order=space_missions["Company Name"].value_counts().index)

ax.set_xscale("log")

ax.axes.set_title("Company vs. # Launches (Log Scale)",fontsize=18)

ax.set_xlabel("Number of Launches (Log Scale)",fontsize=16)

ax.set_ylabel("Company Name",fontsize=16)

ax.tick_params(labelsize=12)

plt.tight_layout()

plt.show()
plt.figure(figsize=(8,8))

ax = sns.countplot(y="Country", data=space_missions, order=space_missions["Country"].value_counts().index)

ax.set_xscale("log")

ax.axes.set_title("Country vs. # Launches (Log Scale)",fontsize=18)

ax.set_xlabel("Number of Launches (Log Scale)",fontsize=16)

ax.set_ylabel("Country",fontsize=16)

ax.tick_params(labelsize=12)

plt.tight_layout()

plt.show()
plt.figure(figsize=(12,36))

ax = sns.countplot(y="Location", data=space_missions, order=space_missions["Location"].value_counts().index)

ax.set_xscale("log")

ax.axes.set_title("Location vs. # Launches (Log Scale)",fontsize=18)

ax.set_xlabel("Number of Launches (Log Scale)",fontsize=16)

ax.set_ylabel("Location",fontsize=16)

ax.tick_params(labelsize=12)

plt.tight_layout()

plt.show()
plt.figure(figsize=(6,6))

ax = sns.countplot(x="Status Rocket", data=space_missions, order=space_missions["Status Rocket"].value_counts().index, palette="pastel")

ax.axes.set_title("Rocket Status vs. Count",fontsize=18)

ax.set_xlabel("Count",fontsize=16)

ax.set_ylabel("Rocket Status",fontsize=16)

ax.tick_params(labelsize=12)

plt.tight_layout()

plt.show()
plt.figure(figsize=(6,6))

ax = sns.countplot(y="Status Mission", data=space_missions, order=space_missions["Status Mission"].value_counts().index, palette="Set2")

ax.set_xscale("log")

ax.axes.set_title("Mission Status vs. Count",fontsize=18)

ax.set_xlabel("Count",fontsize=16)

ax.set_ylabel("Mission Status",fontsize=16)

ax.tick_params(labelsize=12)

plt.tight_layout()

plt.show()
plt.figure(figsize=(8,18))

ax = sns.countplot(y=space_missions['Year'])

ax.axes.set_title("Year vs. # Launches",fontsize=18)

ax.set_xlabel("Year",fontsize=16)

ax.set_ylabel("# Launches",fontsize=16)

ax.tick_params(labelsize=12)

plt.tight_layout()

plt.show()