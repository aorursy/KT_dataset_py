import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv("../input/coronavirusdataset/Case.csv")
data.head()
data.info()
data.describe()
data.infection_case.unique()#returns only unique values
def group(col):

        if "Hospital" in col:

            return "Hospital"

        elif "Church" in col:

            return "Church"

        elif "Center" in col:

            return "Center"

        elif "Shelter" in col or "Nursing Home" in col:

            return "Shelter/Nursing Home"

        else:

            return col
data['place'] = data.infection_case.apply(group)
data
data.place.unique()
data.isnull().sum()
fig, ax = plt.subplots(figsize=(15,8))

chart = sns.countplot(data.place)

chart.set(ylabel="Cases")

chart = chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
fig, ax = plt.subplots(figsize=(6,8))

sns.countplot(data.group)
fig, ax = plt.subplots(figsize=(15,8))

chart = sns.countplot(data.city)

chart.set(ylabel="Cases")

chart = chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
fig, ax = plt.subplots(figsize=(10,8))

chart = sns.countplot(data.province)

chart.set(ylabel="Cases")

chart = chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
fig, ax = plt.subplots(figsize=(12,8))

chart = sns.countplot(data.province, hue=data.place=='Church')

ax.legend(labels=['Other', 'Church'])

chart.set(ylabel="Cases")

chart = chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
fig, ax = plt.subplots(figsize=(12,8))

chart = sns.countplot(data.province, hue=data.place=='overseas inflow')

ax.legend(labels=['Other', 'Overseas inflow'])

chart.set(ylabel="Cases")

chart = chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
fig, ax = plt.subplots(figsize=(12,8))

chart = sns.countplot(data.province, hue=data.place=='Hospital')

ax.legend(labels=['Other', 'Hospital'])

chart.set(ylabel="Cases")

chart = chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
fig, ax = plt.subplots(figsize=(12,8))

chart = sns.countplot(data.province, hue=data.place=='contact with patient')

ax.legend(labels=['Other', 'Hospital'])

chart.set(ylabel="Cases")

chart = chart.set_xticklabels(chart.get_xticklabels(), rotation=90)