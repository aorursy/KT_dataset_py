import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
dataset = pd.read_csv('../input/Grid_Disruption_00_14_standardized - Grid_Disruption_00_14_standardized.csv')
dataset.head()
print("Number of entries: " + str(len(dataset.index)))
len(pd.to_numeric(dataset['Year'], 'coerce').dropna().astype(int))
len(pd.to_numeric(dataset['Demand Loss (MW)'], 'coerce').dropna().astype(int))
len(pd.to_numeric(dataset['Number of Customers Affected'], 'coerce').dropna().astype(int))
print('Demand Loss (MW)')
dataset.iloc[:, 9]
dataset = dataset.iloc[pd.to_numeric(dataset['Demand Loss (MW)'], 'coerce').dropna().astype(int).index, :]
print(len(dataset.index))
print('Number of Customers Affected')
dataset.iloc[:, 10]
dataset = dataset[dataset.columns.difference(['Number of Customers Affected'])]
for column in dataset.columns:
    dataset[column].replace('Unknown', None, inplace=True)
dataset.isnull().any()
print("Total number of rows: " + str(len(dataset.index)))
print("Number of empty values:")
for column in dataset.columns:
    print(" * " + column + ": " + str(dataset[column].isnull().sum()))
dataset = dataset.dropna()
dataset = dataset[dataset.columns.difference(['Event Description'])]

print("Total number of rows: " + str(len(dataset.index)))
print("Number of empty values:")
for column in dataset.columns:
    print(" * " + column + ": " + str(dataset[column].isnull().sum()))
dataset.head()
dataset.loc[dataset['Tags'].str.contains('severe weather', case=False), 'Tags'] = 'severe weather'
dim = (12, 30)
fig, ax = plt.subplots(figsize=dim)
sns.swarmplot(x="Year", y="Tags", ax=ax, data=dataset)
dim = (40, 10)
fig, ax = plt.subplots(figsize=dim)
demand_plot = sns.lvplot(x="Demand Loss (MW)", y="Year", ax=ax, data=dataset)

for item in demand_plot.get_xticklabels():
    item.set_rotation(45)
dim = (30, 10)
fig, ax = plt.subplots(figsize=dim)
tag_plot = sns.countplot(x="Tags", ax=ax, data=dataset)

for item in tag_plot.get_xticklabels():
    item.set_rotation(45)
dim = (20, 10)
fig, ax = plt.subplots(figsize=dim)
sns.countplot(x="Year", ax=ax, data=dataset)