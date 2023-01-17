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
df = pd.read_csv('../input/fertilizers-by-product-fao/FertilizersProduct.csv', encoding='latin-1')
df.head()
df.info()
df.describe()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(10,6))
plt.tight_layout()
sns.set_style('whitegrid')
sns.countplot(x = 'Year', data=df)
India = df[df['Area'] == "India"]
India.head()
len(India.Item.value_counts())
India.Year.hist(bins=20)
plt.figure(figsize=(10,6))
plt.tight_layout()
sns.countplot(x = 'Year', data=India)
plt.figure(figsize=(10,6))
plt.tight_layout()
sns.countplot(x = 'Year', data=India, hue = 'Element')
plt.figure(figsize=(10,6))
plt.tight_layout()
plot = sns.countplot(x = 'Item', data=India, hue = 'Element')
plot.set_xticklabels(plot.get_xticklabels(), rotation=90, horizontalalignment='center')
plot.plot()
plt.figure(figsize=(10,8))
plot = sns.countplot(x = "Item", data = India[India['Element'] == "Agricultural Use"])
plot.set_xticklabels(plot.get_xticklabels(), rotation=90, horizontalalignment='center')
plot.plot()

plt.figure(figsize=(10,8))
plot = sns.countplot(x = "Item",  data = India[India['Element'] == "Export Value"])
plot.set_xticklabels(plot.get_xticklabels(), rotation=90, horizontalalignment='center')
plot.plot()

plt.figure(figsize=(10,8))
plot = sns.countplot(x = "Item",  data = India[India['Element'] == "Import Value"])
plot.set_xticklabels(plot.get_xticklabels(), rotation=90, horizontalalignment='center')
plot.plot()
plt.figure(figsize=(10,8))
plot = sns.countplot(x = "Item",  data = India[India['Element'] == "Production"])
plot.set_xticklabels(plot.get_xticklabels(), rotation=90, horizontalalignment='center')
plot.plot()
data = India[India['Element'] == "Export Value"]
data.head()
plt.figure(figsize=(8,8))
sns.lineplot(x='Year', y='Value', data=data)
plt.figure(figsize=(8,8))
plot = sns.barplot(x="Item", y="Value", data=data)
plot.set_xticklabels(plot.get_xticklabels(), rotation=90, horizontalalignment='center')
plot.plot()
importdata = India[India['Element'] == "Import Value"]
sum(importdata['Value'])
sum(exportdata['Value'])
plt.figure(figsize=(8,8))
sns.lineplot(x='Year', y='Value', data=importdata)
plt.figure(figsize=(10,10))
sns.lineplot(x='Year', y='Value', data=importdata, hue="Item")
plt.figure(figsize=(8,8))
plot = sns.barplot(x="Item", y="Value", data=importdata)
plot.set_xticklabels(plot.get_xticklabels(), rotation=90, horizontalalignment='center')
plot.plot()
agridata = India[India['Element'] == "Agricultural Use"]
agridata.head()
plt.figure(figsize=(8,8))
plot = sns.barplot(x="Item", y="Value", data=agridata)
plot.set_xticklabels(plot.get_xticklabels(), rotation=90, horizontalalignment='center')
plot.plot()
plt.figure(figsize=(8,8))
sns.lineplot(x='Year', y='Value', data=agridata)
df['Element'].unique()
exportdata = India[India['Element'] == "Export Quantity"]
plt.figure(figsize=(8,8))
plot = sns.barplot(x="Item", y="Value", data=exportdata)
plot.set_xticklabels(plot.get_xticklabels(), rotation=90, horizontalalignment='center')
plot.plot()
plt.figure(figsize=(8,8))
sns.lineplot(x='Year', y='Value', data=exportdata)
proddata = India[India['Element'] == "Production"]
proddata.head()
plt.figure(figsize=(8,8))
sns.lineplot(x='Year', y='Value', data=proddata)
plt.figure(figsize=(8,8))
plot = sns.barplot(x="Item", y="Value", data=proddata)
plot.set_xticklabels(plot.get_xticklabels(), rotation=90, horizontalalignment='center')
plot.plot()
plt.figure(figsize=(10,10))
sns.lineplot(x='Year', y='Value', data=proddata, hue=proddata['Item'])
sum(proddata['Value'])
df['Item'].unique()