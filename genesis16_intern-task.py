import warnings  

warnings.filterwarnings('ignore')
import os

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
print(os.listdir("../input"))
# Loading csv file

data = pd.read_csv("../input/heart.csv")

data.head(5)
# Checking shape of data

# 303 rows and 14 features

data.shape
# Checking feature names

for col in data.columns:

    print(col)
# Mapping data for easy readibility

data['sex'] = data['sex'].map({ 1 : 'Male', 0 : 'Female'})

data['exang'] = data['exang'].map({ 1 : 'Yes', 0 : 'No'})

data.head(5)
data.describe()
# Checking if there are missing values in the dataset

data.isnull().sum()
# Checking for data types

data.dtypes
# Dataset summary

data.info()
# Checking distributing of sex in dataset

sns.countplot(x='sex', data=data)

print(data['sex'].value_counts())

# This dataset has twice as many male patients compared to female ones
# Count plot of chest pain types

sns.countplot(x=data['cp'])

# Chest pain type 0 seems to be the most common followed by type 2

print(data['cp'].value_counts())
# Count plot of excercise induced angina(exang) among both genders

sns.countplot(x=data['exang'], hue=data['sex'])

# exang seem to be considerably more in males

print(data['exang'].value_counts())
# slope vs age

sns.boxplot(x='slope', y='age', hue='target', data=data)
# ca vs age

sns.swarmplot(x='ca', y='age', hue='target', data=data)
# thal vs age

sns.violinplot(x='thal', y='age', hue='target', data=data)
# Distribution plot of continuous features

continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']



for i, feature in enumerate(continuous_features):

    sns.distplot(data[feature], kde=True)

    plt.show()
# Heatmap of dataset

corr = data.corr()

_, ax = plt.subplots(figsize=(12,12))

sns.heatmap(corr, cbar=True, square=True, annot=True, cmap='Spectral', ax=ax)

# Checking correlation of target with other columns

print(data.corr()["target"].abs().sort_values(ascending=False))
# Comparing age vs presence of heart disease among both genders

sns.countplot(x='target', hue='sex', data=data)
# Combined plot of chol, thalach, age and trestbps

data.chol.plot(kind="line", color="green", label="chol", grid=True, linestyle=":", figsize= (15,10))

data.thalach.plot(kind="line", color="purple", label="thalach", grid=True, figsize= (15,10))

data.age.plot(kind="line", color="pink", label="age", grid=True, figsize= (15,10))

data.trestbps.plot(kind="line", color="orange", label="trestbps", grid=True, figsize= (15,10))

plt.legend(loc="upper right")

plt.xlabel("indexes")

plt.ylabel("Features")

plt.title("Heart Diseases")

plt.show()
# Pairwise plot of age, trestbps, chol, thalach and oldpeak

sns.pairplot(data, hue = 'target', markers=["o", "s"], vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])