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
df  = pd.read_csv(r'../input/german-house-prices/germany_housing_data.csv')
df.head()
df.drop('Unnamed: 0', axis=1, inplace=True)
df.dtypes
df.drop_duplicates(inplace=True)
df.drop_duplicates(inplace=True)
df.shape
import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(x=df.isnull().sum()/11973, y=df.columns)
sns.violinplot(df.Price)
sns.violinplot(df.Living_space)
sns.violinplot(df.Lot)
sns.violinplot(df.Usable_area)
sns.violinplot(df.Energy_consumption)
plt.xticks(rotation=90)
sns.countplot(df.Rooms)
sns.boxplot(df.Rooms)
plt.xticks(rotation=90)
sns.countplot(df.Bedrooms)
sns.boxplot(df.Rooms)
plt.xticks(rotation=90)
sns.countplot(df.Bathrooms)
sns.boxplot(df.Bathrooms)
sns.countplot(df.Floors)
sns.violinplot(df.Year_built)
sns.violinplot(df.Year_renovated)
plt.xticks(rotation=90)
sns.countplot(df.Garages)
plt.xticks(rotation=90)
sns.countplot(df.Type)
plt.xticks(rotation=90)
sns.countplot(df.Furnishing_quality)
plt.xticks(rotation=90)
sns.countplot(df.Condition)
plt.xticks(rotation=90)
sns.countplot(df.Heating)
plt.xticks(rotation=90)
sns.countplot(df.Energy_certificate)
plt.xticks(rotation=90)
sns.countplot(df.Energy_certificate_type)
plt.xticks(rotation=90)
sns.countplot(df.Energy_efficiency_class)
plt.xticks(rotation=90)
sns.countplot(df.State)
plt.xticks(rotation=90)
sns.countplot(df.Garagetype)
def correlation_heatmap(train):
    correlations = train.corr()

    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show();
correlation_heatmap(df)
sns.scatterplot(x='Living_space', y='Price', data=df)
sns.scatterplot(x='Lot', y='Price', data=df)
sns.scatterplot(x='Usable_area', y='Price', data=df)
sns.scatterplot(x='Floors', y='Price', data=df)
sns.scatterplot(x='Garages', y='Price', data=df)
sns.scatterplot(x='Rooms', y='Price', data=df)
sns.scatterplot(x='Energy_consumption', y='Price', data=df)
sns.scatterplot(x='Bathrooms', y='Price', data=df)
sns.scatterplot(x='Bedrooms', y='Price', data=df)
plt.xticks(rotation=90)
sns.stripplot(x='State', y='Price', data=df)
sns.barplot(x='Price', y='State', data=df)
sns.barplot(x='Energy_certificate', y='Price', data=df)
sns.stripplot(x='Energy_certificate', y='Price', data=df)
sns.barplot(x='Energy_certificate_type', y='Price', data=df)
sns.stripplot(x='Energy_certificate_type', y='Price', data=df)
plt.xticks(rotation=90)
sns.barplot(x='Type', y='Price', data=df)
plt.xticks(rotation=90)
sns.stripplot(x='Type', y='Price', data=df)
plt.xticks(rotation=90)
sns.barplot(x='Condition', y='Price', data=df)
plt.xticks(rotation=90)
sns.stripplot(x='Condition', y='Price', data=df)
plt.xticks(rotation=90)
sns.barplot(x='Garagetype', y='Price', data=df)
plt.xticks(rotation=90)
sns.stripplot(x='Garagetype', y='Price', data=df)
plt.xticks(rotation=90)
sns.stripplot(x='Energy_efficiency_class', y='Price', data=df)
plt.xticks(rotation=90)
sns.barplot(x='Energy_efficiency_class', y='Price', data=df)
plt.xticks(rotation=90)
sns.stripplot(x='Energy_efficiency_class', y='Price', data=df)
plt.xticks(rotation=90)
sns.barplot(x='Furnishing_quality', y='Price', data=df)
plt.xticks(rotation=90)
sns.stripplot(x='Furnishing_quality', y='Price', data=df)

plt.xticks(rotation=90)
sns.barplot(x='Heating', y='Price', data=df)
plt.xticks(rotation=90)
sns.stripplot(x='Heating', y='Price', data=df)
sns.stripplot(x='Energy_efficiency_class', y='Energy_consumption', data=df)
plt.xticks(rotation=90)
sns.stripplot(x='Heating', y='Energy_consumption', data=df)
sns.stripplot(x='Furnishing_quality', y='Living_space', data=df)
plt.xticks(rotation=90)
sns.stripplot(x='State', y='Living_space', data=df)
sns.lmplot(x='Living_space', y='Price', hue='State', data=df)
sns.lmplot(x='Living_space', y='Price', hue='Furnishing_quality', data=df)
sns.lmplot(x='Living_space', y='Price', hue='Type', data=df)
sns.lmplot(x='Living_space', y='Price', hue='Condition', data=df)
sns.lmplot(x='Living_space', y='Price', hue='Energy_efficiency_class', data=df)