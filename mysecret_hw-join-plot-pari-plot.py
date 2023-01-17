import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
dataset = pd.read_csv('../input/housing.csv')
dataset.fillna(method='ffill', inplace=True)
dataset.head()
dataset = dataset[['ocean_proximity','total_rooms','total_bedrooms','median_income','median_house_value']]
import seaborn as sns

sns.set(style="dark", color_codes=True)

g = sns.jointplot(x="total_rooms", y="total_bedrooms", data=df)
g = sns.jointplot(x="median_income", y="median_house_value", data=df)
g = sns.pairplot(dataset, hue="ocean_proximity")