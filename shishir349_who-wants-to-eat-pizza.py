!pip install dabl
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import dabl

import plotly.express as px
import os

print(os.listdir('../input/can-pizza-be-healthy'))
data = pd.read_csv('../input/can-pizza-be-healthy/Pizza.csv')

data.head()
# lets check the the unique no. of pizza brands over there

data['brand'].nunique()
# Lets check the Distribution of these Brands



plt.rcParams['figure.figsize'] = (18, 7)

plt.style.use('fivethirtyeight')



sns.countplot(data['brand'], palette = 'Reds')

plt.title('Distributions of Different Pizza Brands')

plt.show()
# lets compare the data with respect to Calories

dabl.plot(data, target_col = 'cal')
# Lets check which Brand is most calorie prone



sns.boxenplot(data['brand'], data['cal'], palette ='copper')

plt.title('Brand vs Calories')

plt.show()
# Lets check which Brand is most fat prone



sns.violinplot(data['brand'], data['fat'], palette ='Greens')

plt.title('Brand vs Fats')

plt.show()
# Lets check which Brand is most ash prone



sns.swarmplot(data['brand'], data['ash'], palette ='Blues')

plt.title('Brand vs Ashes')

plt.show()
# Lets check which Brand makes most Moisturous pizzas



sns.boxplot(data['brand'], data['mois'], palette ='pink')

plt.title('Brand vs Moisture')

plt.show()
# Lets check which Brand is most sodium prone



sns.stripplot(data['brand'], data['sodium'], palette ='bone')

plt.title('Brand vs Sodium')

plt.show()
# Lets check which Brand is most calorie prone



sns.barplot(data['brand'], data['carb'], palette ='spring')

plt.title('Brand vs Carbohydrates')

plt.show()
data = data.drop(['id'], axis = 1)
data = data.set_index('brand')

data.head()
data.describe()
plt.rcParams['figure.figsize'] = (15, 10)

data.plot(kind ='line', color = ['black', 'darkblue', 'maroon', 'grey', 'violet','orange','darkgreen'])

plt.title('Comparing all at once')

plt.show()