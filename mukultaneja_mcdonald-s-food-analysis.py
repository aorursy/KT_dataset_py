# import useful libraries



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

%matplotlib inline

from matplotlib import pyplot as plt



# setting options

pd.options.mode.chained_assignment = None

sns.set(font_scale=1.2)
# reading data file

data = pd.read_csv('../input/menu.csv', index_col=0)

data = data.reset_index()
# info for every single column

data.info()
# statistic information about dataset

data.describe(include='all')
fig, ax = plt.subplots(figsize=(11, 5))



sns.heatmap(data.corr(), ax=ax)

ax.set_title('Correlation between columns')
neturicious = ['Protein', 'Vitamin A (% Daily Value)',

               'Dietary Fiber', 'Vitamin C (% Daily Value)',

               'Calcium (% Daily Value)', u'Iron (% Daily Value)']



nonneturicious = ['Total Fat', 'Calories from Fat', 'Saturated Fat', 'Trans Fat', 'Cholesterol']



data['neturicious'] = data['Protein'] + data['Vitamin A (% Daily Value)'] + data['Vitamin C (% Daily Value)'] + data['Dietary Fiber'] + data['Calcium (% Daily Value)'] + data['Iron (% Daily Value)']

data['nonneturicious'] = data['Total Fat'] + data['Calories from Fat'] + data['Saturated Fat'] + data['Trans Fat'] + data['Cholesterol']
# selecting Breakfast category to show neutricious food



df = data[data['Category'] == 'Breakfast']

df = df.groupby(['Item']).sum().sort_values(by='neturicious', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(7, 5))

df = df[neturicious]

sns.heatmap(df, ax=ax, annot=True)

ax.set_title('What to choose for Breakfast')

plt.xticks(rotation=45)
# selecting Breakfast category to show non-neutricious food



df = data[data['Category'] == 'Breakfast']

df = df.groupby(['Item']).sum().sort_values(by='nonneturicious', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(9, 5))

df = df[nonneturicious]

sns.heatmap(df, ax=ax, annot=True)

ax.set_title('What not to choose for Breakfast')

plt.xticks(rotation=45)
# selecting Beef & Pork category to show neutricious food



df = data[data['Category'] == 'Beef & Pork']

# removing this item because of unicode error if any

df = df[df['Item'] != 'Jalapeño Double']

df = df.groupby(['Item']).sum().sort_values(by='neturicious', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(9, 5))

df = df[neturicious]

sns.heatmap(df, ax=ax, annot=True)

ax.set_title('What to choose in Beef & Pork Category')

plt.xticks(rotation=45)
# selecting Beef & Pork category to show non-neutricious food



df = data[data['Category'] == 'Beef & Pork']

df = df[df['Item'] != 'Jalapeño Double']

df = df.groupby(['Item']).sum().sort_values(by='nonneturicious', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(9, 5))

df = df[nonneturicious]

sns.heatmap(df, ax=ax, annot=True)

ax.set_title('What not to choose in Beef & Pork Category')

plt.xticks(rotation=45)
# selecting Chicken & Fish category to show neutricious food



df = data[data['Category'] == 'Chicken & Fish']

df = df.groupby(['Item']).sum().sort_values(by='neturicious', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))

df = df[neturicious]

sns.heatmap(df, ax=ax, annot=True)

ax.set_title('What to choose in Chicken & Fish Category')

plt.xticks(rotation=45)
# selecting Chicken & Fish category to show non-neutricious food



df = data[data['Category'] == 'Chicken & Fish']

df = df.groupby(['Item']).sum().sort_values(by='nonneturicious', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))

df = df[nonneturicious]

sns.heatmap(df, ax=ax, annot=True)

ax.set_title('What not to choose in Chicken & Fish Category')

plt.xticks(rotation=45)
# selecting Salads category to show neutricious food



df = data[data['Category'] == 'Salads']

df = df.groupby(['Item']).sum().sort_values(by='neturicious', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))

df = df[neturicious]

sns.heatmap(df, ax=ax, annot=True)

ax.set_title('What to choose in Salads Category')

plt.xticks(rotation=45)
# selecting Salads category to show non-neutricious food



df = data[data['Category'] == 'Salads']

df = df.groupby(['Item']).sum().sort_values(by='nonneturicious', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))

df = df[nonneturicious]

sns.heatmap(df, ax=ax, annot=True)

ax.set_title('What not to choose in Salads Category')

plt.xticks(rotation=45)
# selecting Snacks & Sides category to show neutricious food



df = data[data['Category'] == 'Snacks & Sides']

df = df.groupby(['Item']).sum().sort_values(by='neturicious', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))

df = df[neturicious]

sns.heatmap(df, ax=ax, annot=True)

ax.set_title('What to choose in Snacks & Sides Category')

plt.xticks(rotation=45)
# selecting Snacks & Sides category to show non-neutricious food



df = data[data['Category'] == 'Snacks & Sides']

df = df.groupby(['Item']).sum().sort_values(by='nonneturicious', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))

df = df[nonneturicious]

sns.heatmap(df, ax=ax, annot=True)

ax.set_title('What not to choose in Snacks & Sides Category')

plt.xticks(rotation=45)
# selecting Desserts category to show neutricious food



df = data[data['Category'] == 'Desserts']

df = df.groupby(['Item']).sum().sort_values(by='neturicious', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(12, 5))

df = df[neturicious]

sns.heatmap(df, ax=ax, annot=True)

ax.set_title('What to choose in Desserts')

plt.xticks(rotation=45)
# selecting Desserts category to show non-neutricious food



df = data[data['Category'] == 'Desserts']

df = df.groupby(['Item']).sum().sort_values(by='nonneturicious', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(12, 5))

df = df[nonneturicious]

sns.heatmap(df, ax=ax, annot=True)

ax.set_title('What not to choose in Desserts')

plt.xticks(rotation=45)
# selecting Beverages category to show neutricious food



df = data[data['Category'] == 'Beverages']

df = df.groupby(['Item']).sum().sort_values(by='neturicious', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 5))

df = df[neturicious]

sns.heatmap(df, ax=ax, annot=True)

ax.set_title('What to choose in Beverages')

plt.xticks(rotation=45)
# selecting Beverages category to show non-neutricious food



df = data[data['Category'] == 'Beverages']

df = df.groupby(['Item']).sum().sort_values(by='nonneturicious', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 5))

df = df[nonneturicious]

sns.heatmap(df, ax=ax, annot=True)

ax.set_title('What not to choose in Beverages')

plt.xticks(rotation=45)
# selecting Coffee & Tea category to show neutricious food



df = data[data['Category'] == 'Coffee & Tea']

# removing this entry for unicode error if any

df = df[~df['Item'].str.contains('Frappé')]

df = df.groupby(['Item']).sum().sort_values(by='neturicious', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))

df = df[neturicious]

sns.heatmap(df, ax=ax, annot=True)

ax.set_title('What to choose in Coffee & Tea')

plt.xticks(rotation=45)
# selecting Coffee & Tea category to show non-neutricious food



df = data[data['Category'] == 'Coffee & Tea']

# removing this entry for unicode error if any

df = df[~df['Item'].str.contains('Frappé')]

df = df.groupby(['Item']).sum().sort_values(by='nonneturicious', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(11, 5))

df = df[nonneturicious]

sns.heatmap(df, ax=ax, annot=True)

ax.set_title('What not to choose in Coffee & Tea')

plt.xticks(rotation=45)
# selecting Smoothies & Shakes category to show neutricious food



df = data[data['Category'] == 'Smoothies & Shakes']

# removing because of encoding issue

df = df[~df.Item.str.contains('M&M\’s')]

df = df.groupby(['Item']).sum().sort_values(by='neturicious', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))

df = df[neturicious]

sns.heatmap(df, ax=ax, annot=True)

ax.set_title('What to choose in Smoothies & Shakes')

plt.xticks(rotation=45)
# selecting Smoothies & Shakes category to show non-neutricious food



df = data[data['Category'] == 'Smoothies & Shakes']

# removing because of encoding issue

df = df[~df.Item.str.contains('M&M\’s')]

df = df.groupby(['Item']).sum().sort_values(by='nonneturicious', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))

df = df[nonneturicious]

sns.heatmap(df, ax=ax, annot=True)

ax.set_title('What not to choose in Smoothies & Shakes')

plt.xticks(rotation=45)