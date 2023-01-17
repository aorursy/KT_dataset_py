import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv("../input/data.csv")

# These columns are links and will not be used in this notebook

data = data.drop(axis = 1, columns=['Photo','Flag', 'Club Logo'])  

data.head()
# Clean up value and wage columns

def get_value(value):

    value_num = value.replace('€','')

    if 'M' in value_num:

        value_num = float(value_num.replace('M','')) * 1000000

    elif 'K' in value_num:

        value_num = float(value_num.replace('K','')) * 1000

    return float(value_num) # Ensure both columns are in float format



data['Value'] = data['Value'].apply(lambda x: get_value(x))

data['Wage'] = data['Wage'].apply(lambda x: get_value(x))



data.head()
# Create the top_10_nation pandas series

by_nation = data.Nationality.value_counts()

top_10_nation = by_nation[:10]

top_10_nation
# Method 1: .plot() in pandas

top_10_nation.plot(kind='bar'); # The ';' is to avoid showing a message before the chart
# We can also plot horizontally by using 'barh' in 'kind' argument

top_10_nation.plot(kind='barh');
# Method 2: plt.bar() in matplotlib - we input x and y arguments

plt.bar(top_10_nation.index, top_10_nation);
# Horizontally

plt.barh(top_10_nation.index, top_10_nation);
# Method 3: barplot() in Seaborn

sns.barplot(top_10_nation.index, top_10_nation);
# To plot horizontal bars, just flip the first two arguments and seaborn will sort out the orientation itself

sns.barplot(top_10_nation, top_10_nation.index);
top_10_nation_r = top_10_nation.sort_values(ascending=True)

top_10_nation_r.plot(kind='barh');
# Method 4: Countplot

sns.countplot(y = 'Nationality', data=data);
sns.countplot(y = 'Nationality', data=data, order = data.Nationality.value_counts().iloc[:10].index);
england = data.loc[data.Nationality == 'England'].sort_values('Value', ascending = False)

england.head()
# Top 30 players by value

england_30 = england.head(30).loc[:, ['Name','Value']]

sns.barplot(england_30.Value, england_30.Name);
plt.figure(figsize=(10,7)) # Specify figure size

sns.barplot(england_30.Value / 1000000 , england_30.Name) # in millions

plt.title('Top 30 English Players by Value', fontsize=16)

plt.xlabel('Value (EUR M)')

plt.yticks(fontsize=12) # Larger tick labels

plt.xticks(fontsize=12)

plt.show()
plt.figure(figsize=(10,7)) # Specify figure size

sns.barplot(england_30.Value / 1000000 , england_30.Name, color = 'red') # color argument specifies a single color

plt.title('Top 30 English Players by Value', fontsize=16)

plt.xlabel('Value (EUR M)')

plt.yticks(fontsize=12) # Larger tick labels

plt.xticks(fontsize=12)

plt.show()
plt.figure(figsize=(10,7)) # Specify figure size

sns.barplot(england_30.Value / 1000000 , england_30.Name, palette = 'spring') # palette argument specifies the color map

plt.title('Top 30 English Players by Value', fontsize=16)

plt.xlabel('Value (EUR M)')

plt.yticks(fontsize=12) # Larger tick labels

plt.xticks(fontsize=12)

plt.show()
data['Club'].fillna('None', inplace=True) # Clean up some null values to avoid errors in the next step

arsenal = data[data['Club'].str.contains('Arsenal')]

arsenal = arsenal.sort_values('Wage', ascending=False)

arsenal.head()
avg_arsenal = np.mean(arsenal['Wage'])



plt.figure(figsize=(10,8))

g = sns.barplot(arsenal.Wage / 1000 , arsenal.Name) # in thousands

# Adding labels of data value

for i, v in enumerate(arsenal.Wage / 1000):

    g.text(v+1, i, str(int(v))) # The three arguments are x-coordinate, y-coordinate, and the label

plt.title('Wage of Arsenal Players')

plt.xlabel('Wage (EUR K)')

plt.axvline(avg_arsenal/1000) # The vertical line

g.text(avg_arsenal/1000 + 5, 20, 'Mean wage: ' + str(int(avg_arsenal/1000)) + 'K') # Annotation of the line

plt.show()