# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import seaborn as sns
import matplotlib.pyplot as plt

# changing the font size in sns
sns.set(font_scale=3)
df = pd.read_csv('../input/vgsales.csv', encoding='utf-8')
df.head()
# counts the number of items per Platform
sns.factorplot('Platform', data=df, kind='count', size=10, aspect=2)

# aesthetics
plt.title('Initial Plot')
plt.xlabel('Platform')
plt.ylabel('Number of Games Released')
plt.xticks(rotation=90)

# display
plt.show()
gen2 = ['Wii', 'X360', 'PS3']
df_g2 = df[df['Platform'].isin(gen2)]

sns.factorplot('Platform', data=df_g2, kind='count', size=10, aspect=2)

plt.title('Number of Unique Games Released for Gen. 2')
plt.xlabel('Platform')
plt.ylabel('Number of Games')
plt.xticks(rotation=90)

plt.show()
sns.factorplot('Year', hue='Platform', data=df_g2, kind='count', size=10, aspect=2)

plt.title('Number of Games Released for Year in Gen. 2')
plt.xlabel('Year')
plt.ylabel('Number of Games')
plt.xticks(rotation=90)

plt.show()
df_g2_sales = df_g2.groupby(['Platform', 'Year'], as_index=False)[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']].sum()
df_g2_sales.head()
palette ={"PS3":"Blue","X360":"Green", "Wii":"Red"}

sns.factorplot(x='Year', y='Global_Sales', hue='Platform', 
                   data=df_g2_sales, kind='bar', size=10, aspect=2, palette = palette)

plt.title('Global for Gen. 2')
plt.xlabel('Year')
plt.ylabel('Global Sales (in millions)')
plt.xticks(rotation=90)

plt.show()
sns.factorplot(x='Genre', y='Global_Sales', hue='Platform', 
                   data=df_g2, kind='bar', size=10, aspect=2, palette = palette, n_boot = False)

plt.title('Global Sales for Gen. 2')
plt.xlabel('Genre')
plt.ylabel('Global Sales (in millions)')
plt.xticks(rotation=90)

plt.show()
display(df_g2_sales.groupby(['Platform'])[['Global_Sales']].sum())
# top Xbox 360 ONLY games
df_g2_X360 = df[df['Platform'] == 'X360']['Name'][:50]
df_g2_PS3 = df[df['Platform'] == 'PS3']['Name']
df_g2_X360_only = df_g2[df_g2['Name'].isin(df_g2_X360) & (df_g2['Name'].isin(df_g2_PS3) == False)]

sns.factorplot(x='Name', y='Global_Sales', data=df_g2_X360_only, kind='bar', size=10, aspect=2)

plt.title('Top X360 Only Games')
plt.xlabel('Games')
plt.ylabel('Global Sales (in millions)')
plt.xticks(rotation=90)

plt.show()
# top PS3 ONLY games
df_g2_X360 = df[df['Platform'] == 'X360']['Name']
df_g2_PS3 = df[df['Platform'] == 'PS3']['Name'][:50]
df_g2_PS3_only = df_g2[df_g2['Name'].isin(df_g2_PS3) & (df_g2['Name'].isin(df_g2_X360) == False)]

sns.factorplot(x='Name', y='Global_Sales', data=df_g2_PS3_only, kind='bar', size=10, aspect=2)

plt.title('Top PS3 Only Games')
plt.xlabel('Games')
plt.ylabel('Global Sales (in millions)')
plt.xticks(rotation=90)

plt.show()
