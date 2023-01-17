!pip install -U seaborn
import os.path

import numpy as np
import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
excel_file = "globalterrorismdb_0718dist.xlsx"

if os.path.isfile(excel_file):
    print("Reading local", excel_file)
    df = pd.read_excel(excel_file)
else:
    print("Downloading and reading,", excel_file)
    df = pd.read_excel('http://apps.start.umd.edu/gtd/downloads/dataset/' + excel_file)
df.head()
df.columns.tolist()
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
DROP_THRESHOLD = .70

columns_to_drop = []
for column in df.columns.tolist():
    null_ratio = df[column].isnull().sum() / len(df[column])
    if null_ratio > DROP_THRESHOLD:
        columns_to_drop.append(column)
        print (column, "with null ratio", null_ratio , "will be dropped")

df.drop(columns_to_drop, axis=1, inplace=True)
print("All attacks", len(df))

# Also drop rows where gname is unkown
df = df[df['gname'] != 'Unknown']

print("Attacks where the attack group was known", len(df))
df.head()
df.columns.tolist()
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
df.fillna(0, inplace=True)

sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.relplot(x="iyear", y="nkill", 
            col="region_txt", # Categorical variables that will determine the faceting of the grid.
            hue="success",  # Grouping variable that will produce elements with different colors.
            style="success", # Grouping variable that will produce elements with different styles.
            data=df)
sns.relplot(x="iyear", y="nkill", 
           col="weaptype1_txt", # Categorical variables that will determine the faceting of the grid.
           hue="success",  # Grouping variable that will produce elements with different colors.
           style="success", # Grouping variable that will produce elements with different styles.
           data=df)
df.groupby("gname").size().sort_values(ascending=False).head()
df.groupby("gname")["nkill"].sum().sort_values(ascending=False).head()
df.groupby("targtype1_txt").size().sort_values(ascending=False).head()
df.groupby("natlty1_txt").size().sort_values(ascending=False).head()
df.groupby(['country_txt', 'natlty1_txt']).size()
df.loc[df['country_txt'] == 'Iraq', ['country_txt', 'natlty1_txt']].groupby(['country_txt', 'natlty1_txt']).size()
df.loc[df['country_txt'] == 'United States', ['country_txt', 'natlty1_txt']].groupby(['country_txt', 'natlty1_txt']).size()
df.loc[df['country_txt'] == 'Israel', ['country_txt', 'natlty1_txt']].groupby(['country_txt', 'natlty1_txt']).size()
