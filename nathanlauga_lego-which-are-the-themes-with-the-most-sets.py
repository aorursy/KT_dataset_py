# import packages that we need
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Loading sets we need
sets = pd.read_csv('../input/sets.csv')
themes = pd.read_csv('../input/themes.csv')
print(sets.head())
print(themes.head())
#Create pandas Series with id of theme as index and count of values as value
set_theme_count = sets["theme_id"].value_counts()
#Convert it to dataframe
set_theme_count = pd.DataFrame({'id':set_theme_count.index, 'count':set_theme_count.values})

print(set_theme_count.head())
# Join name of theme
set_theme_count = pd.merge(set_theme_count, themes, on='id')

print(set_theme_count.head())
# Get only themes with no parent
set_theme_count_no_parent = set_theme_count[pd.isnull(set_theme_count['parent_id'])]

print(set_theme_count_no_parent.head())
# Get the top 10 and plot it
set_theme_count_top_10 = set_theme_count_no_parent.sort_values(by=["count"], ascending=False)[:10]
top_10 = set_theme_count_top_10["count"]
top_10.index = set_theme_count_top_10["name"]

top_10.plot.bar()
plt.show()