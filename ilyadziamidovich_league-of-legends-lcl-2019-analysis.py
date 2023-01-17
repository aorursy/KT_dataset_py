import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



data = pd.read_csv('/kaggle/input/league-of-legends-lcl-2019/lcl_match_history.csv')
data.head(10)
data.isna().sum()
data.duplicated().sum()
bool_series = data.duplicated(keep = False) 



data[bool_series] 
data['Team1'].value_counts()
data['Red'].value_counts()
data['Blue'].value_counts()
shortcut_dict = { 'GMB':'Gambit Esports', 'EPG':'Elements Pro Gaming', 'M19':'M19', 'VS':'Vaevictis eSports', 'RoX':'RoX', 'VEG':'Vega Squadron', 'DA':'Dragon Army', 'UOL':'Unicorns Of Love', 'TJ': 'Team Just'}
def replace_shortcuts(row):

    row['Red'] = shortcut_dict[row['Red']]

    row['Blue'] = shortcut_dict[row['Blue']]

    return row

data = data.apply(lambda row: replace_shortcuts(row), axis=1)
data
sides = ['Blue', 'Red']

def count_win_on_side(row):

    if (row['Winner'] == row['Blue']):

        return pd.Series([1, 0], sides)

    else:

        return pd.Series([0, 1], sides)



sides_data = data.apply(lambda row: count_win_on_side(row), axis=1).mean()

sides_data
sides_data.apply(lambda row: '{:.2%}'.format(row))
fig, ax = plt.subplots(figsize=(20, 7), subplot_kw=dict(aspect="equal"))



colors = ['#29b5ce', '#e03364']

plt.pie(sides_data, colors=colors, labels=sides)



ax.set_title('Distribution of the winning percentage by side', pad=20)

plt.axis('equal')

plt.show()