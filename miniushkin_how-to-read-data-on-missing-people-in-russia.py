import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series

%matplotlib inline
import glob
data_files_to_read = glob.glob("../input/missing-people-in-russia/data*")
all_files = [(int(x.split("-")[4][0:4]), int(x.split("-")[4][4:6])//3, x) for x in data_files_to_read]

year_stats = [(year-1, 4, file) for year, quarter, file in all_files if quarter == 0]
all_stats = year_stats + [(year, quarter, file) for year, quarter, file in all_files if quarter != 0]

year_stats
# CSV files have different separators, so we need to keep an eye on it

data = DataFrame()
for (y, q, fn) in all_stats:
    df = pd.read_csv(fn, skiprows=1)
    df['year'] = y
    df['quarter'] = q
    data = pd.concat([data, df], axis=0)
    
data.head()
translation = pd.read_csv('../input/missing-people-in-russia/translate.csv', header = None, names = ['ru', 'en'], index_col = False)
translation.head()
translation['ru'] = translation['ru'].apply(lambda x: x.strip())
translation['en'] = translation['en'].apply(lambda x: x.strip())
data['Subject'] = data['Subject'].apply(lambda x: x.strip())

data = pd.merge(data, translation, left_on='Subject', right_on='ru')

data.rename(columns={'en':'Subject (en)'},inplace=True)
data.drop('ru', axis=1, inplace=True)

data.head()
data = pd.merge(data, translation, left_on='Name of the statistical factor', right_on='ru')
data.drop('ru', axis=1, inplace=True)

data.rename(columns={'en':'Name of the statistical factor (en)'},inplace=True)
data.head()
data.drop(['Subject', 'Point FPSR', 'Name of the statistical factor'], axis=1, inplace=True)
data = data[data['Subject (en)']=="Total for Russia"]
data.head()
lost = DataFrame(data[data['Name of the statistical factor (en)'] == 'Total wanted persons, including those missing'])
found = DataFrame(data[data['Name of the statistical factor (en)'] == 'Identified persons from among the wanted persons, including those missing.'])
lost.rename(columns={'Importance of the statistical factor':'lost'}, inplace=True)
found.rename(columns={'Importance of the statistical factor':'found'}, inplace=True)

res = pd.merge(lost[lost.quarter==4], found[found.quarter==4], on='year')[['year', 'lost', 'found']]
res = res.set_index('year')
res = res.sort_index()
res
plt.plot(res['lost'].apply(lambda x: int(x)),'.-')
plt.ylabel("Lost")
plt.xlabel("Years")
plt.xticks(res.index)
plt.ylim((0, 100000))
plt.title("People missed during a year in Russia")
plt.plot(res['found'].apply(lambda x: int(x)),'.-')
plt.ylabel("Found")
plt.xlabel("Years")
plt.xticks(res.index)
plt.ylim((0, 50000))
plt.title("Missed people found during a year in Russia")
plt.plot(res['lost'].apply(lambda x: int(x)) - res['found'].apply(lambda x: int(x)),'.-')
plt.ylabel("Not found")
plt.xlabel("Years")
plt.xticks(res.index)
plt.ylim((0, 50000))
plt.title("People missed and not found during a year in Russia")
lost['yq'] = lost.year*10 + lost.quarter
lost = lost.set_index('yq')
lost = lost.sort_index()

lost.lost.diff