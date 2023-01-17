# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

sns.set()

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
marvel_ds = pd.read_csv("/kaggle/input/fivethirtyeight-comic-characters-dataset/marvel-wikia-data.csv")

print("Columns: '%s'" % ", ".join(marvel_ds.columns))
marvel_ds.describe()
print("Total characters: %d" % len(marvel_ds))
# Check if we have any date discrepancies

def find_date_discrepancy(row):

    appearance = row['FIRST APPEARANCE']

    if appearance is np.NaN:

        return False

    if row['Year'] is np.NaN:

        return True

    

    month, year = appearance.split('-')

    year = int(year)

    if year < 30:

        year = 2000 + year

    else:

        year = 1900 + year

    return year != row['Year']



len(marvel_ds[marvel_ds.apply(find_date_discrepancy, axis='columns')])
marvel_ds.ALIGN.value_counts().plot(kind="bar")
plt.figure(figsize=(11, 11))

plt.subplot(211)

plt.title("All characters")

sns.distplot(marvel_ds.Year, kde=False)



plt.subplot(212)

plt.title("Characters by alignment")

sns.distplot(marvel_ds[marvel_ds.ALIGN == "Good Characters"].Year, kde=False, label="Good", color='green')

sns.distplot(marvel_ds[marvel_ds.ALIGN == "Bad Characters"].Year, kde=False, label="Bad", color='red')

sns.distplot(marvel_ds[marvel_ds.ALIGN == "Neutral Characters"].Year, kde=False, label="Neutral", color='blue')

plt.legend()
sns.distplot(marvel_ds[marvel_ds.SEX == "Male Characters"].Year, kde=False, label="Male")

sns.distplot(marvel_ds[marvel_ds.SEX == "Female Characters"].Year, kde=False, label="Female")

sns.distplot(marvel_ds[marvel_ds.SEX == "Agender Characters"].Year, kde=False, label="Agender")

sns.distplot(marvel_ds[marvel_ds.SEX == "Genderfluid Characters"].Year, kde=False, label="Genderfluid")

plt.legend()
marvel_ds.SEX.value_counts().plot(kind="bar")
marvel_ds.groupby(['ALIGN', 'SEX']).size()
plt.figure(figsize=(5, 5))



plt.title("Eye color")

marvel_ds.EYE.value_counts().plot(kind="bar")
male_marvel_ds = marvel_ds[marvel_ds.SEX == "Male Characters"]

min_year = male_marvel_ds.Year.idxmin()

male_marvel_ds.loc[min_year, :]
female_marvel_ds = marvel_ds[marvel_ds.SEX == "Female Characters"]

min_year = female_marvel_ds.Year.idxmin()

female_marvel_ds.loc[min_year, :]
idx = marvel_ds.APPEARANCES.idxmax()

marvel_ds.loc[idx]
marvel_ds.nlargest(20, ["APPEARANCES"])
# Most popular bad characters

marvel_ds[marvel_ds.ALIGN == "Bad Characters"].nlargest(20, ["APPEARANCES"])
marvel_ds.ID.value_counts().plot(kind="bar")
sns.distplot(marvel_ds.APPEARANCES, kde=False)
marvel_ds.ALIVE.value_counts().plot(kind="bar")
def select_avengers(row):

    avengers = ["Captain America", "Iron Man", "Hulk", "Natalia Romanova", "Hawkeye", "Thor", "Clinton Barton"]

    for name in avengers:

        if row["name"].startswith(name) and row.APPEARANCES > 100:

            return True

    return False



avengers_first_movie = marvel_ds[marvel_ds.apply(select_avengers, axis='columns')]

avengers_first_movie
# Marvel characters with more than 1000 appearances

marvel_ds[marvel_ds['APPEARANCES'] > 1000]