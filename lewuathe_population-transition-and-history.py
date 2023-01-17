# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/data.csv')

print(df.columns)

df.head()
nonnull_counts = df.isnull().sum(axis=1) / len(df.columns) < 0.05



nonnull_index = [i for (i, v) in enumerate(nonnull_counts) if v]



effective_records = df.iloc[nonnull_index]

print(effective_records.shape)

effective_records.head()
years = [str(y) for y in range(1974, 2016)]



cols = 2

rows = int(170 / cols) # 171 is the number of effective_records



from matplotlib.ticker import ScalarFormatter

xmajor_formatter = ScalarFormatter()

xmajor_formatter.set_powerlimits((-3, 4))

fig, ax = plt.subplots(rows, cols, figsize=(9, 300))

for r in range(rows):

    for c in range(cols):

        ax[r, c].plot(effective_records.iloc[r * cols + c][years])

        indicator_name = effective_records.iloc[r * cols + c]['Indicator Name']

        country_name = effective_records.iloc[r * cols + c, 0]

        ax[r, c].set_title(indicator_name + ' in ' + country_name)

        ax[r, c].yaxis.set_major_formatter( xmajor_formatter )
for i in range(171):

    plt.scatter(years, effective_records.iloc[i][years].values)
population_decreased_countries = effective_records[effective_records['2015'] - effective_records['1975'] < 0.0]



population_decreased_countries['Decreased Ratio'] = (population_decreased_countries['2015'] - population_decreased_countries['1975']) / population_decreased_countries['1975']



population_decreased_countries.sort('Decreased Ratio')
rows = 5

cols = 2

fig, ax = plt.subplots(rows, cols, figsize=(9, 15))



for r in range(rows):

    for c in range(cols):

        country_name = population_decreased_countries.iloc[r * cols + c, 0]

        ax[r, c].plot(population_decreased_countries.iloc[r * cols + c][years])

        ax[r, c].set_title("{}".format(country_name))

        ax[r, c].yaxis.set_major_formatter( xmajor_formatter )
rwanda = effective_records[effective_records.iloc[:, 0] == 'Rwanda']



plt.plot(years, rwanda.iloc[0][years], label='Population')

plt.axvline(x=1994, linewidth=3, label='Genocide')

plt.axvline(x=1991, linewidth=10, label='Civil War', color='red')

plt.title('Rwanda')

plt.legend()
kosovo = effective_records[effective_records.iloc[:, 0] == 'Kosovo']



plt.plot(years, kosovo.iloc[0][years], label='Population')

plt.axvline(x=1996, linewidth=3, label='Begin of Civil War')

plt.axvline(x=1999, linewidth=3, label='End of Civil War')

plt.title('Kosovo')

plt.legend()
kazakhstan = effective_records[effective_records.iloc[:, 0] == 'Kazakhstan']



plt.plot(years, kazakhstan.iloc[0][years], label='Population')

plt.axvline(x=1989, linewidth=3, label='Dissolution of USSR')

plt.title('Kazakhstan')

plt.legend()
czech = effective_records[effective_records.iloc[:, 0] == 'Czech Republic']



plt.plot(years, czech.iloc[0][years], label='Population')

plt.axvline(x=1989, linewidth=3, label='Velvet Revolution')

plt.axvline(x=1993, linewidth=3, label='Separate from Slovakia', color='red')

plt.axvline(x=2004, linewidth=3, label='Join EU', color='green')

plt.title('Czech Republic')

plt.legend(loc=2)