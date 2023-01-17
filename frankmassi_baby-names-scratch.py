# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from bokeh.plotting import figure, show, output_notebook

from bokeh.palettes import brewer

colors = brewer['Spectral'][10]

output_notebook()

import matplotlib.pyplot as plt

%matplotlib inline





# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/NationalNames.csv')

df.head()
family_names = df[((df.Name.isin(['Sybil', 'Mirabel','Shayna'])) & (df.Gender == 'F')) | ((df.Name == 'Frank') & (df.Gender == 'M'))][['Name', 'Year', 'Count']]
years = range(1880, 2015)

f, ax = plt.subplots(figsize=(10, 6))

ax.set_xlim([1880, 2014])



colors = ['r','b','g','y']

for name, color in zip(list(set(family_names.Name)), colors):

    if name != 'Frank':

        plt.plot(family_names[family_names.Name == name].Year, 

                             family_names[df.Name == name].Count,

             label=name, color=color)

    ax.set_ylabel(name)

ax.set_xlabel('Year')

# ax.set_title('Average length of names')

legend = plt.legend(loc='best', frameon=True, borderpad=1, borderaxespad=1)
for color, name in x:

    print(color, name)