# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.patches as patches

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
frame = pd.read_csv("../input/cpj.csv")

frame.head()
frame['Date'] = frame['Date'].apply(lambda x: x.replace(')', ''))

frame['Date'] = frame['Date'].apply(lambda x: x.replace(',', ''))

frame['Year'] = frame.Date.apply(lambda x: x[len(x) - 4:])

frame.head()
labels = sorted(set(frame['Year']))

ind = np.arange(len(labels))

male_heights = [frame[frame["Year"] == year]['Sex'].value_counts()['Male'] for year in labels]

female_heights = [frame[frame["Year"] == year]['Sex'].value_counts()['Female'] if 'Female' in frame[frame["Year"] == year]['Sex'].value_counts().index else 0 for year in labels]

plt.figure(figsize=(10, 8))

plt.bar(ind, male_heights, width=.45, color=(.35,.92,.54))

plt.bar(ind+.45, female_heights, width=.45, color=(.87,.21,.16))

handles = [patches.Patch(label="Male", color=(.35,.92,.54)),patches.Patch(label="Female", color=(.87,.21,.16))]

plt.gca().set_xticks(ind+.45)

plt.gca().set_xticklabels(labels)

plt.xticks(rotation=90)

plt.legend(handles=handles)

plt.title("Deaths Per Year")

plt.tight_layout()

plt.show()
labels = set(frame['Job'])

ind = np.arange(len(labels))

male_heights = [frame[frame["Job"] == year]['Sex'].value_counts()['Male'] if 'Male' in frame[frame["Job"] == year]['Sex'].value_counts().index else 0 for year in labels]

female_heights = [frame[frame["Job"] == year]['Sex'].value_counts()['Female'] if 'Female' in frame[frame["Job"] == year]['Sex'].value_counts().index else 0 for year in labels]

plt.figure(figsize=(20, 16))

plt.bar(ind, male_heights, width=.45, color=(.35,.92,.54))

plt.bar(ind+.45, female_heights, width=.45, color=(.87,.21,.16))

handles = [patches.Patch(label="Male", color=(.35,.92,.54)),patches.Patch(label="Female", color=(.87,.21,.16))]

plt.gca().set_xticks(ind+.45)

plt.gca().set_xticklabels(labels)

plt.xticks(rotation=90)

plt.legend(handles=handles)

plt.title("Deaths Per Job")

plt.tight_layout()

plt.show()
labels = set(frame['Type_death'])

ind = np.arange(len(labels))

male_heights = [frame[frame["Type_death"] == year]['Sex'].value_counts()['Male'] if 'Male' in frame[frame["Type_death"] == year]['Sex'].value_counts().index else 0 for year in labels]

female_heights = [frame[frame["Type_death"] == year]['Sex'].value_counts()['Female'] if 'Female' in frame[frame["Type_death"] == year]['Sex'].value_counts().index else 0 for year in labels]

plt.figure(figsize=(10, 8))

plt.bar(ind, male_heights, width=.45, color=(.35,.92,.54))

plt.bar(ind+.45, female_heights, width=.45, color=(.87,.21,.16))

handles = [patches.Patch(label="Male", color=(.35,.92,.54)),patches.Patch(label="Female", color=(.87,.21,.16))]

plt.gca().set_xticks(ind+.45)

plt.gca().set_xticklabels(labels)

plt.xticks(rotation=90)

plt.legend(handles=handles)

plt.title("Causes of Death")

plt.tight_layout()

plt.show()
key = frame['Year'].value_counts()

x = [int(i) for i in sorted(frame['Year'].value_counts().index) if i != 'nown']

y = [key[str(i)] for i in x]
plt.title("Deaths Per Year")

plt.ylabel("Deaths")

plt.xlabel("Year")

plt.plot(x, y)

plt.show()