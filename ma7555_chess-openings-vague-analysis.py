# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/games.csv')

data.head()
# Remove variants

def main_openinig_stripper(opening):

    if ':' in opening:

        opening = opening.split(':')[0]

    while '|' in opening:

        opening = opening.split('|')[0]

    if '#' in opening:

        opening = opening.split('#')[0]

    if 'Accepted' in opening:

        opening = opening.replace('Accepted', '')

    if 'Declined' in opening:

        opening = opening.replace('Declined', '')

    if 'Refused' in opening:

        opening = opening.replace('Refused', '')

    return opening.strip()

data['main_opening'] = data.opening_name.apply(main_openinig_stripper)
# Which is the most played opening?

top_x = 15



plt.figure(figsize=(15,25))

plt.subplot(211)

chart = data.groupby('main_opening').size().nlargest(top_x).plot('bar')

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)



sizes = np.append(data.main_opening.value_counts().iloc[:top_x].values, data.main_opening.value_counts().iloc[top_x+1:].values.sum())

labels = np.append(data.main_opening.value_counts().iloc[:top_x].index, 'Other')

plt.subplot(212)

plt.pie(sizes, labels=labels, autopct='%.1f%%',

        shadow=True, pctdistance=0.85, labeldistance=1.05, startangle=90, explode = [0 if i > 0 else 0.2 for i in range(len(sizes))])

plt.axis('equal')

plt.show()
# Drop rare openings that we do not have enough data for

rate = int(len(data) * 0.005) # if played less than 0.1%

data = data.groupby('main_opening').filter(lambda x: len(x) > rate)

# plt.figure(figsize=(15,10))

# chart = data.groupby('main_opening').size().nsmallest(top_x).plot('bar')

# chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

# plt.show()
# Who is the player who played the most games?



top_x = 15

plt.figure(figsize=(15,8))

chart = sns.countplot(data.black_id.append(data.white_id), order=data.black_id.append(data.white_id).value_counts().iloc[:top_x].index)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

plt.show()
surprise = data[((data.white_rating > data.black_rating) & ((data.winner=='black') | (data.winner == 'draw'))) |

                ((data.white_rating < data.black_rating) & ((data.winner=='white') | (data.winner == 'draw')))]

openings_grouped_counts = surprise.groupby(['main_opening', 'winner']).id.count()

openings_grouped_percentage = openings_grouped_counts.groupby(level=[0]).apply(lambda g: g / g.sum())

openings_grouped = pd.concat([openings_grouped_counts, openings_grouped_percentage], axis=1, keys=['counts', 'percentage'])
openings_grouped.head(6)
#let's see strong trends

interesting_openings = openings_grouped[openings_grouped.percentage > 0.5]



plt.figure(figsize=(15,8))

chart = interesting_openings.percentage.nlargest(len(interesting_openings)).plot('bar')

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

plt.show()