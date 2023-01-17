import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



fifa = pd.read_csv('../input/FullData.csv')

nation = fifa[['Name','Nationality', 'Rating']]

nation.head()
pd.options.mode.chained_assignment = None

best = pd.DataFrame()

i=0

while i<11:

    nation['max'] = nation.groupby('Nationality')['Rating'].transform('max')

    delete = nation[nation['max']==nation['Rating']].drop_duplicates('Nationality')

    best = best.append(delete)

    merged = pd.merge(nation,delete, how='outer', indicator=True)

    nation = merged[merged['_merge'] == 'left_only'].drop(['_merge','max'], 1)

    i+=1

ranking = best.groupby('Nationality')['Rating'].mean().sort_values(ascending=False)

ranking = pd.DataFrame(ranking).reset_index()[:20]

ranking.head()
plt.bar(np.arange(len(ranking.Nationality)), ranking.Rating.values, tick_label=ranking.Nationality, color='g')

plt.ylim(60,90)

plt.xticks(rotation = 70)

plt.show()