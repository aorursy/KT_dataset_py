import pandas as pd
import numpy as np
data = {
    'birds': [
        'Cranes', 'Cranes', 'plovers', 'spoonbills', 'spoonbills', 'Cranes',
        'plovers', 'Cranes', 'spoonbills', 'spoonbills'
    ],
    'age': [3.5, 4, 1.5, np.nan, 6, 3, 5.5, np.nan, 8, 4],
    'visits': [2, 4, 3, 4, 3, 4, 2, 2, 3, 2],
    'priority':
    ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']
}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame.from_dict(data)
labels = pd.DataFrame.from_dict(labels)
df['labels'] = labels
df = df.set_index('labels')

df
df.describe()
df[1:3]
df[['birds', 'age']]
df[['birds', 'age', 'visits']].iloc[[2, 3, 7]]
df[df['visits']<4]
temp= df[df['age'].isnull()]
temp[['birds', 'visits']]
df[(df['birds'] == 'Cranes') & (df['age'] < 4)]
df[(df['age'] <= 4) & (df['age'] >= 2)]
# df[df['birds'] == 'Cranes']
df[(df['birds'] == 'Cranes') & (df['visits'].notnull())].sum()
birds = df.groupby('birds')

for bird, bird_df in birds:
    print(bird, bird_df['age'].mean())
df.loc['k'] = ['Cranes', 3.5, 3, 'yes']
df
df = df.drop('k')
df
df['birds'].value_counts()
df.sort_values('age')
df.sort_values('visits')
df['priority'] = df['priority'].replace({'yes': 1, 'no': 0})
df
df['birds'] = df['birds'].replace({'Cranes': 'trumpeters'})
df
