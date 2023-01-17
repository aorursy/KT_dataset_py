import pandas as pd
df = pd.DataFrame({

    'group1': [1,1,1,2,2,2,3,3,3,3,4,4,4],

    'group2': [1,1,2,1,2,2,2,3,3,3,1,2,4],

    'value': [1,2,3,4,5,6,7,8,9,10,11,12,13]

})

df
df.groupby(['group1', 'group2']).cumcount() + 1
# group1とgroup2のペアが変わった場所を探る

y = df['group1'].astype(str) + ' ' + df['group2'].astype(str)

df['count'] = y.groupby((y != y.shift()).cumsum()).cumcount() + 1

# group1 != group2 の行は1

df.loc[df['group1']!=df['group2'], 'count'] = 1

df