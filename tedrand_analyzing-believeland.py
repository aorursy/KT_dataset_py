import pandas as pd;import matplotlib.pyplot as plt

import matplotlib;import seaborn as sns

matplotlib.style.use('ggplot')

%matplotlib inline

df = pd.read_csv('../input/shot_logs.csv')
df.columns.values
kyrie = df[df['player_name'] == 'kyrie irving']

love = df[df['player_name'] == 'kevin love']

bron = df[df['player_name'] == 'lebron james']



# Include some warriors

steph = df[df['player_name'] == 'stephen curry']

klay = df[df['player_name'] == 'klay thompson']

dray = df[df['player_name'] == 'draymond green']
plt.hist([kyrie['PERIOD'],bron['PERIOD'],love['PERIOD'],steph['PERIOD']])

plt.legend(['Kyrie Irving', 'Lebron James', 'Kevin Love', 'Steph'])
def get_pct(df, feature='SHOT_RESULT', y='made'):

    return len(df[df[feature] == y])/len(df)
big3 = [bron, kyrie, love]

names = ['Lebron', 'Kyrie', 'Kevin Love']

pct = []

for dude in big3:

    pct.append(get_pct(dude))

df3 = pd.DataFrame()

df3['n'] = names

df3['p'] = pct



fig, ax = plt.subplots()

ax.bar(df3.index, df3['p'])
plt.hist([bron['SHOT_DIST'], kyrie['SHOT_DIST'], love['SHOT_DIST']], label=names)

plt.legend()
sns.violinplot(bron['SHOT_DIST'])
sns.violinplot(steph['SHOT_DIST'])