import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv("../input/database.csv")

df = df[df.Weapon != 'Unknown']

topweapons = df.Weapon.value_counts().index.tolist()

threeweapons = topweapons[:3]

print(threeweapons)
df_group = (df.groupby(['State', 'Weapon']).size() / df.groupby('State').size() * 100).reset_index()

df_group.columns.values[2] = 'Percentage'



result = df_group.sort_values(['State', 'Percentage'], ascending=[1, 0]).groupby('State').head(3)

print(result)
####################################################################

state = 'West Virginia'

####################################################################

df_state = result[result['State'] == state]

print(df_state)
colors = []

for i in range(3):

    if df_state.ix[:, 'Weapon'].tolist()[i] == threeweapons[i]:

        colors.append("#00BEC5")

    else:

        colors.append("#FA8072")



ax = df_state.plot(x="Weapon", y="Percentage", color=colors, width=0.8, kind="bar", legend=False, fontsize=25, rot=360)

ax.set_xlabel("")

plt.title("Top 3 Weapons used in {} ".format(state), fontsize=25)

ax.text(1.85, 24, 'Normal weapons', style='italic',

        bbox={'facecolor':"#00BEC5", 'alpha':0.8})

ax.text(1.85, 20, 'Unusual weapons', style='italic',

        bbox={'facecolor':"#FA8072", 'alpha':0.8})

plt.show()