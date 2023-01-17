# importing libraries

import numpy as np

import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
# important pandas options

pd.set_option("display.max.columns", None)
fifa_df = pd.read_csv('../input/fifa-20-complete-player-dataset/players_20.csv')
fifa_df.head()
chelsea_df = fifa_df.loc[(fifa_df['club'] == "Chelsea") | (fifa_df['short_name'] == "H. Ziyech")]
# shape of data

chelsea_df.shape
chelsea_df.head()
# selecting the columns i want

chelsea_squad = chelsea_df[['short_name', 'age','player_positions', 'overall', 'potential', 'nationality' ,'value_eur', 'wage_eur', 'preferred_foot','height_cm', 'weight_kg', 'contract_valid_until']].copy()

# replacing values i don't have on H. Ziyech

chelsea_squad.loc[chelsea_squad['short_name'] == 'H. Ziyech', ['contract_valid_until', 'wage_eur', 'value_eur']] = np.NaN
# renaming some columns

chelsea_squad.rename(columns={"short_name": "name", "contract_valid_until": "contract_expiry"}, inplace=True)
chelsea_squad
chelsea_squad['overall'].describe()
# Age Distribution 

plt.figure(figsize=(18,10))

plt.title('Age Distribution in Club')

sns.distplot(a=chelsea_squad['age'], kde=False, bins=10)
chelsea_squad[(chelsea_squad['overall'] != chelsea_squad['potential']) & (chelsea_squad['age'] <= 25)].sort_values(by='potential', ascending=False)[['name', 'age', 'player_positions','overall', 'potential']]
chelsea_squad[chelsea_squad['overall'] == chelsea_squad['potential']][['name', 'age', 'overall', 'contract_expiry' ,'value_eur', 'wage_eur']].sort_values(by='age', ascending=False)
# Nationality Representation

chelsea_squad['nationality'].value_counts()
plt.figure(figsize=(18,13))

plt.title('Nationality Represention at the Club')

sns.countplot(x="nationality", data=chelsea_squad, order = chelsea_squad['nationality'].value_counts().index)
# Height Distribution 

plt.figure(figsize=(18,10))

plt.title('Height Distribution in Club')

sns.distplot(a=chelsea_squad['height_cm'], kde=False)
# mean height

chelsea_squad['height_cm'].mean()
# Weight Distribution 

plt.figure(figsize=(18,10))

plt.title('Weight Distribution in Club')

sns.distplot(a=chelsea_squad['weight_kg'], kde=False)
# mean weight

chelsea_squad['weight_kg'].mean()
fig, ax = plt.subplots(ncols=2, figsize=(18,10))

sns.regplot(x=chelsea_squad['age'], y=chelsea_squad['overall'], ax=ax[0])

sns.regplot(x=chelsea_squad['age'], y=chelsea_squad['potential'], ax=ax[1])

ax[0].set_title('Age vs Overall')

ax[1].set_title('Age vs Potential')
plt.figure(figsize=(18, 10))

plt.title('Heatmap of Numerical Values in the Club')

sns.heatmap(data=chelsea_squad[['age', 'overall', 'potential', 'height_cm', 'weight_kg', 'value_eur', 'wage_eur']].corr(), annot=True)
attackers = ['ST', 'RW', 'LW']

mid = ['CAM', 'CM', 'RM', 'LM', 'CDM']

defenders = ['CB', 'LB', 'RB', 'LWB', 'RWB']



# method to find main playing position

def player_position(positions):

    # main position is going to be in the first three letters in the string

    main = positions[:4]

    main = main.replace(',','') # removing commas

    main = main.strip() # removing spaces

    

    if main in attackers:

        return 'Attacker'

    elif main in mid:

        return 'Midfielder'

    elif main in defenders:

        return 'Defender'

    else:

        return 'Goalkeeper'
result = []



for idx, pos in chelsea_squad.iterrows():

    position = player_position(pos['player_positions'])

    result.append(position)



chelsea_squad['position'] = result
chelsea_squad
chelsea_squad.groupby('position').overall.describe()
plt.figure(figsize=(15,8))

plt.title("Box Plot: Position vs Overall Rating")

sns.boxplot(x='position', y='overall', data=chelsea_squad)
chelsea_squad[chelsea_squad['position'] == 'Attacker'].sort_values(by='overall', ascending=False)
chelsea_squad[chelsea_squad['position'] == 'Midfielder'].sort_values(by='overall', ascending=False)
chelsea_squad[chelsea_squad['position'] == 'Defender'].sort_values(by='overall', ascending=False)
chelsea_squad[chelsea_squad['position'] == 'Goalkeeper'].sort_values(by='overall', ascending=False)
loan_df = fifa_df[fifa_df['loaned_from'] == 'Chelsea'][['short_name', 'age', 'club', 'overall', 'potential', 'player_positions', 'contract_valid_until']]
loan_df