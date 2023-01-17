import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
objData = pd.read_csv('../input/objValues.csv')

objData.head()
nums = ['num_' + str(x) for x in range(1, 17)]

objDataMelted = pd.melt(objData, id_vars=['ObjType'], value_vars=nums)

objDataMelted = objDataMelted[objDataMelted.value.notnull()]
objectives = set(obj[1:] for obj in objDataMelted.ObjType)



plt.figure(figsize=(12, 9))

for i, obj in enumerate(objectives):

    plt.subplot(2, 3, i + 1)

    plt.title(obj)

    sns.distplot(objDataMelted[objDataMelted.ObjType == 'b' + obj].value, kde=False, color='blue')

    sns.distplot(objDataMelted[objDataMelted.ObjType == 'r' + obj].value, kde=False, color='red')



plt.show()
deathData = pd.read_csv('../input/deathValues.csv')

matchData = pd.read_csv('../input/LeagueofLegends.csv')

merged = pd.merge(deathData, matchData, on='MatchHistory')





def jungler_involved_in_lane_kill(row):

    if pd.isnull(row['Killer']):  # when executed by neutral camps

        return False

    

    victim_role = killer_role = None

    assist_roles = set()

    

    # find roles of involved players

    for team in ['red', 'blue']:

        for role in ['Top', 'Jungle', 'Middle', 'ADC', 'Support']:

            assists = [row['Assist_%d' % i] for i in range(1, 5)

                       if not pd.isnull(row['Assist_%d' % i])]



            assert not pd.isnull(row['Victim'])

            if row[team + role] in row['Killer']:

                killer_role = role

            elif row[team + role] in row['Victim']:

                victim_role = role

            elif any(row[team + role] in a for a in assists):

                assist_roles.add(role)



    # try to determine the location of and people involved in the teamfight

    if (victim_role != 'Jungle' and killer_role != 'Jungle'

            and 'Jungle' not in assist_roles):

        return False   # jungler not involved

    elif (victim_role == 'Jungle' and killer_role == 'Jungle'

              and not assist_roles):

        return False   # 1v1 between junglers (not a lane kill)

    else:

        return True





merged['jungler_in_lane_kill'] = merged.apply(jungler_involved_in_lane_kill, axis=1)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.title('Kills involving the jungler')

sns.distplot(

    merged[np.logical_and(merged.TeamColor == 'Blue', merged.jungler_in_lane_kill)].Time,

    kde=False, color='blue'

)

sns.distplot(

    merged[np.logical_and(merged.TeamColor == 'Red', merged.jungler_in_lane_kill)].Time,

    kde=False, color='red'

)



plt.subplot(1, 2, 2)

plt.title('Towers destroyed')

sns.distplot(objDataMelted[objDataMelted.ObjType == 'bTowers'].value, kde=False, color='blue')

sns.distplot(objDataMelted[objDataMelted.ObjType == 'rTowers'].value, kde=False, color='red')

plt.show()