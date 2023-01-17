import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = (10, 6)
plt.style.use('ggplot')
battles = pd.read_csv('../input/battles.csv')

'''
there is an error on the data
on the Battle of Castle Black, it was Mance Rayder the attacker, not Stannis Baratheon
'''
battles.set_value(27, 'attacker_king', 'Mance Rayder')
battles.set_value(27, 'defender_king', 'Stannis Baratheon')

print(battles.info())
battles.head(5)
battles.groupby(['attacker_king', 'defender_king']).count()['name'].plot(kind = 'barh')
battles.groupby(['attacker_king','attacker_outcome']).count()['name'].unstack().plot(kind = 'barh')
def commander_battles(attack, defend):
    namesdict = {}
    for i in range(len(attack.index)):
        for name in attack.index[i].split(', '):
            if name in namesdict:
                namesdict[name] = namesdict[name] + attack.ix[i]
            else:
                namesdict[name] = attack.ix[i]
    for i in range(len(defend.index)):
        for name in defend.index[i].split(', '):
            if name in namesdict:
                namesdict[name] = namesdict[name] + defend.ix[i]
            else:
                namesdict[name] = defend.ix[i]
    return namesdict

commander_exp = commander_battles(battles.groupby('attacker_commander').count()['name'], battles.groupby('defender_commander').count()['name'])
commander_exp = pd.DataFrame.from_dict(commander_exp, 'index')
commander_exp.rename(columns = {0:'num_battles'}, inplace = True)
commander_exp.sort_values('num_battles').tail(15).plot(kind = 'barh')
battles.groupby(['attacker_king', 'battle_type']).count()['name'].unstack().plot(kind = 'barh')
ax = battles[battles['attacker_outcome'] == 'win'].plot(kind = 'scatter', y = 'attacker_size', x = 'defender_size', label = 'win', s=100)
battles[battles['attacker_outcome'] == 'loss'].plot(kind = 'scatter', y = 'attacker_size', x = 'defender_size', label = 'loss', color = 'r', s=100, ax = ax)
battles.groupby(['region']).count()['name'].sort_values().plot(kind = 'barh')
