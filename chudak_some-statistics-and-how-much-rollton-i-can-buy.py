# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt

%matplotlib inline



# Any results you write to the current directory are saved as output.
cards = pd.read_csv("../input/cards_flat.csv", encoding='iso-8859-1')
mechanic_types = ('ADJACENT_BUFF',

 'AI_MUST_PLAY',

 'AURA',

 'AUTOATTACK',

 'BATTLECRY',

 'CANT_ATTACK',

 'CANT_BE_TARGETED_BY_HERO_POWERS',

 'CANT_BE_TARGETED_BY_SPELLS',

 'CHARGE',

 'CHOOSE_ONE',

 'COMBO',

 'DEATHRATTLE',

 'DIVINE_SHIELD',

 'ENRAGED',

 'EVIL_GLOW',

 'FORGETFUL',

 'FREEZE',

 'INSPIRE',

 'ImmuneToSpellpower',

 'InvisibleDeathrattle',

 'MORPH',

 'POISONOUS',

 'RITUAL',

 'SECRET',

 'SILENCE',

 'STEALTH',

 'TAG_ONE_TURN_EFFECT',

 'TAUNT',

 'TOPDECK',

 'WINDFURY')
def f(card): # make mechanics from string to list

    output=()

    if type(card['mechanics']) == str:

        output = tuple(subs.strip()[1:-1] for subs in card['mechanics'][1:-1].split(','))

        

        return (output)

    return (output)

cards['mechanics'] = cards.apply(f,axis=1)
cc = cards[(cards['collectible']==True) & (cards['set']!='HERO_SKINS') & (cards['type'] != 'HERO')]
plot_data = dict()

a = cc.groupby([ 'player_class','type']).size()

for pl_class in set(cc['player_class']):

    weapons = a[pl_class]['WEAPON'] if 'WEAPON' in a[pl_class] else 0

    spell = a[pl_class]['SPELL'] if 'SPELL' in a[pl_class] else 0

    minion = a[pl_class]['MINION'] if 'MINION' in a[pl_class] else 0

    if pl_class != 'NEUTRAL':

        plot_data[pl_class] = (minion, spell, weapons)



labels = ['PALADIN', 'HUNTER', 'SHAMAN', 'WARRIOR', 'ROGUE', 'PRIEST', 'MAGE', 'DRUID', 'WARLOCK']

minions_sizes = [plot_data[x][0] for x in labels]

colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'moccasin', 'darkcyan', 'lightslategrey', 'mediumorchid', 'royalblue']



plt.figure(0)

plt.pie(minions_sizes, labels=labels,

        autopct='%1.1f%%', shadow=False, startangle=90, colors=colors )

plt.title('MINIONS', fontsize=14, fontweight='bold', y = 1.1)

plt.axis('equal')



plt.figure(1)

plt.title('SPELLS', fontsize=14, fontweight='bold', y = 1.1)

plt.pie([plot_data[x][1] for x in labels], labels=labels,

        autopct='%1.1f%%', shadow=False, startangle=90, colors=colors)

plt.axis('equal');



plt.figure(2)

plt.title('WEAPONS', fontsize=14, fontweight='bold', y = 1.1)

weapon_labels = ['PALADIN', 'HUNTER', 'SHAMAN', 'WARRIOR', 'ROGUE']

plt.pie([plot_data[x][2] for x in weapon_labels], labels=weapon_labels,

        autopct='%1.1f%%', shadow=False, startangle=90, colors=colors)

plt.axis('equal');

cc = cc.fillna(0)

cc['stats_sum'] = cc['attack'] + cc['health']
cc['cost_without_effect'] = cc['cost']
standart = dict(cc[(cc['text']==0) & (cc['type']=='MINION')].groupby(['cost']).max()['stats_sum'])

standart[9] = 18 # there is no minions for 9 mana without effects, so lets decide, that its stats_sum should be 18

for i in cc.index:

    for cost in range(0, 10):

        try:

            if cc.loc[i, 'stats_sum'] <= standart[cost]:

                cc.loc[i, 'cost_without_effect'] = cost

                break

        except KeyError:

            pass

    else:

        cc.loc[i, 'cost_without_effect'] = 10

        cc.loc[i, 'positive'] = 1



    try:

        if cc.loc[i,'stats_sum'] <= standart[cc.loc[i,'cost']]:

            cc.loc[i,'positive'] = 1

        elif cc.loc[i,'cost'] >= 7:

            cc.loc[i,'positive'] = 1

        else:

            cc.loc[i,'positive'] = 0

    except KeyError:

        cc.loc[i, 'positive'] = -1

        pass
neg_cc = cc[(cc['type']=='MINION') & (cc['positive']==0)]
plt.scatter(neg_cc['cost'], neg_cc['cost_without_effect'], alpha=0.5)

plt.axis('equal')

plt.axis([0, 10, 0, 10])

plt.grid(True)

plt.xlabel('cost')

plt.ylabel('cost without effects')

plt.show()

positive_cc = cc[(cc['type']=='MINION') & (cc['positive']==1)]
f, axarr = plt.subplots(2, 2)

plt.tight_layout(pad=0.4, w_pad=0.6, h_pad=1.5)

axarr[0, 0].scatter(positive_cc[positive_cc['rarity']=='COMMON']['cost'], positive_cc[positive_cc['rarity']=='COMMON']['cost_without_effect'], alpha=0.5)

axarr[0, 0].set_title('Common Cards')

axarr[0, 1].scatter(positive_cc[positive_cc['rarity']=='RARE']['cost'], positive_cc[positive_cc['rarity']=='RARE']['cost_without_effect'], alpha=0.5)

axarr[0, 1].set_title('Rare Cards')

axarr[1, 0].scatter(positive_cc[positive_cc['rarity']=='EPIC']['cost'], positive_cc[positive_cc['rarity']=='EPIC']['cost_without_effect'], alpha=0.5)

axarr[1, 0].set_title('Epic Cards')

axarr[1, 1].scatter(positive_cc[positive_cc['rarity']=='LEGENDARY']['cost'], positive_cc[positive_cc['rarity']=='LEGENDARY']['cost_without_effect'], alpha=0.5)

axarr[1, 1].set_title('Legendary Cards');
cc_sizes_byrarity = cc.groupby(['rarity']).size()

in_packs = (cc_sizes_byrarity['COMMON']*40+cc_sizes_byrarity['RARE']*100+cc_sizes_byrarity['EPIC']*400+cc_sizes_byrarity['LEGENDARY']*1600)/40

in_packs_golden = (cc_sizes_byrarity['COMMON']*100+cc_sizes_byrarity['RARE']*400+cc_sizes_byrarity['EPIC']*1600+cc_sizes_byrarity['LEGENDARY']*3200)/40

in_rubles = in_packs/60*3500

in_rubles_golden = in_packs_golden/60*3500

in_rollton = in_rubles/15*1.5/100 #high in meters in rollton

in_rollton_golden = in_rubles_golden/15*1.5/100 #high in meters in rollton
in_rollton_golden