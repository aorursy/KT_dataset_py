import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
chardeath = pd.read_csv('../input/character-deaths.csv')
#Missing Columns shown in yellow

ax = plt.figure(figsize = (12,6))

sns.heatmap(chardeath.isnull(),yticklabels=False,cbar=False,cmap='viridis')
chardeath.drop('Book of Death',axis=1,inplace=True)

chardeath.drop('Death Chapter',axis=1,inplace=True)
Lannister_Got = chardeath.loc[chardeath['Allegiances'] == "Lannister", 'GoT'].sum()

Lannister_CoK = chardeath.loc[chardeath['Allegiances'] == "Lannister", 'CoK'].sum()

Lannister_SoS = chardeath.loc[chardeath['Allegiances'] == "Lannister", 'SoS'].sum()

Lannister_FfC = chardeath.loc[chardeath['Allegiances'] == "Lannister", 'FfC'].sum()

Lannister_DwD = chardeath.loc[chardeath['Allegiances'] == "Lannister", 'DwD'].sum()



HouTargaryen_Got = chardeath.loc[chardeath['Allegiances'] == "House Targaryen", 'GoT'].sum()

HouTargaryen_CoK = chardeath.loc[chardeath['Allegiances'] == "House Targaryen", 'CoK'].sum()

HouTargaryen_SoS = chardeath.loc[chardeath['Allegiances'] == "House Targaryen", 'SoS'].sum()

HouTargaryen_FfC = chardeath.loc[chardeath['Allegiances'] == "House Targaryen", 'FfC'].sum()

HouTargaryen_DwD = chardeath.loc[chardeath['Allegiances'] == "House Targaryen", 'DwD'].sum()



Targaryen_Got = chardeath.loc[chardeath['Allegiances'] == "Targaryen", 'GoT'].sum()

Targaryen_CoK = chardeath.loc[chardeath['Allegiances'] == "Targaryen", 'CoK'].sum()

Targaryen_SoS = chardeath.loc[chardeath['Allegiances'] == "Targaryen", 'SoS'].sum()

Targaryen_FfC = chardeath.loc[chardeath['Allegiances'] == "Targaryen", 'FfC'].sum()

Targaryen_DwD = chardeath.loc[chardeath['Allegiances'] == "Targaryen", 'DwD'].sum()



HouTargaryen_Got = HouTargaryen_Got + Targaryen_Got

HouTargaryen_CoK = HouTargaryen_CoK + Targaryen_CoK

HouTargaryen_SoS = HouTargaryen_SoS + Targaryen_SoS

HouTargaryen_FfC = HouTargaryen_FfC + Targaryen_FfC

HouTargaryen_DwD = HouTargaryen_DwD + Targaryen_DwD



Targaryen_Death = np.array([HouTargaryen_Got,HouTargaryen_CoK,HouTargaryen_SoS,HouTargaryen_FfC,HouTargaryen_DwD])

Targaryen_Death = pd.Series.from_array(Targaryen_Death)



HouStark_GoT = chardeath.loc[chardeath['Allegiances'] == "House Stark", 'GoT'].sum()

HouStark_CoK = chardeath.loc[chardeath['Allegiances'] == "House Stark", 'CoK'].sum()

HouStark_SoS = chardeath.loc[chardeath['Allegiances'] == "House Stark", 'SoS'].sum()

HouStark_FfC = chardeath.loc[chardeath['Allegiances'] == "House Stark", 'FfC'].sum()

HouStark_DwD = chardeath.loc[chardeath['Allegiances'] == "House Stark", 'DwD'].sum()



Stark_GoT = chardeath.loc[chardeath['Allegiances'] == "Stark", 'GoT'].sum()

Stark_CoK = chardeath.loc[chardeath['Allegiances'] == "Stark", 'CoK'].sum()

Stark_SoS = chardeath.loc[chardeath['Allegiances'] == "Stark", 'SoS'].sum()

Stark_FfC = chardeath.loc[chardeath['Allegiances'] == "Stark", 'FfC'].sum()

Stark_DwD = chardeath.loc[chardeath['Allegiances'] == "Stark", 'DwD'].sum()



HouStark_GoT = HouStark_GoT + Stark_GoT

HouStark_CoK = HouStark_CoK + Stark_CoK

HouStark_SoS = HouStark_SoS + Stark_SoS

HouStark_FfC = HouStark_FfC + Stark_FfC

HouStark_DwD = HouStark_DwD + Stark_DwD



Stark_Death = np.array([HouStark_GoT,HouStark_CoK,HouStark_SoS,HouStark_FfC,HouStark_DwD])

Stark_Death = pd.Series.from_array(Stark_Death)



NightWatch_GoT = chardeath.loc[chardeath['Allegiances'] == "Night's Watch", 'GoT'].sum()

NightWatch_CoK = chardeath.loc[chardeath['Allegiances'] == "Night's Watch", 'CoK'].sum()

NightWatch_SoS = chardeath.loc[chardeath['Allegiances'] == "Night's Watch", 'SoS'].sum()

NightWatch_FfC = chardeath.loc[chardeath['Allegiances'] == "Night's Watch", 'FfC'].sum()

NightWatch_DwD = chardeath.loc[chardeath['Allegiances'] == "Night's Watch", 'DwD'].sum()





NightWatch_Death = np.array([NightWatch_GoT,NightWatch_CoK,NightWatch_SoS,NightWatch_FfC,NightWatch_DwD])

NightWatch_Death = pd.Series.from_array(NightWatch_Death)





Baratheon_GoT = chardeath.loc[chardeath['Allegiances'] == "Baratheon", 'GoT'].sum()

Baratheon_CoK = chardeath.loc[chardeath['Allegiances'] == "Baratheon", 'CoK'].sum()

Baratheon_SoS = chardeath.loc[chardeath['Allegiances'] == "Baratheon", 'SoS'].sum()

Baratheon_FfC = chardeath.loc[chardeath['Allegiances'] == "Baratheon", 'FfC'].sum()

Baratheon_DwD = chardeath.loc[chardeath['Allegiances'] == "Baratheon", 'DwD'].sum()



HouBaratheon_GoT = chardeath.loc[chardeath['Allegiances'] == "House Baratheon", 'GoT'].sum()

HouBaratheon_CoK = chardeath.loc[chardeath['Allegiances'] == "House Baratheon", 'CoK'].sum()

HouBaratheon_SoS = chardeath.loc[chardeath['Allegiances'] == "House Baratheon", 'SoS'].sum()

HouBaratheon_FfC = chardeath.loc[chardeath['Allegiances'] == "House Baratheon", 'FfC'].sum()

HouBaratheon_DwD = chardeath.loc[chardeath['Allegiances'] == "House Baratheon", 'DwD'].sum()



Baratheon_GoT = Baratheon_GoT + HouBaratheon_GoT

Baratheon_CoK = Baratheon_CoK + HouBaratheon_CoK

Baratheon_SoS = Baratheon_SoS + HouBaratheon_SoS

Baratheon_FfC = Baratheon_FfC + HouBaratheon_FfC

Baratheon_DwD = Baratheon_DwD + HouBaratheon_DwD





Baratheon_Death = np.array([Baratheon_GoT,Baratheon_CoK,Baratheon_SoS,Baratheon_FfC,Baratheon_DwD])

Baratheon_Death = pd.Series.from_array(Baratheon_Death)

Got = pd.Series([Baratheon_GoT,HouStark_GoT,HouTargaryen_Got,NightWatch_GoT,Lannister_Got],

                index=["Baratheon", "Stark", "Targaryen", "Nightwatch", "Lannister"])

Got_death = max(Got)



SoS = pd.Series([Baratheon_SoS,HouStark_SoS,HouTargaryen_SoS,NightWatch_SoS,Lannister_SoS],

                index=["Baratheon", "Stark", "Targaryen", "Nightwatch", "Lannister"])

SoS_death = max(SoS)



FfC = pd.Series([Baratheon_FfC,HouStark_FfC,HouTargaryen_FfC,NightWatch_FfC,Lannister_FfC],

                index=["Baratheon", "Stark", "Targaryen", "Nightwatch", "Lannister"])

FfC_death = max(FfC)



CoK = pd.Series([Baratheon_CoK,HouStark_CoK,HouTargaryen_CoK,NightWatch_CoK,Lannister_CoK], 

                index=["Baratheon", "Stark", "Targaryen", "Nightwatch", "Lannister"])

CoK_death = max(CoK)



DwD = pd.Series([Baratheon_DwD,HouStark_DwD,HouTargaryen_DwD,NightWatch_DwD,Lannister_DwD],

                index=["Baratheon", "Stark", "Targaryen", "Nightwatch", "Lannister"])

DwD_death = max(DwD)
Death = np.array([Got_death,SoS_death,FfC_death,CoK_death,DwD_death])

Death = pd.Series.from_array(Death)





Houses = pd.Series(['Lannister','NightWatch','Baratheon/Stark','Stark','Nightwatch'])



BookList1 = np.array(['Game of Thrones', 'Clash of Kings', 'Storm of Swords', 'Feast for Crows', 'Dance with Dragons'])

Allegiance1 = [0,1,2,3,4]



Death = Death.sort_values()

font = {'family': 'serif',

        'color':  'black',

        'weight': 'normal',

        'size': 14,

        }



sns.set_style("whitegrid")

ax3 = plt.figure(figsize = (12,6))

ax2 = sns.barplot(Allegiance1,Death,palette = "Blues_d")

ax2.set_xlabel("Maximum no. of Death in each book and respective house",fontdict = font)

rects = ax2.patches

for rect, label in zip(rects, Houses):

    height = rect.get_height()

    ax2.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize = 18)

plt.ylabel("Max death in various Allegiances",fontdict = font)

plt.xticks(Allegiance1,BookList1, rotation = 45)