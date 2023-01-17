import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib_venn import venn3, venn3_circles
plt.style.use('seaborn-deep')
import seaborn as sns
sns.set_style("whitegrid")
matplotlib.rc('font', size=20)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('axes', labelsize=20)
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('legend', fontsize=20)
matplotlib.rc('figure', titlesize=20)
np.random.seed(42)
%matplotlib inline
heroes_info = pd.read_csv("../input/heroes_information.csv")
heroes_info.drop(heroes_info.columns[0], axis=1, inplace=True)
heroes_info = heroes_info.rename(columns={'name':'hero_names'})
heroes_powers = pd.read_csv("../input/super_hero_powers.csv")
heroes_powers.sample(5)
corr = heroes_powers.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
powers = heroes_powers.corr().columns.values
for col, row in ((heroes_powers.corr().abs() > 0.4) & (heroes_powers.corr().abs() < 1)).iteritems():
    if (len(powers[row.values])>0):
        print(col, powers[row.values]) 
powers_cat = {
    'Vision':['hero_names','Vision - X-Ray','Vision - Telescopic',
               'Vision - Microscopic', 'Vision - Night',
              'Vision - Heat'],
    'Reality Distortion':['hero_names','Reality Warping', 'Dimensional Awareness', 'Omnipotent',
                         'Omnipresent', 'Omniscient', 'Dimensional Awareness', 
                         'Time Manipulation','Dimensional Travel','Shapeshifting'],
    'Strength':['hero_names','Durability','Super Strength','Super Speed','Stamina','Reflexes',
               'Energy Armor','Force Fields'],
    'Thermal':['hero_names','Cold Resistance','Energy Absorption','Heat Resistance',
               'Fire Resistance']
}
#vision = heroes_powers[powers_cat['Vision']]
vision = heroes_powers[(heroes_powers[powers_cat['Vision']] == True).any(axis=1)][powers_cat['Vision']]
vision.sample(5)
reality_dis = heroes_powers[(heroes_powers[powers_cat['Reality Distortion']] == True).any(axis=1)][powers_cat['Reality Distortion']]
strength = heroes_powers[(heroes_powers[powers_cat['Strength']] == True).any(axis=1)][powers_cat['Strength']]
thermal = heroes_powers[(heroes_powers[powers_cat['Thermal']] == True).any(axis=1)][powers_cat['Thermal']]
print (vision.shape, reality_dis.shape, strength.shape, thermal.shape)
vision_heroes = set(list(vision.hero_names))
reality_dist_heroes = set(list(reality_dis.hero_names))
strength_heroes = set(list(strength.hero_names))
thermal_heroes = set(list(thermal.hero_names))
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(15,15))
venn3([vision_heroes, reality_dist_heroes, strength_heroes], 
      ('Vision', 'Reality Distortion', 'Strength'), ax = ax[0][0])
venn3([reality_dist_heroes, strength_heroes, thermal_heroes], 
      ('Reality Distortion', 'Strength', 'Thermal'), ax = ax[0][1])
venn3([strength_heroes, thermal_heroes, vision_heroes], 
      ('Strength', 'Thermal', 'Vision'), ax = ax[1][0])
venn3([vision_heroes, thermal_heroes, reality_dist_heroes], 
      ('Vision', 'Thermal', 'Reality Distortion'), ax = ax[1][1])
thermal_hero_prop = pd.merge(heroes_info, thermal, on='hero_names')
thermal_hero_prop.replace([-99, np.NaN],0, inplace=True)
realDis_hero_prop = pd.merge(heroes_info, reality_dis, on='hero_names')
realDis_hero_prop.replace(-99,0, inplace=True)
vision_hero_prop = pd.merge(heroes_info, vision, on='hero_names')
vision_hero_prop.replace(-99,0, inplace=True)
sns.jointplot(thermal_hero_prop.Weight, thermal_hero_prop.Height, 
              data=thermal_hero_prop, kind='kde', size=10)
#plt.title("Height & Weight of heroes with Thermal abilities")
f, ax = plt.subplots(ncols=2, nrows=2, figsize=(20,20))
explode = (0, 0.1, 0, 0)
x = ax[0][0].pie(list(thermal_hero_prop['Alignment'].value_counts()),
             labels=list(thermal_hero_prop['Alignment'].unique()),
             autopct='%1.1f%%', shadow=True, startangle=90)
y = ax[0][1].pie(list(thermal_hero_prop['Gender'].value_counts()),
             labels=list(thermal_hero_prop['Gender'].unique()),
             autopct='%1.1f%%', shadow=True, startangle=90)
sns.barplot(x = 'Publisher', y='index', 
            data=pd.DataFrame(thermal_hero_prop.Publisher.value_counts()).reset_index(), 
            orient='h',ax=ax[1][0])
sns.barplot(x = 'index', y='Eye color', 
            data=pd.DataFrame(thermal_hero_prop['Eye color'].value_counts()).reset_index(), ax=ax[1][1])
ax[1][0].set_ylabel('Publisher')
ax[1][0].set_xlabel('Count')
ax[1][1].set_ylabel('Count')
ax[1][1].set_xlabel('Eye Color')
plt.setp(ax[1][1].get_xticklabels(), rotation=45)

plt.suptitle("Characterstics Superheroes with Thermal Abilities")
sns.jointplot(realDis_hero_prop.Weight, realDis_hero_prop.Height, 
              data=realDis_hero_prop, kind='kde', size=10)
f, ax = plt.subplots(ncols=2, nrows=2, figsize=(20,20))
explode = (0, 0.1, 0, 0)
x = ax[0][0].pie(list(realDis_hero_prop['Alignment'].value_counts()),
             labels=list(realDis_hero_prop['Alignment'].unique()),
             autopct='%1.1f%%', shadow=True, startangle=90)
y = ax[0][1].pie(list(realDis_hero_prop['Gender'].value_counts()),
             labels=list(realDis_hero_prop['Gender'].unique()),
             autopct='%1.1f%%', shadow=True, startangle=90)
sns.barplot(x = 'Publisher', y='index', 
            data=pd.DataFrame(realDis_hero_prop.Publisher.value_counts()).reset_index(), 
            orient='h',ax=ax[1][0])
sns.barplot(x = 'index', y='Eye color', 
            data=pd.DataFrame(realDis_hero_prop['Eye color'].value_counts()).reset_index(), ax=ax[1][1])
ax[1][0].set_ylabel('Publisher')
ax[1][0].set_xlabel('Count')
ax[1][1].set_ylabel('Count')
ax[1][1].set_xlabel('Eye Color')
plt.setp(ax[1][1].get_xticklabels(), rotation=45)

plt.suptitle("Characterstics Superheroes with Reality Distortion Abilities")
sns.jointplot(vision_hero_prop.Weight, vision_hero_prop.Height, 
              data=realDis_hero_prop, kind='kde', size=10)
f, ax = plt.subplots(ncols=2, nrows=2, figsize=(20,20))
explode = (0, 0.1, 0, 0)
x = ax[0][0].pie(list(vision_hero_prop['Alignment'].value_counts()),
             labels=list(vision_hero_prop['Alignment'].unique()),
             autopct='%1.1f%%', shadow=True, startangle=90)
y = ax[0][1].pie(list(vision_hero_prop['Gender'].value_counts()),
             labels=list(vision_hero_prop['Gender'].unique()),
             autopct='%1.1f%%', shadow=True, startangle=90)
sns.barplot(x = 'Publisher', y='index', 
            data=pd.DataFrame(vision_hero_prop.Publisher.value_counts()).reset_index(), 
            orient='h',ax=ax[1][0])
sns.barplot(x = 'index', y='Eye color', 
            data=pd.DataFrame(vision_hero_prop['Eye color'].value_counts()).reset_index(), ax=ax[1][1])
ax[1][0].set_ylabel('Publisher')
ax[1][0].set_xlabel('Count')
ax[1][1].set_ylabel('Count')
ax[1][1].set_xlabel('Eye Color')
plt.setp(ax[1][1].get_xticklabels(), rotation=45)

plt.suptitle("Characterstics Superheroes with Vision Related Super Powers")