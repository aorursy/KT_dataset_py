# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# To perform Hierarchical clustering

from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree

from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans
# import all libraries and dependencies for machine learning

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.decomposition import IncrementalPCA

from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

from math import isnan

# Data display coustomization

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
us=pd.read_csv("/kaggle/input/us-police-shootings/shootings.csv")

us.head()
us_d=us.copy()

us_d.drop_duplicates(subset=None, inplace=True)
us_d.shape
us.shape
del us_d
us.info()
us.describe()
us.shape
us['year'] = pd.DatetimeIndex(us['date']).year

us['month'] = pd.DatetimeIndex(us['date']).month

us['day'] = pd.DatetimeIndex(us['date']).day

us['week']= pd.DatetimeIndex(us['date']).weekofyear

us['quarter']= pd.DatetimeIndex(us['date']).quarter
us['Weapon']=us['armed']+'-' + us['arms_category']
us.head()
(us.isnull().sum() * 100 / len(us)).value_counts(ascending=False)
us.isnull().sum().value_counts(ascending=False)
(us.isnull().sum(axis=1) * 100 / len(us)).value_counts(ascending=False)
us.isnull().sum(axis=1).value_counts(ascending=False)
us.head()
us.nunique()
plt.figure(figsize=(10,5))

ax=us.name.value_counts(ascending=False)[:5].plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=45)

ax.set(ylabel = 'Frequency', xlabel= 'Name')

plt.show()
us.name.value_counts()[:5]
plt.figure(figsize=(10,5))

ax=us.date.value_counts()[:3].plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=45)

ax.set(ylabel = 'Frequency', xlabel= 'Date')

plt.show()
us.date.value_counts()[:3]
plt.figure(figsize=(10,5))

ax=us.manner_of_death.value_counts().plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=45)

ax.set(ylabel = 'Frequency', xlabel= 'Manner of Death')

plt.show()
round(us.manner_of_death.value_counts()/len(us) * 100 , 2)
plt.figure(figsize=(30,5))

ax=us.armed.value_counts().plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=90)

ax.set(ylabel = 'Frequency', xlabel= 'Armed')

plt.show()
round(us.armed.value_counts()/len(us) * 100 , 2)
us['armed']=us['armed'].replace(['shovel', 'hammer', 'hatchet', 'sword', 'machete', 'box cutter',

                                 'metal object', 'screwdriver', 'lawn mower blade', 'flagpole',

                                 'guns and explosives', 'cordless drill', 'metal pole', 'Taser',

                                 'metal pipe', 'metal hand tool', 'blunt object', 'metal stick',

                                 'sharp object', 'meat cleaver', 'carjack', 'chain',

                                 "contractor's level", 'stapler', 'crossbow', 'bean-bag gun',

                                 'baseball bat and fireplace poker', 'straight edge razor',

                                 'gun and knife', 'ax', 'brick', 'baseball bat', 'hand torch',

                                 'chain saw', 'garden tool', 'scissors', 'pole', 'pick-axe',

                                 'flashlight', 'nail gun', 'spear', 'chair', 'pitchfork',

                                 'hatchet and gun', 'rock', 'piece of wood', 'bayonet', 'pipe',

                                 'glass shard', 'motorcycle', 'pepper spray', 'metal rake', 'baton',

                                 'crowbar', 'oar', 'machete and gun', 'air conditioner',

                                 'pole and knife', 'beer bottle', 'baseball bat and bottle',

                                 'fireworks', 'pen', 'chainsaw', 'gun and sword', 'gun and car',

                                 'pellet gun', 'BB gun', 'incendiary device', 'samurai sword',

                                 'bow and arrow', 'gun and vehicle', 'vehicle and gun', 'wrench',

                                 'walking stick', 'barstool', 'grenade', 'BB gun and vehicle',

                                 'wasp spray', 'air pistol', 'baseball bat and knife',

                                 'vehicle and machete', 'ice pick', 'car, knife and mace'],'Other')
round(us.armed.value_counts()/len(us) * 100 , 2)
plt.figure(figsize=(10,5))

ax=us.armed.value_counts()[:5].plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=90)

ax.set(ylabel = 'Frequency', xlabel= 'Armed')

plt.show()
us.age.describe()
plt.figure(figsize=(30,5))

ax=us.age.value_counts().plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=90)

ax.set(ylabel = 'Frequency', xlabel= 'Age')

plt.show()
us.age.describe()
bins = [0,10,18,30,40,50,60,70,80,90,100]

labels =['babies','teen','20s','30s','40s','50s','60s','70s','80s','90s']

us['age'] = pd.cut(us['age'], bins,labels=labels)

us.age.value_counts(ascending=False)



plt.figure(figsize=(30,5))

ax=us.age.value_counts().plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=90)

ax.set(ylabel = 'Frequency', xlabel= 'Age')

plt.show()
us['age']=us['age'].replace(['60s','70s','80s','90s'],'Senior Citizen')

us['age']=us['age'].replace(['babies','teen'],'Children')

us.age.value_counts()
round(us.age.value_counts()/len(us) * 100 , 2)
plt.figure(figsize=(10,5))

ax=us.age.value_counts().plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=45)

ax.set(ylabel = 'Frequency', xlabel= 'Age Group')

plt.show()
plt.figure(figsize=(10,5))

ax=us.month.value_counts().plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=45)

ax.set(ylabel = 'Frequency', xlabel= 'Months')

plt.show()
dict ={1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sept',10:'Oct',11:'Nov',12:'Dec'}

us['month']= us['month'].map(dict) 
plt.figure(figsize=(10,5))

ax=us.month.value_counts().plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=45)

ax.set(ylabel = 'Frequency', xlabel= 'Months')

plt.show()
round(us.month.value_counts()/len(us) * 100 , 2)
plt.figure(figsize=(25,5))

ax=us.week.value_counts().plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=45)

ax.set(ylabel = 'Frequency', xlabel= 'Week')

plt.show()
round(us.week.value_counts()/len(us) * 100 , 2)
plt.figure(figsize=(20,5))

ax=us.day.value_counts().plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=45)

ax.set(ylabel = 'Frequency', xlabel= 'Days')

plt.show()
round(us.day.value_counts()/len(us) * 100 , 2)
plt.figure(figsize=(10,5))

ax=us.quarter.value_counts().plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=45)

ax.set(ylabel = 'Frequency', xlabel= 'Quarters')

plt.show()
round(us.quarter.value_counts()/len(us) * 100 , 2)
plt.figure(figsize=(10,5))

ax=us.gender.value_counts().plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=45)

ax.set(ylabel = 'Frequency', xlabel= 'Gender')

plt.show()
round(us.gender.value_counts()* 100 / len(us),2)
plt.figure(figsize=(10,5))

ax=us.race.value_counts().plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=45)

ax.set(ylabel = 'Frequency', xlabel= 'Race')

plt.show()
round(us.race.value_counts()* 100 / len(us),2)
plt.figure(figsize=(10,5))

ax=us.city.value_counts()[:5].plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=45)

ax.set(ylabel = 'Frequency', xlabel= 'City')

plt.show()
round(us.city.value_counts()* 100 / len(us),2)[:5]
plt.figure(figsize=(10,5))

ax=us.state.value_counts()[:5].plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=45)

ax.set(ylabel = 'Frequency', xlabel= 'State')

plt.show()
round(us.state.value_counts()* 100 / len(us),2)[:5]
plt.figure(figsize=(10,5))

ax=us.signs_of_mental_illness.value_counts().plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=45)

ax.set(ylabel = 'Frequency', xlabel= 'Signs of Mental Illness')

plt.show()
round(us.signs_of_mental_illness.value_counts()* 100 / len(us),2)
plt.figure(figsize=(10,5))

ax=us.threat_level.value_counts().plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=45)

ax.set(ylabel = 'Frequency', xlabel= 'Threat level')

plt.show()
round(us.threat_level.value_counts()* 100 / len(us),2)
plt.figure(figsize=(10,5))

ax=us.flee.value_counts().plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=45)

ax.set(ylabel = 'Frequency', xlabel= 'Flee')

plt.show()
round(us.flee.value_counts()* 100 / len(us),2)
plt.figure(figsize=(10,5))

ax=us.body_camera.value_counts().plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=45)

ax.set(ylabel = 'Frequency', xlabel= 'Body Camera')

plt.show()
round(us.body_camera.value_counts()* 100 / len(us),2)
plt.figure(figsize=(10,5))

ax=us.arms_category.value_counts().plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=45)

ax.set(ylabel = 'Frequency', xlabel= 'Arm Catergory')

plt.show()
round(us.arms_category.value_counts()* 100 / len(us),2)
us['arms_category']=us['arms_category'].replace(['Hand tools','Explosives','Electrical devices',

                                                'Piercing objects','Multiple'],'Other')

round(us.arms_category.value_counts()* 100 / len(us),2)
plt.figure(figsize=(10,5))

ax=us.arms_category.value_counts().plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=45)

ax.set(ylabel = 'Frequency', xlabel= 'Arm Catergory')

plt.show()
plt.figure(figsize=(10,5))

ax=us.Weapon.value_counts()[:5].plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=45)

ax.set(ylabel = 'Frequency', xlabel= 'Weapon')

plt.show()
round(us.Weapon.value_counts()* 100 / len(us),2)
us['Weapon']=us['Weapon'].replace(['shovel-Blunt instruments','nail gun-Piercing objects',

                                   'hammer-Blunt instruments', 'hatchet-Blunt instruments',

                                   'sword-Sharp objects', 'machete-Sharp objects',

                                   'box cutter-Sharp objects', 'metal object-Blunt instruments',

                                   'screwdriver-Piercing objects', 'lawn mower blade-Sharp objects',

                                   'flagpole-Blunt instruments', 'guns and explosives-Multiple',

                                   'cordless drill-Piercing objects', 'metal pole-Blunt instruments',

                                   'Taser-Electrical devices', 'metal pipe-Blunt instruments',

                                   'metal hand tool-Hand tools', 'blunt object-Blunt instruments',

                                   'metal stick-Blunt instruments', 'sharp object-Sharp objects',

                                   'meat cleaver-Sharp objects', 'carjack-Blunt instruments',

                                   'chain-Other unusual objects',

                                   "contractor's level-Other unusual objects",

                                   'stapler-Other unusual objects', 'crossbow-Piercing objects',

                                   'bean-bag gun-Guns', 'baseball bat and fireplace poker-Multiple',

                                   'straight edge razor-Sharp objects', 'gun and knife-Multiple',

                                   'ax-Blunt instruments', 'brick-Blunt instruments',

                                   'baseball bat-Blunt instruments',

                                   'hand torch-Other unusual objects', 'chain saw-Sharp objects',

                                   'garden tool-Blunt instruments', 'scissors-Sharp objects',

                                   'pole-Blunt instruments', 'pick-axe-Piercing objects',

                                   'flashlight-Other unusual objects',

                                   'spear-Piercing objects', 'chair-Other unusual objects',

                                   'pitchfork-Piercing objects', 'hatchet and gun-Multiple',

                                   'rock-Blunt instruments', 'piece of wood-Other unusual objects',

                                   'bayonet-Sharp objects', 'pipe-Blunt instruments',

                                   'glass shard-Sharp objects', 'motorcycle-Vehicles',

                                   'pepper spray-Other unusual objects',

                                   'metal rake-Blunt instruments', 'baton-Blunt instruments',

                                   'crowbar-Blunt instruments', 'oar-Other unusual objects',

                                   'machete and gun-Multiple',

                                   'air conditioner-Other unusual objects', 'pole and knife-Multiple',

                                   'beer bottle-Sharp objects', 'baseball bat and bottle-Multiple',

                                   'fireworks-Explosives', 'pen-Piercing objects',

                                   'chainsaw-Sharp objects', 'gun and sword-Multiple',

                                   'gun and car-Multiple', 'pellet gun-Guns', 'BB gun-Guns',

                                   'incendiary device-Explosives', 'samurai sword-Sharp objects',

                                   'bow and arrow-Multiple', 'gun and vehicle-Multiple',

                                   'vehicle and gun-Multiple', 'wrench-Blunt instruments',

                                   'walking stick-Blunt instruments',

                                   'barstool-Other unusual objects', 'grenade-Explosives',

                                   'BB gun and vehicle-Multiple', 'wasp spray-Other unusual objects',

                                   'air pistol-Guns', 'baseball bat and knife-Multiple',

                                   'vehicle and machete-Multiple', 'ice pick-Piercing objects',

                                   'car, knife and mace-Multiple'],'Other')

round(us.Weapon.value_counts()* 100 / len(us),2)
plt.figure(figsize=(10,5))

ax=us.Weapon.value_counts().plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=90)

ax.set(ylabel = 'Frequency', xlabel= 'Weapon')

plt.show()
us.pop('id')

us.head()
plt.figure(figsize=(10,5))

ax=us.year.value_counts().plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=90)

ax.set(ylabel = 'Frequency', xlabel= 'Year')

plt.show()
round(us.year.value_counts()* 100 / len(us),2)
us['Address']=us['city']+'-'+us['state']
plt.figure(figsize=(30,5))

ax=us.Address.value_counts()[:15].plot(kind="bar",color='Red')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=90)

ax.set(ylabel = 'Frequency', xlabel= 'Address')

plt.show()
round(us.Address.value_counts()* 100 / len(us),2)
plt.figure(figsize = (5,5))

sns.heatmap(us.corr(), annot = True, cmap="rainbow")

plt.show()
us.head()
us.drop(['date', 'armed','arms_category','city','state',

        'year', 'month', 'day', 'week', 'quarter','name'], 1, inplace = True)
us.head()
df_White = us[us['race'] == 'White']

df_White.head()
White_o=round((df_White.groupby(['race','gender','manner_of_death','Weapon']).size() / len(df_White) * 100),2)

White_o

df_Black = us[us['race'] == 'Black']

df_Black.head()
Black_o=round((df_Black.groupby(['race','gender','manner_of_death','Weapon']).size() / len(df_Black) * 100),2)

Black_o
df_Asian = us[us['race'] == 'Asian']

df_Asian.head()
Asian_o=round((df_Asian.groupby(['race','gender','manner_of_death','Weapon']).size() / len(df_Asian) * 100),2)

Asian_o
df_Native = us[us['race'] == 'Native']

df_Native.head()
Native_o=round((df_Native.groupby(['race','gender','manner_of_death','Weapon']).size() / len(df_Native) * 100),2)

Native_o
df_Other = us[us['race'] == 'Other']

df_Other.head()
Other_o=round((df_Other.groupby(['race','gender','manner_of_death','Weapon']).size() / len(df_Other) * 100),2)

Other_o
df_Hispanic = us[us['race'] == 'Hispanic']

df_Hispanic.head()
Hispanic_o=round((df_Hispanic.groupby(['race','gender','manner_of_death','Weapon']).size() / len(df_Hispanic) * 100),2)

Hispanic_o
Hispanic_p=round((df_Hispanic.groupby(['race','Address']).size() / len(us) * 100),2)

Hispanic_p
Black_p=round((df_Black.groupby(['race','Address']).size() / len(df_Black) * 100),2)

Black_p
Asian_p=round((df_Asian.groupby(['race','Address']).size() / len(df_Asian) * 100),2)

Asian_p
White_p=round((df_White.groupby(['race','Address']).size() / len(df_White) * 100),2)

White_p
Other_p=round((df_Other.groupby(['race','Address']).size() / len(df_Other) * 100),2)

Other_p
Native_p=round((df_Native.groupby(['race','Address']).size() / len(df_Native) * 100),2)

Native_p