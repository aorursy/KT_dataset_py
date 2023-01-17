# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/data-police-shootings/fatal-police-shootings-data.csv')

# Exploring the dataset
data.info()
cardinality = {}
for col in data:
    cardinality[col] = data[col].nunique()

cardinality
missing = data.isna().sum() * 100 / data.shape[0]

missing
data.armed.unique()
data.armed = data.armed.fillna('undetermined')
data['arms'] = data['armed']
lethal = ['gun', 'hatchet', 'machete', 'guns and explosives', 'gun and knife', 'ax', 'hand torch', 'chain saw', 'hatchet and gun', 'machete and gun', 'chainsaw', 'gun and sword', 'gun and car', 'incendiary device', 'gun and vehicle', 'vehicle and gun', 'grenade', 'air pistol', 'vehicle and machete']
semilethal = ['nail gun', 'knife', 'shovel', 'hammer', 'sword', 'lawn mower blade', 'cordless drill', 'crossbow', 'Taser', 'metal object', 'metal hand tool', 'metal stick', 'sharp object', 'meat cleaver', 'bean-bag gun', 'straight edge razor', 'baton', 'spear', 'bayonet', 'crowbar', 'tire iron', 'pole and knife', 'pellet gun', 'BB gun', 'samurai sword', 'bow and arrow', 'wrench', 'BB gun and vehicle', 'Airsoft pistol', 'baseball bat and knife', 'ice pick', 'car, knife and mace']
nonlethal = ['toy weapon', 'box cutter', 'screwdriver', 'flagpole', 'metal pole', 'pick-axe', 'metal rake', 'metal pipe', 'blunt object', 'carjack', 'chain', "contractor's level", 'stapler', 'beer bottle', 'baseball bat and fireplace poker', 'brick', 'baseball bat', 'garden tool', 'scissors', 'pole', 'flashlight', 'vehicle', 'chair', 'pitchfork', 'rock', 'piece of wood', 'pipe', 'glass shard', 'motorcycle', 'pepper spray', 'oar', 'air conditioner', 'baseball bat and bottle', 'fireworks', 'pen', 'walking stick', 'barstool', 'wasp spray']
unarmed = ['unarmed']
unknown = ['claimed to be armed', 'unknown weapon']
undetermined = ['undetermined']
for i in data.armed.unique():
    if i in lethal:
        data.armed = data.armed.replace(i, 'Lethal')
    elif i in semilethal:
        data.armed = data.armed.replace(i, 'Semi-Lethal')
    elif i in nonlethal:
        data.armed = data.armed.replace(i, 'Non-Lethal')
    elif i in unarmed:
        data.armed = data.armed.replace(i, 'Unarmed')
    elif i in unknown:
        data.armed = data.armed.replace(i, 'Unknown')
    elif i in undetermined:
        data.armed = data.armed.replace(i, 'Undetermined')
data.age.describe()
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')

data.age = imputer.fit_transform(data[['age']])

data.age.value_counts().sum()
data.race.value_counts()
data.race = data.race.fillna('Unknown')

data.race = data.race.replace('W', 'Caucasian').replace('B', 'African-American').replace('A', 'Asian').replace('N', 'Native-American').replace('H', 'Hispanic').replace('O', 'Other')

data.race.value_counts()
data.flee.value_counts()
data.flee = data.flee.fillna('Not fleeing')
data.threat_level.unique()
data.threat_level = data.threat_level.replace('attack', 'High').replace('other', 'Semi/Low').replace('undetermined', 'Undetermined')
data.gender = data.gender.replace('M', 'Male').replace('F', 'Female')
data.gender.value_counts()
genderless = data[data.gender.isnull()]

genderless
data = data.dropna(axis = 0)

data.info()
data['year'] = pd.to_datetime(data['date']).dt.year
data['month'] = pd.to_datetime(data['date']).dt.month
data['day'] = pd.to_datetime(data['date']).dt.day

data.info()
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize = (15, 5))

horder = ['Asian', 'African-American', 'Caucasian', 'Hispanic', 'Native-American', 'Other', 'Unknown']

sns.countplot(x = 'year', hue = 'race', data = data, palette = 'colorblind', hue_order = horder)
ax.legend(loc = 'upper right')
ax.set(ylim = (0, 550))
plt.xlabel('Year')
plt.ylabel('')
plt.title('Distribution of cases by race and year')
plt.show()
fig, ax = plt.subplots(figsize = (15, 10))
ax = sns.stripplot(x = 'age', y = 'race', data = data, hue = 'gender')
ax = sns.boxplot(x = 'age', y = 'race', data = data, palette = 'pastel', saturation = 0.5)
plt.xlabel('Age')
plt.ylabel('Racial Background')
ax.legend(loc = 'upper right')
plt.title('The Distribution of people according to Age, Gender and Race')
plt.show()
fig, ax = plt.subplots(figsize = (15, 5))

order = ['Lethal', 'Semi-Lethal', 'Non-Lethal', 'Unknown', 'Undetermined', 'Unarmed']
horder = ['Asian', 'African-American', 'Caucasian', 'Hispanic', 'Native-American', 'Other', 'Unknown']

sns.countplot(x = 'armed', hue = 'race', data = data, palette = 'colorblind', order = order, hue_order = horder)
ax.legend(loc = 'upper right')
ax.set(ylim = (0, 1600))
plt.xlabel('Type of Arm')
plt.ylabel('')
plt.title('Distribution of cases according to type of arm and race')
plt.show()
fig = plt.figure(figsize = (20, 5))

horder = ['Asian', 'African-American', 'Caucasian', 'Hispanic', 'Native-American', 'Other', 'Unknown']

ax1 = fig.add_subplot(1, 3, 1)
ax1 = sns.countplot(x = 'threat_level', hue = 'race', data = data, palette = 'colorblind', hue_order = horder)
ax1.get_legend().remove()
ax1.set(ylim = (0, 1800))
plt.xlabel('Level of Threat')
plt.ylabel('')

ax2 = fig.add_subplot(1, 3, 2)
ax2 = sns.countplot(x = 'signs_of_mental_illness', hue = 'race', data = data, palette = 'colorblind', hue_order = horder)
ax2.legend(loc = 'upper right')
ax2.set(ylim = (0, 1800))
plt.xlabel('Signs of Mental Illness')
plt.ylabel('')


ax3 = fig.add_subplot(1, 3, 3)
ax3 = sns.countplot(x = 'flee', hue = 'race', data = data, palette = 'colorblind', hue_order = horder)
ax3.get_legend().remove()
ax3.set(ylim = (0, 1800))
plt.xlabel('Fleeing')
plt.ylabel('')
plt.show()
fig = plt.figure(figsize = (15, 8))

horder = ['Asian', 'African-American', 'Caucasian', 'Hispanic', 'Native-American', 'Other', 'Unknown']
yorder = ['2015', '2016', '2017', '2018', '2019', '2020']

for i in range(1, 7):
    y = data.year.unique()
    ax = fig.add_subplot(2, 3, i)
    ax = sns.countplot(x = 'body_camera', hue = 'race', data = data[data.year == y[i - 1]], palette = 'colorblind', hue_order = horder)
    ax.set(ylim = (0, 500))
    ax.legend(loc = 'upper right')
    plt.xlabel('Body Camera ' + yorder[i - 1])
    plt.ylabel('')
plt.show()
statewise = data['state'].value_counts()[:10]
statewise = pd.DataFrame(statewise).reset_index()

citywise = data['city'].value_counts()[:10]
citywise = pd.DataFrame(citywise).reset_index()

fig = plt.figure(figsize = (20, 5))

ax1 = fig.add_subplot(1, 2, 1)
ax1 = sns.barplot(y = 'index', x = 'state', data = statewise, palette = 'cubehelix')
plt.ylabel('Top 10 States')
plt.xlabel('')

ax2 = fig.add_subplot(1, 2, 2)
ax2 = sns.barplot(y = 'index', x = 'city', data = citywise, palette = 'cubehelix')
plt.ylabel('Top 10 Cities')
plt.xlabel('')
plt.show()
california = data[data.state == 'CA']

fig = plt.figure(figsize = (24, 10))

order = ['Lethal', 'Semi-Lethal', 'Non-Lethal', 'Unknown', 'Undetermined', 'Unarmed']
horder = ['Asian', 'African-American', 'Caucasian', 'Hispanic', 'Native-American', 'Other', 'Unknown']

ax1 = fig.add_subplot(1, 3, 1)
ax1 = sns.countplot(data = california, y = 'year', hue = 'race', hue_order = horder, palette = 'colorblind')
ax1.get_legend().remove()
plt.ylabel('Number of Deaths by Year')
plt.xlabel('')

ax2 = fig.add_subplot(1, 3, 3)
ax2 = sns.countplot(data = california, y = 'armed', hue = 'race', order = order, hue_order = horder, palette = 'colorblind')
ax2.legend(loc = 'lower right')
plt.xlabel('Total Number of Deaths')
plt.ylabel('')

ax3 = fig.add_subplot(2, 3, 2)
ax3 = sns.countplot(data = california, x = 'threat_level', hue = 'race', hue_order = horder, palette = 'colorblind')
ax3.get_legend().remove()
plt.xlabel('Level of Threat')
plt.ylabel('')

ax4 = fig.add_subplot(2, 3, 5)
ax4 = sns.countplot(data = california, x = 'signs_of_mental_illness', hue = 'race', hue_order = horder, palette = 'colorblind')
ax4.get_legend().remove()
plt.xlabel('Signs of Mental Illness')
plt.ylabel('')

plt.show()
texas = data[data.state == 'TX']

fig = plt.figure(figsize = (24, 10))

order = ['Lethal', 'Semi-Lethal', 'Non-Lethal', 'Unknown', 'Undetermined', 'Unarmed']
horder = ['Asian', 'African-American', 'Caucasian', 'Hispanic', 'Native-American', 'Other', 'Unknown']

ax1 = fig.add_subplot(1, 3, 1)
ax1 = sns.countplot(data = texas, y = 'year', hue = 'race', hue_order = horder, palette = 'colorblind')
ax1.get_legend().remove()
plt.ylabel('Number of Deaths by Year')
plt.xlabel('')

ax2 = fig.add_subplot(1, 3, 3)
ax2 = sns.countplot(data = texas, y = 'armed', hue = 'race', order = order, hue_order = horder, palette = 'colorblind')
ax2.legend(loc = 'lower right')
plt.xlabel('Total Number of Deaths')
plt.ylabel('')

ax3 = fig.add_subplot(2, 3, 2)
ax3 = sns.countplot(data = texas, x = 'threat_level', hue = 'race', hue_order = horder, palette = 'colorblind')
ax3.get_legend().remove()
plt.xlabel('Level of Threat')
plt.ylabel('')

ax4 = fig.add_subplot(2, 3, 5)
ax4 = sns.countplot(data = texas, x = 'signs_of_mental_illness', hue = 'race', hue_order = horder, palette = 'colorblind')
ax4.get_legend().remove()
plt.xlabel('Signs of Mental Illness')
plt.ylabel('')

plt.show()
florida = data[data.state == 'FL']

fig = plt.figure(figsize = (24, 10))

order = ['Lethal', 'Semi-Lethal', 'Non-Lethal', 'Unknown', 'Undetermined', 'Unarmed']
horder = ['Asian', 'African-American', 'Caucasian', 'Hispanic', 'Native-American', 'Other', 'Unknown']

ax1 = fig.add_subplot(1, 3, 1)
ax1 = sns.countplot(data = florida, y = 'year', hue = 'race', hue_order = horder, palette = 'colorblind')
ax1.get_legend().remove()
plt.ylabel('Number of Deaths by Year')
plt.xlabel('')

ax2 = fig.add_subplot(1, 3, 3)
ax2 = sns.countplot(data = florida, y = 'armed', hue = 'race', order = order, hue_order = horder, palette = 'colorblind')
ax2.legend(loc = 'lower right')
plt.xlabel('Total Number of Deaths')
plt.ylabel('')

ax3 = fig.add_subplot(2, 3, 2)
ax3 = sns.countplot(data = florida, x = 'threat_level', hue = 'race', hue_order = horder, palette = 'colorblind')
ax3.get_legend().remove()
plt.xlabel('Level of Threat')
plt.ylabel('')

ax4 = fig.add_subplot(2, 3, 5)
ax4 = sns.countplot(data = florida, x = 'signs_of_mental_illness', hue = 'race', hue_order = horder, palette = 'colorblind')
ax4.get_legend().remove()
plt.xlabel('Signs of Mental Illness')
plt.ylabel('')

plt.show()
unwarranted = data[(data.armed == 'Unarmed') & (data.flee == 'Not fleeing')]

fig, ax = plt.subplots(figsize = (15, 5))

sns.countplot(data = unwarranted, x = 'year', hue = 'race', hue_order = horder, palette = 'colorblind')
ax.legend(loc = 'upper right')
plt.xlabel('Year')
plt.ylabel('Number of Killings')
plt.title('Distribution of unwarranted deaths by Year and Race')
plt.show()