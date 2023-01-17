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


import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv('/kaggle/input/us-police-shootings/shootings.csv')
df.info()
raceScore=[]

for i in np.arange(6):

    raceScore.append(df.race[df.race==df.race.unique()[i]].count())
plt.figure(figsize=(7,7)) 

labels = df.race.unique()

sizes = raceScore

colors = ['lightcoral', 'orange', 'gold', 'cyan', 'springgreen', 'mediumorchid']

explode = (0.3, 0.1, 0, 0.1, 0.1, 0.2)  # explode 1st slice



# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=180)

plt.title('Racial rates in data')

plt.axis('equal')

plt.show()
manner_of_deathScore=[]

for i in np.arange(df.manner_of_death.nunique()):

    manner_of_deathScore.append(df.manner_of_death[df.manner_of_death==df.manner_of_death.unique()[i]].count())
plt.figure(figsize=(7,7)) 

labels = df.manner_of_death.unique()

sizes = manner_of_deathScore

colors = ['springgreen', 'orange']

explode = (0.1, 0)  # explode 1st slice



# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', startangle=180)

plt.title('manner of death rates in data')

plt.axis('equal')

plt.show()
armedScore=[]

for i in np.arange(11):

    armedScore.append(df.armed.value_counts()[df.armed.value_counts()>15][i])

plt.figure(figsize=(20,20)) 

labels = df.armed.value_counts()[df.armed.value_counts()>15].index

sizes = armedScore

explode = (0,0,0,0,0,0,0.1,0.2,0.3,0.4, 0.5)

# Plot

plt.pie(sizes, labels=labels,

autopct='%1.1f%%', explode=explode, startangle=180,textprops={'fontsize': 14})

plt.axis('equal')

plt.title("types of arm")

plt.show()
df.age=df.age.replace(37.11793090137039, 37)
plt.figure(figsize=(16,4)) 

sns.set(style="darkgrid")

ax = sns.countplot(x="age", data=df)

plt.title("Age Distribution")

ax.set_xticklabels(ax.get_xticklabels(), rotation=70, horizontalalignment='right',fontsize=12);
genderScore=[]

for i in np.arange(df.gender.nunique()):

    genderScore.append(df.gender[df.gender==df.gender.unique()[i]].count())
plt.figure(figsize=(7,7)) 

labels = df.gender.unique()

sizes = genderScore

colors = ['gold', 'springgreen']

explode = (0.1, 0)  # explode 1st slice



# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', startangle=180)

plt.title("Gender rate")

plt.axis('equal')

plt.show()
CaRaceScore=[]

for i in np.arange(df.race[df.city == 'Los Angeles'].nunique()):

    CaRaceScore.append(df.race[df.city == 'Los Angeles'][df.race[df.city == 'Los Angeles']==df.race[df.city == 'Los Angeles'].unique()[i]].count())
plt.figure(figsize=(7,7)) 

labels = df.race[df.city == 'Los Angeles'].unique()

sizes = CaRaceScore

colors = [ 'gold', 'cyan', 'springgreen', 'mediumorchid']



# Plot

plt.pie(sizes,labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=180)

plt.title('Racial mortality rates in Los Angeles')

plt.axis('equal')

plt.show()
MhRaceScore=[]

for i in np.arange(df.race[df.city == 'Memphis'].nunique()):

    MhRaceScore.append(df.race[df.city == 'Memphis'][df.race[df.city == 'Memphis']==df.race[df.city == 'Memphis'].unique()[i]].count())
plt.figure(figsize=(7,7)) 

labels = df.race[df.city == 'Memphis'].unique()

sizes = MhRaceScore

colors = [ 'orange', 'gold', 'springgreen']



# Plot

plt.pie(sizes,labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=180)

plt.title('Racial mortality rates in Memphis')

plt.axis('equal')

plt.show()
OcScore=[]

for i in np.arange(df.race[df.city == 'Oklahoma City'].nunique()):

    OcScore.append(df.race[df.city == 'Oklahoma City'][df.race[df.city == 'Oklahoma City']==df.race[df.city == 'Oklahoma City'].unique()[i]].count())
plt.figure(figsize=(7,7)) 

labels = df.race[df.city == 'Oklahoma City'].unique()

sizes = OcScore

colors = ['lightcoral', 'orange', 'gold', 'cyan', 'springgreen', 'mediumorchid']



# Plot

plt.pie(sizes,labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=220)

plt.title('Racial mortality rates in Oklahoma City')

plt.axis('equal')

plt.show()
DallasRaceScore=[]

for i in np.arange(df.race[df.city == 'Dallas'].nunique()):

    DallasRaceScore.append(df.race[df.city == 'Dallas'][df.race[df.city == 'Dallas']==df.race[df.city == 'Dallas'].unique()[i]].count())
plt.figure(figsize=(7,7)) 

labels = df.race[df.city == 'Dallas'].unique()

sizes = DallasRaceScore

colors = ['lightcoral', 'springgreen', 'mediumorchid']



# Plot

plt.pie(sizes,labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=220)

plt.title('Racial mortality rates in Dallas')

plt.axis('equal')

plt.show()
PortlandRaceScore=[]

for i in np.arange(df.race[df.city == 'Portland'].nunique()):

    PortlandRaceScore.append(df.race[df.city == 'Portland'][df.race[df.city == 'Portland']==df.race[df.city == 'Portland'].unique()[i]].count())
plt.figure(figsize=(7,7)) 

labels = df.race[df.city == 'Portland'].unique()

sizes = PortlandRaceScore

colors = ['springgreen', 'mediumorchid']



# Plot

plt.pie(sizes,labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=220)

plt.title('Racial mortality rates in Portland')

plt.axis('equal')

plt.show()
signs_of_mental_illnessScore=[]

for i in np.arange(df.signs_of_mental_illness.nunique()):

    signs_of_mental_illnessScore.append(df.signs_of_mental_illness[df.signs_of_mental_illness==df.signs_of_mental_illness.unique()[i]].count())

plt.figure(figsize=(7,7)) 

labels = df.signs_of_mental_illness.unique()

sizes = signs_of_mental_illnessScore

colors = ['gold', 'springgreen']

explode = (0.1, 0)  # explode 1st slice



# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', startangle=180)

plt.title("signs of mental illness rate")

plt.axis('equal')

plt.show()
threatScore=[]

for i in np.arange(df.threat_level.nunique()):

    threatScore.append(df.threat_level[df.threat_level==df.threat_level.unique()[i]].count())

plt.figure(figsize=(7,7)) 

labels = df.threat_level.unique()

sizes = threatScore



# Plot

plt.pie(sizes, labels=labels,autopct='%1.1f%%', startangle=180)

plt.title("Gender rate")

plt.axis('equal')

plt.show()
fleeScore=[]

for i in np.arange(df.flee.nunique()):

    fleeScore.append(df.flee[df.flee==df.flee.unique()[i]].count())

plt.figure(figsize=(7,7)) 

labels = df.flee.unique()

sizes = fleeScore



# Plot

plt.pie(sizes, labels=labels,autopct='%1.1f%%', startangle=180)

plt.title("Flee rates")

plt.axis('equal')

plt.show()
cameraScore=[]

for i in np.arange(df.body_camera.nunique()):

    cameraScore.append(df.body_camera[df.body_camera==df.body_camera.unique()[i]].count())

plt.figure(figsize=(7,7)) 

labels = df.body_camera.unique()

sizes = cameraScore



# Plot

plt.pie(sizes, labels=labels,autopct='%1.1f%%', startangle=180)

plt.title("body camera rates")

plt.axis('equal')

plt.show()
plt.figure(figsize=(12,4)) 

sns.set(style="darkgrid")

ax = sns.countplot(x="arms_category", data=df,palette="Set3")

plt.title("arms category count")

ax.set_xticklabels(ax.get_xticklabels(), rotation=70,fontsize=12);
for i in np.arange(df.date.count()):

    df.date[i]=df.date[i][0:4]
plt.figure(figsize=(8,5)) 

sns.set(style="darkgrid")

ax = sns.countplot(x='date', hue='gender', data=df,palette="Set3")

plt.title("deaths by years")

ax.set_xticklabels(ax.get_xticklabels(), rotation=70);



g = sns.catplot(x="race", hue="gender", col="date",

                data=df, kind="count",

                height=4, aspect=.7,palette="Set2")

g.set_xticklabels(rotation=70);

plt.subplots_adjust(top=0.8)

g.fig.suptitle('Racial Death Numbers by years')

g.set(ylim=(-10, 480))

g.fig.set_size_inches(13,4)
count = []
for i in np.arange(df.state.nunique()):

    count.append(df[df.state.unique()[i]==df.state].count().id)
plt.figure(figsize=(15,5))

plt.title("number of deaths by state")

sns.set_color_codes("pastel")

sns.barplot(x=df.state.unique()[0:25], y=count[0:25])
plt.figure(figsize=(15,5)) 

plt.title("number of deaths by state")

sns.set_color_codes("pastel")

sns.barplot(x=df.state.unique()[26:51], y=count[26:51])