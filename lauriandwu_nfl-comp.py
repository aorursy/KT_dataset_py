# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

%config InlineBackend.figure_format ='retina'

%matplotlib inline
injury = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/InjuryRecord.csv')

playlist = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayList.csv')

#playertrack = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv')
injury.info()

playlist.info()

#playertrack.info()
injury.sort_values(by='PlayerKey')
# check for nan

injury.isnull().sum()

playlist.isnull().sum()
injury['PlayerKey'].nunique()#.head().sort_values(by='PlayerKey')
injury.head()

injury['BodyPart'].unique()
injury.Surface.value_counts()
sns.countplot(injury["Surface"])

plt.box(False)

plt.title('Injury Distribution Surfaces')
#injury['BodyPart'].value_counts().plot(kind='bar')



sns.countplot(injury["BodyPart"])

plt.box(False)

plt.title('Distribution Body Parts Injured')
injury.BodyPart.value_counts()
order = injury['Surface'].value_counts(ascending=True).index

sns.countplot(x='Surface', hue = 'BodyPart', data=injury, order = order)

plt.box(False)

plt.title("Injuries in Body Part by Surface Type")
injury.head()

injury.isnull().sum()
injury.apply(pd.Series.value_counts)
injury.head()
injury.DM_M1.value_counts()

injury.DM_M7.value_counts()

injury.DM_M28.value_counts()

injury.DM_M42.value_counts()
# undummy time missing after injuries

injury.loc[(injury['DM_M1'] + injury['DM_M7'] + injury['DM_M28'] + injury['DM_M42'] == 4), 'Time_missing'] = '42+ days'

injury.loc[(injury['DM_M1'] + injury['DM_M7'] + injury['DM_M28'] + injury['DM_M42'] == 3), 'Time_missing'] = '28+ days'

injury.loc[(injury['DM_M1'] + injury['DM_M7'] + injury['DM_M28'] + injury['DM_M42'] == 2), 'Time_missing'] = '7+ days'

injury.loc[(injury['DM_M1'] + injury['DM_M7'] + injury['DM_M28'] + injury['DM_M42'] == 1), 'Time_missing'] = '1+ days'



injury.head()
#visualize time missing

plt.figure(figsize=(10,6))

sns.countplot(x='Time_missing', hue = 'BodyPart', data=injury, order = ['42+ days','28+ days','7+ days','1+ days'])#palette="Set3"

plt.box(False)

plt.title("")
#visualize time missing

plt.figure(figsize=(10,6))

sns.countplot(x='BodyPart', hue = 'Time_missing', data=injury, order = ['Knee','Ankle','Foot','Toes','Heel'],palette="nipy_spectral_r")

plt.box(False)

plt.legend(loc='upper right')
plt.figure(figsize=(10,7))

order3 = injury['Surface'].value_counts(ascending=True).index

sns.countplot(x='Surface', hue = 'Time_missing', data=injury, order = order3, palette="Set3")

plt.box(False)

plt.legend(loc='upper right')

plt.show()
# Statistical Test on whether significant

#chi-square?
playlist['PlayerKey'].nunique()
comb = pd.merge(playlist, injury, on='PlayerKey', how='outer')

comb.head(10)

comb.info()


comb['BodyPart'] = comb.BodyPart.fillna('noInjury')
comb.BodyPart.unique()
comb.Surface.value_counts()

comb.Surface.isnull().sum()
comb[comb['GameID_x'] == '26624-1']
comb[['DM_M1','DM_M7','DM_M28','DM_M42']] = comb[['DM_M1','DM_M7','DM_M28','DM_M42']].fillna(0)
comb = comb.drop(['GameID_y','PlayKey_y','Surface'], axis=1)

comb
comb.loc[(comb['DM_M1'] + comb['DM_M7'] + comb['DM_M28'] + comb['DM_M42'] == 0), 'Time_missing'] = 'none'
comb.head()
#visualize Position and Injury

plt.figure(figsize=(18,6))

sns.countplot(x='RosterPosition', hue = 'BodyPart', data=comb, palette="nipy_spectral_r")

plt.box(False)

plt.legend(loc='upper right')
plt.figure(figsize=(20,6))

sns.countplot(comb.StadiumType)

plt.box(False)
plt.figure(figsize=(20,6))

sns.countplot(comb.FieldType)

plt.box(False)



comb.FieldType.value_counts()
#visualize FieldType and Injury

plt.figure(figsize=(18,6))

sns.countplot(x='FieldType', hue = 'BodyPart', data=comb, palette="nipy_spectral_r")

plt.box(False)

plt.legend(loc='upper right')
#visualize FieldType and Injury



#comb['Perc_FieldType']= comb.FieldType.value_counts('Synthetic')/comb.FieldType.value_counts()
comb.head()