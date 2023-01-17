# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# sns.set(style="dark")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
PlayList = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayList.csv')

PlayerTrackData = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv')

InjuryRecord = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/InjuryRecord.csv')
PlayList.head()
PlayList.info()
PlayList.isna().sum()
PlayList.describe(include='all')
PlayList.RosterPosition.value_counts()
def bar_lables(var, tit, df, size=1):

    f, ax = plt.subplots(1, 1, figsize=(4*size, 5))

    total = float(len(df))

    g = sns.countplot(df[var], order=df[var].value_counts().index, palette='Blues', edgecolor='Navy')

    g.set_title(f"Count per {tit}", fontsize=20)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=10)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x() + p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total), ha='center', fontsize=10)
bar_lables('RosterPosition', 'Position', PlayList, size=4)
PlayList.StadiumType.value_counts()
List_Sta_typ = PlayList["StadiumType"].astype(str).str.strip().str.lower().unique()

# print(List_Sta_typ)



List_initials = []



for i in range(len(List_Sta_typ)):

    List_initials.append(List_Sta_typ[i][0:3])
List_initials = set(List_initials)



for j in List_initials:

    print(f"Initial: {j}")

    print(PlayList[PlayList["StadiumType"].fillna("").str.lower().str.startswith(j)].StadiumType.unique())

    print()
outdoor = ['Outdoor', 'Outdoors', 'Cloudy', 'Heinz Field', 

              'Outdor', 'Ourdoor', 'Outside', 'Outddors', 

              'Outdoor Retr Roof-Open', 'Oudoor', 'Bowl']



indoor_closed = ['Indoors', 'Indoor', 'Indoor, Roof Closed', 'Indoor, Roof Closed',

                   'Retractable Roof', 'Retr. Roof-Closed', 'Retr. Roof - Closed', 'Retr. Roof Closed']



indoor_open = ['Indoor, Open Roof', 'Open', 'Retr. Roof-Open', 'Retr. Roof - Open']



dome_closed = ['Dome', 'Domed, closed', 'Closed Dome', 'Domed', 'Dome, closed']



dome_open = ['Domed, Open', 'Domed, open']



List_Types_Stat = [outdoor, indoor_closed, indoor_open, dome_closed, dome_open]
conditions_stadium = [PlayList['StadiumType'].isin(outdoor),

                      PlayList['StadiumType'].isin(indoor_closed),

                      PlayList['StadiumType'].isin(indoor_open),

                      PlayList['StadiumType'].isin(dome_closed),

                      PlayList['StadiumType'].isin(dome_open)]



List_Types_Stat_name = ["outdoor", "indoor_closed", "indoor_opened", "dome_closed", "dome_opened"]



PlayList['StadiumType_cl'] = np.select(conditions_stadium, List_Types_Stat_name, default='others')
PlayList['StadiumType_cl'].unique() # Cleaned data for Stadium Type
# plt.figure(figsize=(10,6))

# ax = sns.countplot(x='StadiumType_cl', order=PlayList.StadiumType_cl.value_counts().index, data=PlayList)

# ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')

# plt.title("Qty per Stadium Type")

# plt.tight_layout()



bar_lables('StadiumType_cl', 'Statium Type', PlayList, size=4)
PlayList['FieldType'].value_counts(normalize=True) * 100
# # sns.countplot(x='FieldType', data = PlayList)

# # plt.title("Qty per FieldType")



# bar_lables('FieldType', 'Field Type', PlayList)



colors = ['royalblue', 'cornflowerblue']



PlayList.FieldType.str.get_dummies().sum().plot.pie(label='FieldType', 

                                                    autopct='%1.0f%%', 

                                                    colors=colors, startangle=90, fontsize=15)

plt.axis('equal');
# plt.figure(figsize=(10,8))

# sns.countplot(x='FieldType', hue='StadiumType_cl', data = PlayList)



g = sns.catplot(x='StadiumType_cl', col='FieldType', data=PlayList, kind='count',

                height=6, palette='Blues', aspect=.7)

g.set_xticklabels(rotation=45)
InjuryRecord.head()
InjuryRecord['BodyPart'].value_counts()
plt.figure(figsize=(10,6))

ax = sns.countplot(x='BodyPart', order=InjuryRecord.BodyPart.value_counts().index, data=InjuryRecord, palette='Blues')

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')

plt.title("Qty per Body Part")

plt.tight_layout()
InjuryRecord
sns.catplot(x='BodyPart', col='Surface', data=InjuryRecord, kind='count',

                height=6, palette='Blues', aspect=.7)
# InjuryRecord.loc.groupby('BodyPart').sum()

InjuryRecord[['BodyPart','DM_M1','DM_M7','DM_M28','DM_M42']].groupby('BodyPart').sum()
Inj_Surf_BodyPart = InjuryRecord.groupby(['Surface','BodyPart'])['DM_M1','DM_M7','DM_M28','DM_M42'].sum().reset_index()

Inj_Surf_BodyPart
Inj_Surf_BodyPart_New = pd.melt(Inj_Surf_BodyPart, id_vars=['Surface','BodyPart'], value_vars=['DM_M1','DM_M7','DM_M28','DM_M42'])



sns.catplot(x='BodyPart', y='value', hue='Surface', col='variable',

            data=Inj_Surf_BodyPart_New, kind='bar', height=4, aspect=.7, palette='Blues')
Relations_table = pd.merge(PlayList, InjuryRecord, how='left', on='PlayerKey')
col = ['PlayerKey', 'RosterPosition', 'PlayerDay', 'FieldType', 'Temperature', 'PlayType', 'Position', 'PositionGroup',

       'StadiumType_cl', 'BodyPart', 'Surface', 'DM_M1', 'DM_M7', 'DM_M28', 'DM_M42']



Relations_table = Relations_table[col].copy()
Relations_table.info()
sns.heatmap(Relations_table.isnull(), cbar=False)
Relations_table.dropna(inplace=True)
Relations_table.info()
Relations_table
plt.figure(figsize=(16,8))

g = sns.countplot(y="PlayType", hue="Surface", 

                  order=Relations_table['PlayType'].value_counts().index, 

                  data=Relations_table, palette='Blues')

plt.title("Compare PlayType per Surface", fontsize=20)
sns.catplot(x="Temperature", y="Surface", row="BodyPart",

                kind="box", orient="h", height=1.5, aspect=5,

                data=Relations_table.query("Temperature > 0"), palette='Blues');
sns.catplot(x="Temperature", y="Surface", row="PlayType",

                kind="box", orient="h", height=1.5, aspect=5,

                data=Relations_table.query("Temperature > 0"), palette='Blues');
plt.figure(figsize=(16,8))

g = sns.countplot(y="Position", hue="Surface", 

                  order=Relations_table['Position'].value_counts().index, 

                  data=Relations_table, palette='Blues')

plt.title("Injuries: Position per Surface", fontsize=20)