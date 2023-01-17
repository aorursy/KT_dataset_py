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

import missingno as msno



data = pd.read_csv('../input/fifa19/data.csv', encoding='utf-8')

data.shape
data.columns
msno.matrix(data)
data.dropna(subset=['Name', 'Overall', 'Value'], how='all', inplace=True)

data.shape
def scatter_them(data, col1 = 'Age', col2='Overall'):

    plt.figure(figsize=(13,7))

    plt.title("{0} AND {1}".format(col1, col2))

    plt.scatter(data[col1], data[col2], color='dodgerblue', s=6)

    plt.xlabel(col1)

    plt.ylabel(col2)
df = data.copy()

scatter_them(df)
def box_them(data, col1='Age', col2='Overall', intervals=[15, 20, 25, 30, 35, 40, 50]):

    df = data.copy()

    df['for_purpose'] = pd.cut(df[col1], intervals, labels=[str(point) for point in intervals[1:]])

    plt.figure(figsize=(13,7))

    sns.boxplot(data=df, x='for_purpose', y=col2)

    plt.title("{0} AND {1}".format(col1, col2))
df = data.copy()

box_them(data=df)
box_them(data=df, col2='Potential')
df = data.copy()

df['Value']
def first_and_last(x):

    return (x[0], x[-1])



units = df['Value'].apply(first_and_last)

set(units)
def change_money(x):

    if x[-1] == '0':

        return int(x[1:])

    elif x[-1] == 'K':

        return 1000*float(x[1:-1])

    else:

        return 1000000*float(x[1:-1])
data['Value'] = data['Value'].apply(change_money)
data['Value'].min(), data['Value'].max()
df = data.copy()



df.sort_values(by='Value', ascending=False)[['Name', 'Club', 'Nationality', 'Age', 'Value', 'Overall']].iloc[:10]
box_them(data=df, col1='Age', col2='Value')
by_value = df.sort_values(by='Value', ascending=False)



box_them(data=by_value[:1000], col1='Age', col2='Value')
box_them(data=by_value[500:], col1='Age', col2='Value')
def line_them(data=df, col='Overall', groupby='Age'):

    x = df.groupby(groupby)[col].mean()

    plt.figure(figsize=(13,7))

    plt.title("{0} Over {1}".format(col, groupby))

    plt.plot(x, color='dodgerblue')

    plt.xlabel(groupby)

    plt.ylabel(col)
line_them()
df[df['Age']>=42].sort_values(by='Overall', ascending=False)
df = data[data['Age']<42].copy()

line_them(data=df, col='Overall', groupby='Age')
line_them(col='Value', groupby='Age')
line_them(col='Potential')
plt.figure(figsize=(13,7))

plt.plot(df.groupby('Age')['Overall'].mean(), color='red')

plt.plot(df.groupby('Age')['Potential'].mean(), color='darkblue')

plt.title("Overall and Potential stats over time")

plt.xlabel('Age')

plt.ylabel('Potential and Overall')
df.groupby('Age')['Overall'].mean() == df.groupby('Age')['Potential'].mean()
stats = df[['Crossing',

       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

       'Marking', 'StandingTackle']]



f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(stats.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
stats.corr().sum().sort_values(ascending=False)
top5 = stats.corr().sum().sort_values(ascending=False)[:5]

bot5 = stats.corr().sum().sort_values(ascending=False)[-5:]



print("The top 5 most influencial stats were \n{0} \nand the bottom 5 were \n{1}".format(top5, bot5))
df = data.copy()



stat_and_val = df[['Overall', 'Value', 'Crossing',

       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

       'Marking', 'StandingTackle']]



stat_and_val.corr()[['Value', 'Overall']].sort_values(by='Value', ascending=False).drop(index=['Value', 'Overall'])
df = data.copy()



## TOP 10 most clubs with the best players

df.groupby("Club")['Overall'].mean().sort_values(ascending=False)[:10]
## TOP 10 clubs with the most pricy guys

df.groupby("Club")['Value'].mean().sort_values(ascending=False)[:10]
df['Wage'][:5]
data['Wage'] = data['Wage'].apply(change_money)



df = data.copy()



df.groupby("Club")['Wage'].sum().sort_values(ascending=False)[:10]

## TOP 10 clubs with highest payrolls
for_overall = df.groupby("Club")['Overall'].mean()

for_wage = df.groupby("Club")['Wage'].sum()



wage_efficiency = for_overall / for_wage

wage_efficiency.sort_values(ascending=False)[:10]
## BOTTOM 10 in terms of wage efficiency

wage_efficiency.sort_values()[:10]
big_clubs = df.groupby("Club")['Overall'].mean().sort_values(ascending=False)[:10].index

big_clubs
df = data.copy()

df[df['Club'].isin(big_clubs)].groupby('Club')['ShortPassing'].mean().sort_values(ascending=False)
df[df['Club'].isin(big_clubs)].groupby('Club')['SprintSpeed'].mean().sort_values(ascending=False)
df[df['Club'].isin(big_clubs)].groupby('Club')['BallControl'].mean().sort_values(ascending=False)
df[df['Club'].isin(big_clubs)].groupby('Club')['Aggression'].mean().sort_values(ascending=False)
df[df['Club'].isin(big_clubs)].groupby('Club')['Age'].mean().sort_values(ascending=False)
df = data.copy()

df.keys()
positions = df[['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',

       'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',

       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']]



positions.head()
df.iloc[3]
for_position = df.dropna(subset=['LS', 'ST', 'RCB', 'CM'], how='all')
positions = for_position[['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',

       'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',

       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']]
len(for_position), len(df)
def position_value(x):

    return int(x[:2])+int(x[3])



positions = positions.applymap(position_value)
positions.iloc[3][:3]
main_position = positions.idxmax(axis=1)



for_position['Position'] = main_position
for_position.drop(columns=['Unnamed: 0'], inplace=True)

for_position.head()
for_position['Position']
positions.columns
for_position[for_position['Position']=='LW'].sort_values(by='Overall', ascending=False).iloc[:3]
forward = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW']

midfield = ['LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM']

defense = ['RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']



fw = for_position[for_position['Position'].isin(forward)]

mf = for_position[for_position['Position'].isin(midfield)]

df = for_position[for_position['Position'].isin(defense)]



len(fw), len(mf), len(df)
## Statements on positions aggregate



print("Among pro field players, number of forwards were {0}({1:.2f}%), midfielders were {2}({3:.2f}%), and defenders were {4}({5:.2f}%).".format(len(fw), len(fw)*100/len(for_position), len(mf), len(mf)*100/len(for_position), len(df), len(df)*100/len(for_position)))