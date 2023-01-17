# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="white", color_codes=True)
fifa = pd.read_csv("../input/CompleteDataset.csv", skipinitialspace=True)

fifa
fifa.columns
fifa.shape
fifa = fifa.drop('Unnamed: 0', axis=1)
fifa = fifa[['Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Wage', 'Club', 'Value', 'Preferred Positions']]

fifa.head(5)
fifa['Growth'] = fifa['Potential'] - fifa['Overall']

fifa.head(5)
fifa_growth = fifa.groupby(['Age'])['Growth'].mean()

fifa_potential = fifa.groupby(['Age'])['Potential'].mean()

fifa_overall = fifa.groupby(['Age'])['Overall'].mean()

summary = pd.concat([fifa_growth, fifa_potential, fifa_overall], axis=1)



axis = summary.plot()

plt.show()

axis.set_ylabel('Rating')

axis.set_title('Average growth potential by age')
counter = 85

players = fifa[fifa['Overall']>counter]

# Grouping the players by club

group = players.groupby('Club')

number_of_players = group.count()['Name'].sort_values(ascending=False)

ax = sns.countplot(x='Club', data=players, order=number_of_players.index)

ax.set_xticklabels(labels = number_of_players.index, rotation='vertical')

ax.set_ylabel('Number of players (Over 85)')

ax.set_xlabel('Club')

ax.set_title('Top players (Overall > %.i)' %counter)
def extract_value_from(value):

    out = value.replace('€', '')

    if 'M' in out:

        out = float(out.replace('M', ''))*1000000

    elif 'K' in value:

        out = float(out.replace('K', ''))*1000

    return float(out)
fifa['Value'] = fifa['Value'].apply(lambda x: extract_value_from(x))

fifa['Wage'] = fifa['Wage'].apply(lambda x: extract_value_from(x))
fifa_wage = fifa.groupby(['Overall'])['Wage'].mean()

fifa_value = fifa.groupby(['Overall'])['Value'].mean()

fifa_wage = fifa_wage.apply(lambda x: x/1000)

fifa_value = fifa_value.apply(lambda x: x/1000000)

fifa["Wage(by Potential)"] = fifa["Wage"]

fifa["Value(by Potential)"] = fifa["Value"]

fifa_wage_p = fifa.groupby(['Potential'])['Wage(by Potential)'].mean()

fifa_value_p = fifa.groupby(['Potential'])['Value(by Potential)'].mean()

fifa_wage_p = fifa_wage_p.apply(lambda x: x/1000)

fifa_value_p = fifa_value_p.apply(lambda x: x/1000000)

summary = pd.concat([fifa_wage, fifa_value, fifa_wage_p, fifa_value_p], axis=1)



axis = summary.plot()

axis.set_ylabel('Wage / Value')

axis.set_title('Average Wage / Value by Rating')
fifa_wage_a = fifa.groupby(['Age'])['Wage'].mean()

fifa_value_a = fifa.groupby(['Age'])['Value'].mean()

fifa_wage_a = fifa_wage_a.apply(lambda x: x/1000)

fifa_value_a = fifa_value_a.apply(lambda x: x/1000000)

summary = pd.concat([fifa_wage_a, fifa_value_a], axis=1)



axis = summary.plot()

axis.set_ylabel('Wage / Value')

axis.set_title('Average Age')