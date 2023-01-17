import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/industrial-safety-and-health-analytics-database/IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv')

data = data.drop(['Unnamed: 0'], axis=1)

data.shape
import matplotlib.pyplot as plt

import seaborn as sns
sns.countplot(y='Employee or Third Party', data=data, orient='h', palette='Blues_r')

acc_level = data['Employee or Third Party'].value_counts()

acc_level
# there are some typos in the data, the columns Genre is actually meant to be Gender.

sns.countplot(y='Genre', data=data, orient='h', palette='Greens')

acc_level = data['Genre'].value_counts()

acc_level


level_map = {'I': 1, 'II': 2,'III': 3 , 'IV' : 4, 'V': 5, 'VI' : 6}

data['Accident Level'] = pd.Series([level_map[x] for x in data['Accident Level']], index=data.index)
sns.countplot(x='Accident Level', data=data, palette='cool')

acc_level = data['Accident Level'].value_counts()

print("Ratio of level 1 incedents to level 2+ incedents: ",round(acc_level[1]/(acc_level.sum()-acc_level[1]), 3),'\n\n')

print(acc_level)
acc_level = data.loc[data['Employee or Third Party'] == 'Third Party']['Accident Level'].value_counts()

print("Ratio of level 1 incedents to level 2+ incedents, Third Party: ",round(acc_level[1]/(acc_level.sum()-acc_level[1]), 3),'\n\n')

print(acc_level)



acc_level = data.loc[data['Employee or Third Party'] == 'Third Party (Remote)']['Accident Level'].value_counts()

print("\n\n\nRatio of level 1 incedents to level 2+ incedents, Third Party (Remote): ",round(acc_level[1]/(acc_level.sum()-acc_level[1]), 3),'\n\n')

print(acc_level)



sns.catplot(x="Accident Level",col="Employee or Third Party",



                data=data, kind="count",



                height=4, aspect=1.2, palette='winter')
246/179
acc_level = data.loc[data['Genre'] == 'Female']['Accident Level'].value_counts()

print("Ratio of level 1 incedents to level 2+ incedents: ",round(acc_level[1]/(acc_level.sum()-acc_level[1]), 3),'\n\n')

print(acc_level)





sns.catplot(x="Accident Level", col="Genre",



                data=data, kind="count",



                height=4, aspect=1.2, palette='winter')