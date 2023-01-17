# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt

%matplotlib inline

# Any results you write to the current directory are saved as output.
fifa = pd.read_csv('../input/FullData.csv', parse_dates=['Club_Joining'])

fifa.head()
fifa.shape
fifa.columns
fifa['Name'].unique().tolist()[:5]
fifa['Club'].unique().tolist()[:5]
fifa.isnull().sum()
speed = fifa.groupby('Name')[['Acceleration','Speed','Stamina','Strength', 'Balance','Agility']].sum().sort_values(by='Speed', ascending=False)

speed[:10]
movement = fifa.groupby('Name')[['Acceleration','Speed','Stamina','Reactions', 'Balance','Agility']].sum().sort_values(by='Stamina', ascending=False)

movement.sum(axis=1)[:10]
movement.sum(axis=1)[:10].plot('bar')

plt.xlabel('Player name')

plt.ylabel('Movement -Acceleration, Speed, Stamina, Reactions, Balance, Agility')

plt.title('Player Movement')
Skill = fifa.groupby('Name')[['Dribbling', 'Curve', 'Freekick_Accuracy', 'Long_Pass', 'Ball_Control']].sum()

Skill.sum(axis=1).nlargest(10)
fifa[fifa['Name']=='Felipe']
fifa['Rating']
speedm = fifa.groupby(fifa.Name=='Lionel Messi')['Acceleration','Speed','Stamina','Strength', 'Balance','Agility']

speedm.sum(axis=1)

fifa['Club_Position'].unique()
ClubPosition = pd.Categorical(fifa["Club_Position"])



ClubPosition = ClubPosition.rename_categories([

 'LW', 'RW', 'ST', 'GK', 'Sub', 'RCM', 'CAM', 'LCB', 'LCM', 'RS',

       'RB', 'RCB', 'LM', 'LDM', 'RM', 'LB', 'CDM', 'RDM', 'LF', 'CB',

       'LAM', 'Res', 'CM', 'LS', 'RF', 'RWB', 'RAM', 'LWB',  'CF'])            

Cposition = ClubPosition.dropna()

Cposition

print(Cposition.describe())
Cposition.describe().plot(kind='barh', figsize=(10,9))

plt.ylabel('Club Position')