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
data=pd.read_csv('/kaggle/input/fifa19/data.csv')
import seaborn as sns

sns.catplot(x="Position", y="Age", data=data,kind='violin',aspect=3,order=data.groupby('Position')['Age'].mean().sort_values(ascending=True).index.values)
sns.catplot(x="Position", y="Overall", data=data,kind='violin',aspect=3, order=data.groupby('Position')['Overall'].mean().sort_values(ascending=True).index.values)
sns.catplot(y="Overall", x="Age",data=data,kind='strip',aspect=3)
data2=pd.read_csv('/kaggle/input/fifa19/data.csv')

data2['Wage']=data2['Wage'].str[1:-1]

data2=data2[data2['Wage']!='']

data2['Wage']=data2['Wage'].astype(int)
sns.catplot(y="Wage", x="Position",data=data2,kind='strip',aspect=3,order=data2.groupby('Position')['Wage'].mean().sort_values(ascending=True).index.values)
sns.catplot(x='Age',y='Wage',data=data2,kind='box',aspect=3)
data_3=pd.read_csv('/kaggle/input/fifa19/data.csv')
for i in range(len(data_3['Position'])):

    if data_3.loc[i,'Position']=='ST' or data_3.loc[i,'Position']=='LS' or data_3.loc[i,'Position']=='RS':

        data_3.loc[i,'Position']='F'

    if data_3.loc[i,'Position']=='CF' or data_3.loc[i,'Position']=='RF' or data_3.loc[i,'Position']=='LF':

        data_3.loc[i,'Position']='F'

    if data_3.loc[i,'Position']=='LW' or data_3.loc[i,'Position']=='RW':

        data_3.loc[i,'Position']='AM'

    if data_3.loc[i,'Position']=='LAM' or data_3.loc[i,'Position']=='CAM' or data_3.loc[i,'Position']=='RAM':

        data_3.loc[i,'Position']='AM'

    if data_3.loc[i,'Position']=='LM' or data_3.loc[i,'Position']=='LCM' or data_3.loc[i,'Position']=='CM' or data_3.loc[i,'Position']=='RCM' or data_3.loc[i,'Position']=='RM':

        data_3.loc[i,'Position']='M'

    if data_3.loc[i,'Position']=='LWB' or data_3.loc[i,'Position']=='LDM' or data_3.loc[i,'Position']=='CDM' or data_3.loc[i,'Position']=='RDM' or data_3.loc[i,'Position']=='RWB':

        data_3.loc[i,'Position']='DM'

    if data_3.loc[i,'Position']=='LB' or data_3.loc[i,'Position']=='LCB' or data_3.loc[i,'Position']=='CB' or data_3.loc[i,'Position']=='RCB' or data_3.loc[i,'Position']=='RB':

        data_3.loc[i,'Position']='D'
sns.catplot(x="Age", y="Overall",row="Position", data=data_3, aspect=4)
data_3['Wage']=data_3['Wage'].str[1:-1]

data_3=data_3[data_3['Wage']!='']

data_3['Wage']=data_3['Wage'].astype(int)
sns.catplot(x="Overall", y="Wage",row="Position", data=data_3, aspect=4)
data_4=data_3[['Crossing',

       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

       'Marking', 'StandingTackle', 'SlidingTackle','Overall']]

data_4.dropna(inplace=True)

X = data_4[['Crossing',

       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

       'Marking', 'StandingTackle', 'SlidingTackle']]

y = data_4[['Overall']]
from yellowbrick.model_selection import FeatureImportances

from yellowbrick.features import RadViz

from sklearn.linear_model import LinearRegression

visualizer = RadViz(size=(1080, 720))

model=LinearRegression()

viz = FeatureImportances(model)

viz.fit(X, y)

viz.show()
viz.score(X,y)