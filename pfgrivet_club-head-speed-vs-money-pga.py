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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#Import PGA Historical Data

#2019_data = pd.read_csv("../input/pga-tour-20102018-data/2019_data.csv")

#2020_data = pd.read_csv("../input/pga-tour-20102018-data/2020_data.csv")

PGA = pd.read_csv("../input/pga-tour-20102018-data/PGA_Data_Historical.csv")
results_PGA_Money = PGA[PGA['Variable'] == "Official Money - (MONEY)"]
results_PGA_Money.head()
# Need to modify values within 'Values'

results_PGA_Money['Value']= results_PGA_Money['Value'].apply(lambda value:value.replace('$',''))

results_PGA_Money['Value']= results_PGA_Money['Value'].apply(lambda value:value.replace(',',''))

results_PGA_Money['Value']=results_PGA_Money['Value'].astype(float)
# Value is now float

results_PGA_Money.info()
# Get the Club Head Speed from statistic

results_PGA_CHS = PGA[PGA['Variable'] == 'Club Head Speed - (AVG.)']

results_PGA_CHS.head()
# Need to transform Value from results_PGA_CHS

results_PGA_CHS['Value']=results_PGA_CHS['Value'].astype(float)
# Want PGA Money by players from season 2010 to 2018

results_PGA_MoneyByPlayer = results_PGA_Money.groupby('Player Name').agg({'Value':'sum'})

results_PGA_MoneyByPlayer.head()
# Want the Club Head Speed by players from season 2010 to 2018

results_PGA_CHSMeanByPlayer = results_PGA_CHS.groupby('Player Name').agg({'Value':'mean'})
results_PGA_CHSMeanByPlayer.head()
# Want to merge results_PGA_MoneyByPlayer and results_PGA_CHSMeanByPlayer

results = results_PGA_MoneyByPlayer.merge(results_PGA_CHSMeanByPlayer, left_index=True, right_index= True)
results.head()
# Let's try to show if there is a real correlation between Club Head Speed and the Money ?

fig = sns.jointplot(x='Value_y', y ='Value_x', data = results,kind='reg', height =10, ratio=5, space=0.1)

fig.set_axis_labels('Club Head Speed - (AVG.)	','Official Money - (MONEY)')

fig.ax_marg_y.set_ylim(6000, 36313377)
#Intersting Figure player with CHS > 120

results_PGA_MoneyByPlayer[results_PGA_MoneyByPlayer['Value'] == 36313377]
#Intersting Figure playet with the fastest CHS

results_PGA_CHSMeanByPlayer[results_PGA_CHSMeanByPlayer['Value'] > 128]
# It's technology which allow players to send the ball very far...

results_PGA_CHSMeanBySeason = results_PGA_CHS.groupby('Season').agg({'Value':'mean'})

results_PGA_CHSMeanBySeason

#The Club Head Speed (mean) doesn't really vary along seasons...
# For the Smash Factor... it's almost the same

results_PGA_SamshFactor = PGA[PGA['Variable'] == 'Smash Factor - (AVG.)']

results_PGA_SamshFactor['Value']=results_PGA_SamshFactor['Value'].astype(float)

results_PGA_SmashFactorMeanBySeason = results_PGA_SamshFactor.groupby('Season').agg({'Value':'mean'})

results_PGA_SmashFactorMeanBySeason