#https://www.kaggle.com/themlphdstudent/premier-league-player-stats-data



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualization Libraries

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/premier-league-player-stats-data/Premier League Player Stats.csv')



#Calculation of Goal Percentage ,Shot on Target Percentage 

#and Minutes needed for one goal



df.loc[df['G'] > 0, 'OG%'] =  round(df['SOG']/df['SHOTS'],3)*100

df.loc[df['G'] > 0, 'G%'] =  round(df['G']/df['SHOTS'],3)*100

df.loc[df['G'] > 0, 'MP_Goal'] = round(df['MIN']/df['G'],1)



# For those who didn't score at all, the % will be set to zero.

df.loc[df['G'] <= 0, 'OG%'] =  0

df.loc[df['G'] <= 0, 'G%'] =  0

df.loc[df['G'] <= 0, 'MP_Goal'] = 0



print('Three new columns are added as below :\n')

df.head()





print('The top 5 player with highest On Goal Shots :\n')

df_OG = df.sort_values(by=['OG%'], ascending=False,kind='mergesort')

df_OG.head(5)



# From the result above, it can be seen that some players have little

# shot attempts and high percentage which is not very significant. 

# Therefore, I try to adjust this by adding threshold 

# that shot attempt >= 38 ,i.e. at least one shot per match



df_OG = df_OG[df_OG['SHOTS']>37]

df_OG.head(10)
print('The top 5 player with highest Goal/Shots Accurary:\n')

df_G = df[df['SHOTS']>37]

df_G = df_G.sort_values(by=['G%'], ascending=False,kind='mergesort')

df_G.head(5)



# This part will analyze the data by team level

#1. Number of players appeared for each team

#2. Number of players scored 

#3. Number of players assisted



#Count the number of players appeared

df_appear = df[(df.MIN > 0) ]

print(df_appear.shape)





df_appear = pd.DataFrame(df_appear['TEAM'].value_counts())

print(df_appear)



df_appear.plot.bar(color = 'yellow', figsize = (20, 7))



                                                    
# No. of players with at least 1 goal

df_G = df[(df.G > 0) ]

print(df_G.shape)



# There are 251 players having goals

df_G = pd.DataFrame(df_G['TEAM'].value_counts())

print(df_G)



df_G.plot.bar(color = 'orange', figsize = (20, 7))

# No. of players with at least 1 Assist

df_Asst = df[(df.ASST > 0) ]

print(df_Asst.shape)



# There are 253 players having assists

df_Asst = pd.DataFrame(df_Asst['TEAM'].value_counts())

print(df_Asst)



df_Asst.plot.bar(color = 'orange', figsize = (20, 7))