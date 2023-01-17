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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly_express as pxs

#Reading the dataset

df=pd.read_csv("../input/ipl-2008-to-2019/ball_by_ball_details.csv",low_memory=False)
#Printing the the first five rows the dataset 

df.head()

# slicing the dataset to get only data for 2018 and 2019 season 

df_1=df.loc[df.season>2017]
df_1
df_1['ov_c']=pd.cut(df_1.over,[0,6,17,20],labels=["Powerplay", "Middle_overs", "Death_overs"])
df_1.shape
#Slicing the dataset to find out no of wicket each player has taken in Powerplay

df_2=df_1.loc[(df_1.ov_c=="Powerplay")&(df_1.player_out.notnull())&(df_1.kind !='run out')&(df_1.kind !='retired hurt' )]

df_2[df.season==2018].bowler.value_counts()

df_2[df.season==2019].bowler.value_counts()
df_2.bowler.value_counts()
df_3=pd.DataFrame(df_2.groupby(['season','bowler']).bowler.count())

df_3.columns=['no_w']

df_3.reset_index(inplace=True)
df_3.season=df_3.astype({'season': 'str'})
df_3.dtypes


pxs.bar(df_3,'bowler','no_w',color='season',width=1800,height=600)

#displaying every bowlers 

df.bowler.unique()
#slicing the bowling data of Deepak Chahar

dc=df.loc[(df.bowler=='DL Chahar')]
#Categorising overs into Poweplay,Middleovers,Deathovers

dc['ov_c']=pd.cut(dc.over,[0,6,17,20],labels=["Powerplay", "Middle_overs", "Death_overs"])
dc.over.value_counts()
#Groupby dc with season and ov_c so that you can find the no of ball bowled in each season and part of the match(Powerplay,middleovers,deathovers)

dc_s=pd.DataFrame(dc.groupby(['season','ov_c']).ov_c.count())

dc_s.columns=['no']

dc_s.reset_index(inplace=True)

#plotting the above data

sns.barplot('season','no',hue='ov_c',data=dc_s)

plt.ylabel("NO of balls bowled")

fig=plt.gcf()

fig.set_size_inches(15,7)
# This gives the percentage of balls Deppak Chahar had bowled in each part of the match(Poweplay,Middle,Death_overs)

(dc.ov_c.value_counts(normalize=True)*100)
#ploting the above 

(dc.ov_c.value_counts(normalize=True)*100).plot(kind="bar")

plt.title("Deppak chahars overs categorisation")

plt.ylabel("(Percentage)No of balls bowled")

fig=plt.gcf()

fig.set_size_inches(15,7)
dc.ov_c.value_counts()
#To calculate the srikerate of chahar in each part of the match

dc.ov_c.value_counts()/dc.loc[(dc.player_out.notnull())&(dc.kind !='run out')].ov_c.value_counts()
#ploting the above(strikerate)

dc_o=pd.DataFrame(dc.ov_c.value_counts()/dc.loc[(dc.player_out.notnull())&(dc.kind !='run out')].ov_c.value_counts())

dc_o.drop(['Middle_overs'],inplace=True)

dc_o.plot(kind='bar')

plt.ylabel("Strike rate")
#No of wickets DC has taken in powerplay,death_overs,middleovers

dc.loc[(dc.player_out.notnull())&(dc.kind !='run out')].ov_c.value_counts()
#plotting the percentage of the wickets taken in each part of the match

ofd=dc.loc[(dc.player_out.notnull())&(dc.kind !='run out')].ov_c.value_counts(normalize=True)*100

ofd.plot(kind="bar")



plt.ylabel("Percentage(No of wickets taken)")

plt.title("Percentage of the wickets taken in each part of the match")
