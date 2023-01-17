# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
ipl_match_data=pd.read_csv('/kaggle/input/ipldata/matches.csv')

ipl_match_data =ipl_match_data[['id', 'season', 'city', 'date', 'team1', 'team2', 'result', 'winner', 'venue']]



#shortening the name

df=ipl_match_data



#shortening the name for further use

df=ipl_match_data



#converting dates for sorting the data

df['date'] = pd.to_datetime(df['date'])

df.sort_values(by='date',inplace=True)





#replacing Delhi Capitals with Delhi Daredevils to match the entire data

df.replace("Delhi Capitals","Delhi Daredevils",inplace=True)

df.replace("Feroz Shah Kotla Ground","Feroz Shah Kotla",inplace=True)



#Keeping records where Delhi Daredevils is included

#Used subset as team1 so as to protect any nan data of our team

df=df.where((df['team1']=="Delhi Daredevils") | (df['team2']=="Delhi Daredevils")).dropna(subset=['team1'])



#replacing nan values to make data look clean

df['winner'] = df['winner'].replace(np.nan, 'No Winner')

df['city'] = df['city'].replace(np.nan, 'Abroad')



#df['player_of_match'] = df['player_of_match'].replace(np.nan, 'N/A')  ----> removed the column so not needed



#setting home ground as Feroz Shah Kotla as it is the only ground in delhi where IPL matches have been played

Homeground= 'Feroz Shah Kotla'



#added homeground, win and homeground win columns to make plots

df['homeground']=["YES" if (x==Homeground) else "NO" for x in df['venue']]

df['wins']=["YES" if (x=="Delhi Daredevils") else "NO" for x in df['winner']]

df['homeground_win']=np.where((df['homeground']=="YES") & (df['wins']=="YES"), "YES", "NO")



#needed to calculate homeground only matches win %

df_edited=df.where(df['homeground']=="YES").dropna()



#delted irrelevant column

df.drop(['venue','winner'],axis=1,inplace=True)



homeground_percent = df['homeground'].value_counts("YES")["YES"]*100

homeground_win_percent = (df_edited['homeground_win'].value_counts("YES")["YES"])*100

win_percent = df['wins'].value_counts("YES")["YES"]*100



#some debugging stuff

'''

print(df['homeground'].value_counts()["YES"])

print(df['wins'].value_counts()["YES"])

print(df['homeground_win'].value_counts()["YES"])

print(df['homeground_win'].value_counts()["NO"])

'''



#I want to create a plot for my assignment submission

#where i need to show delhi daredevils performance in its homeground

#please help me plot it in a better way

#this is my verison 



##Thanks##



text="Delhi Daredevils : IPL 2008-2019"

plt.figure(figsize=(20,5))

plt.suptitle(text,x=0.51,y=1.05,size=16,weight='bold',fontstyle='italic')



plt.subplot(1,3,1)

sns.swarmplot(y=df['season'],x=df['wins'])

plt.yticks(df['season'].unique());

plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)

plt.ylabel('Season',size=12)

plt.xlabel("Wins",size=12)

plt.legend(("YES","NO"),loc="center", frameon=False,bbox_to_anchor=(0, 0.85, 1, .1));

y_text = plt.text(0.5, 1.05, 'Wins = %.2f%%' % (win_percent),horizontalalignment='center',verticalalignment='center', transform=plt.gca().transAxes, bbox=dict(fc='turquoise',ec='blue'));



plt.subplot(1,3,2)

sns.swarmplot(y=df_edited['season'],x=df_edited['homeground_win'])

plt.yticks(df['season'].unique());

plt.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)

plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)

plt.ylabel("")

plt.xlabel("Homeground Wins",size=12)

plt.legend(("YES","NO"),loc="center", frameon=False,bbox_to_anchor=(0, 0.85, 1, .1));

y_text = plt.text(0.5, 1.05, 'Homeground Wins = %.2f%%' % (homeground_win_percent),horizontalalignment='center',verticalalignment='center', transform=plt.gca().transAxes, bbox=dict(fc='turquoise',ec='blue'));



plt.subplot(1,3,3)

sns.swarmplot(y=df['season'],x=df['homeground'])

plt.yticks(df['season'].unique());

plt.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)

plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)

plt.ylabel("")

plt.xlabel("Homeground",size=12)

plt.legend(("YES","NO"),loc="center", frameon=False,bbox_to_anchor=(0, 0.85, 1, .1));

y_text = plt.text(0.5, 1.05, 'Homeground Matches = %.2f%%' % (homeground_percent),horizontalalignment='center',verticalalignment='center', transform=plt.gca().transAxes, bbox=dict(fc='turquoise',ec='blue'));
