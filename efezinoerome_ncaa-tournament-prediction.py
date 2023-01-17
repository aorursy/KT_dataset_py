# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
temp = os.listdir("../input")
all_csv = {}
for i in range(0,len(temp)):
#for i in range(0,33):
    if(temp[i].split(".")[1] == "csv"):
        all_csv[temp[i].split(".")[0]] = pd.read_csv("../input/"+temp[i],encoding = 'ISO-8859-1')

print(all_csv.keys())
all_csv['Teams'].head()
print(all_csv['Teams'].info())
print(100*"*")
print(all_csv['Teams'].describe(include = ['O']))
print(100*"*")
print(all_csv['Teams'].describe(exclude =['O']))
print(100*"*")
print(all_csv['Teams'].isnull().sum())
all_csv['Seasons'].head()
print(all_csv['Seasons'].info())
print(100*"*")
print(all_csv['Seasons'].describe(include = ['O']))
print(100*"*")
print(all_csv['Seasons'].describe(exclude =['O']))
print(100*"*")
print(all_csv['Seasons'].isnull().sum())
print(all_csv['Seasons'].loc[:,'RegionW'].value_counts())
print(100*'*')
print(all_csv['Seasons'].loc[:,'RegionX'].value_counts())
print(100*'*')
print(all_csv['Seasons'].loc[:,'RegionY'].value_counts())
print(100*'*')
print(all_csv['Seasons'].loc[:,'RegionZ'].value_counts())
all_csv['NCAATourneySeeds'].head()
print(all_csv['NCAATourneySeeds'].info())
print(100*"*")
print(all_csv['NCAATourneySeeds'].describe(include = ['O']))
print(100*"*")
print(all_csv['NCAATourneySeeds'].describe(exclude =['O']))
print(100*"*")
print(all_csv['NCAATourneySeeds'].isnull().sum())
all_csv['RegularSeasonCompactResults'].head()
print(all_csv['RegularSeasonCompactResults'].info())
print(100*"*")
print(all_csv['RegularSeasonCompactResults'].describe(include = ['O']))
print(100*"*")
print(all_csv['RegularSeasonCompactResults'].describe(exclude =['O']))
print(100*"*")
print(all_csv['RegularSeasonCompactResults'].isnull().sum())
all_csv['RegularSeasonCompactResults'].loc[:,'WLoc'].value_counts()
all_csv['NCAATourneyCompactResults'].head()
print(all_csv['NCAATourneyCompactResults'].info())
print(100*"*")
print(all_csv['NCAATourneyCompactResults'].describe(include = ['O']))
print(100*"*")
print(all_csv['NCAATourneyCompactResults'].describe(exclude =['O']))
print(100*"*")
print(all_csv['NCAATourneyCompactResults'].isnull().sum())
print(all_csv['NCAATourneyCompactResults'].loc[:,'WLoc'].value_counts())
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
all_csv['Teams'].head()
fig1, ax1 = plt.subplots(2,2)
fig1.set_size_inches(10,10)
sns.countplot(y = 'FirstD1Season', data = all_csv['Teams'], ax = ax1[0][0])
sns.countplot(y = 'FirstD1Season', data = all_csv["Teams"].loc[all_csv['Teams']['FirstD1Season'] != 1985, ['FirstD1Season']], ax = ax1[0][1])
sns.countplot(y = 'LastD1Season', data = all_csv['Teams'], ax = ax1[1][0])
sns.countplot(y = 'LastD1Season', data = all_csv["Teams"].loc[all_csv['Teams']['LastD1Season'] !=2018, ['LastD1Season']], ax = ax1[1][1])
plt.tight_layout()
all_csv['Teams'][all_csv['Teams']['LastD1Season']==2011]
all_csv['Seasons'].head()
fig2, ax2 = plt.subplots(4,1)
fig2.set_size_inches(8,12)
#sns.countplot(x = 'RegionW', data = all_csv['Seasons'])
sns.countplot(x = 'RegionW', data = all_csv['Seasons'], ax = ax2[0])
sns.countplot(x = 'RegionX', data = all_csv['Seasons'], ax = ax2[1])
sns.countplot(x = 'RegionY', data = all_csv['Seasons'], ax = ax2[2])
sns.countplot(x = 'RegionZ', data = all_csv['Seasons'], ax = ax2[3])
plt.tight_layout()
all_csv['NCAATourneySeeds'].head()
all_csv['NCAATourneySeeds']['Seed_NoReg'] = [i[1:] for i in all_csv['NCAATourneySeeds'].Seed]
all_csv['NCAATourneySeeds'].head()
temp = pd.merge(all_csv['NCAATourneySeeds'],all_csv['Teams'].loc[:,['TeamID','TeamName']],on = 'TeamID')
temp1 = temp.groupby(['TeamName','Seed_NoReg'])['Seed_NoReg'].count()
grid_kws = {"height_ratios": (.95, .01), "hspace": .03}
fig3, (ax3,cbar3) = plt.subplots(2, gridspec_kw=grid_kws)
fig3.set_size_inches((50,200))
ax3.yaxis.label.set_size(60)
ax3.xaxis.label.set_size(60)
ax3.tick_params(labelsize=35)
ax = sns.heatmap(data = temp1.unstack().fillna(0), ax = ax3, cbar_ax=cbar3, linewidth=2,vmax = 14,\
                  cbar_kws={"orientation": "horizontal","ticks":[0,14]})
cbar3.tick_params(labelsize=40)
cbar3.set_title(label = 'Seed Rating',fontsize=50,loc = 'left')
plt.show()
all_csv['RegularSeasonCompactResults']['WMargin'] = all_csv['RegularSeasonCompactResults']["WScore"] - \
all_csv['RegularSeasonCompactResults']['LScore']
all_csv['RegularSeasonCompactResults'].head()
fig5, ax5 = plt.subplots(1,3)
fig5.set_size_inches((12,4))
sns.distplot(all_csv['RegularSeasonCompactResults'].WScore,label = "Winning Team",\
            ax = ax5[0],hist = True, kde = True)
sns.distplot(all_csv['RegularSeasonCompactResults'].LScore,label = "Losing Team",\
            ax = ax5[0], hist = True, kde = True)
sns.distplot(all_csv['RegularSeasonCompactResults'].WMargin,label = "Score Margin",\
            ax = ax5[0], hist = True, kde = True)
sns.regplot(x = 'Season',y = 'WScore', ax = ax5[1],fit_reg = False,label = 'Winning Team',\
          data = all_csv['RegularSeasonCompactResults'].groupby(["Season"])['WScore'].mean().reset_index())
sns.regplot(x = 'Season',y = 'LScore', ax = ax5[1],fit_reg = False,label = 'Losing Team',\
          data = all_csv['RegularSeasonCompactResults'].groupby(["Season"])['LScore'].mean().reset_index())
sns.regplot(x = 'Season',y = 'WMargin', ax = ax5[2],fit_reg = False,label = 'Score Margin',\
          data = all_csv['RegularSeasonCompactResults'].groupby(["Season"])['WMargin'].mean().reset_index())
#sns.boxplot(x="Season", y="Wscore", ax = ax5[2], \
#            data = all_csv['RegularSeasonCompactResults'].groupby(["Season"])['WScore','LScore'].mean().reset_index())
plt.tight_layout()
ax5[0].set_xlabel('Score')
ax5[1].set_ylabel('Yearly Average Score')
ax5[2].set_ylabel('Winning Team Margin - Yearly Average')
ax5[0].legend()
ax5[1].legend()
ax5[2].legend()
ax5[2].set_ylim([9,15])
fig6, ax6 = plt.subplots(1,1)
fig6.set_size_inches((6,4))
sns.boxplot(x = 'WLoc', y = 'WMargin',data = all_csv['RegularSeasonCompactResults'],ax = ax6)
sns.lmplot(x = 'WScore', y = 'LScore',hue = 'WLoc', fit_reg = False,\
              data = all_csv['RegularSeasonCompactResults'])
plt.tight_layout()
ax6.set_ylabel('Winning Team Margin')
all_csv['Teams'][['TeamID','TeamName']].head()
temp4 = all_csv['RegularSeasonCompactResults'].merge(all_csv['Teams'][['TeamID','TeamName']],left_on = 'WTeamID',right_on = 'TeamID')
temp4.drop(['TeamID'],axis = 1,inplace = True)
temp4.head()
grid_kws = {"height_ratios": (.95, .01), "hspace": .03}
fig7, (ax7,cbar7) = plt.subplots(2, gridspec_kw=grid_kws)
fig7.set_size_inches((25,150))
ax7.yaxis.label.set_size(25)
ax7.xaxis.label.set_size(25)
ax7.tick_params(labelsize=15)
ax = sns.heatmap(data = temp4.groupby(['Season','TeamName'])['WMargin'].mean().unstack().fillna(0).transpose(), ax = ax7, cbar_ax=cbar7, linewidth=2,\
                  cbar_kws={"orientation": "horizontal"})
cbar7.tick_params(labelsize=15)
cbar7.set_title(label = 'Winning Team Margin',fontsize=30,loc = 'left')
plt.show()
all_csv['NCAATourneyCompactResults']['WMargin'] = all_csv['NCAATourneyCompactResults']["WScore"] - \
all_csv['NCAATourneyCompactResults']['LScore']
all_csv['NCAATourneyCompactResults'].head()
fig8, ax8 = plt.subplots(1,3)
fig8.set_size_inches((12,4))
sns.distplot(all_csv['NCAATourneyCompactResults'].WScore,label = "Winning Team",\
            ax = ax8[0],hist = True, kde = True)
sns.distplot(all_csv['NCAATourneyCompactResults'].LScore,label = "Losing Team",\
            ax = ax8[0], hist = True, kde = True)
sns.distplot(all_csv['NCAATourneyCompactResults'].WMargin,label = "Score Margin",\
            ax = ax8[0], hist = True, kde = True)
sns.regplot(x = 'Season',y = 'WScore', ax = ax8[1],fit_reg = False,label = 'Winning Team',\
          data = all_csv['NCAATourneyCompactResults'].groupby(["Season"])['WScore'].mean().reset_index())
sns.regplot(x = 'Season',y = 'LScore', ax = ax8[1],fit_reg = False,label = 'Losing Team',\
          data = all_csv['NCAATourneyCompactResults'].groupby(["Season"])['LScore'].mean().reset_index())
sns.regplot(x = 'Season',y = 'WMargin', ax = ax8[2],fit_reg = False,label = 'Score Margin',\
          data = all_csv['NCAATourneyCompactResults'].groupby(["Season"])['WMargin'].mean().reset_index())
#sns.boxplot(x="Season", y="Wscore", ax = ax5[2], \
#            data = all_csv['RegularSeasonCompactResults'].groupby(["Season"])['WScore','LScore'].mean().reset_index())
plt.tight_layout()
ax8[0].set_xlabel('Score')
ax8[1].set_ylabel('Yearly Average Score')
ax8[2].set_ylabel('Winning Team Margin - Yearly Average')
ax8[0].legend()
ax8[1].legend()
ax8[2].legend()
ax8[2].set_ylim([7,18])
fig9, ax9 = plt.subplots(1,1)
fig9.set_size_inches((6,4))
sns.boxplot(x = 'WLoc', y = 'WMargin',data = all_csv['NCAATourneyCompactResults'],ax = ax9)
sns.lmplot(x = 'WScore', y = 'LScore',hue = 'WLoc', fit_reg = False,\
              data = all_csv['NCAATourneyCompactResults'])
plt.tight_layout()
ax9.set_ylabel('Winning Team Margin')
temp5 = all_csv['NCAATourneyCompactResults'].merge(all_csv['Teams'][['TeamID','TeamName']],left_on = 'WTeamID',right_on = 'TeamID')
temp5.drop(['TeamID'],axis = 1,inplace = True)
temp5.head()
grid_kws = {"height_ratios": (.95, .01), "hspace": .03}
fig10, (ax10,cbar10) = plt.subplots(2, gridspec_kw=grid_kws)
fig10.set_size_inches((25,150))
ax10.yaxis.label.set_size(25)
ax10.xaxis.label.set_size(25)
ax10.tick_params(labelsize=15)
ax = sns.heatmap(data = temp5.groupby(['Season','TeamName'])['WMargin'].mean().unstack().fillna(0).transpose(), ax = ax10, cbar_ax=cbar10, linewidth=2,\
                  cbar_kws={"orientation": "horizontal"})
cbar10.tick_params(labelsize=15)
cbar10.set_title(label = 'Winning Team Margin',fontsize=30,loc = 'left')
plt.show()



















