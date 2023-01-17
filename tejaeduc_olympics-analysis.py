import pandas as pd
import numpy as np
import seaborn as sns
%matplotlib inline
import pylab as plt
sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(16,12)})
df = pd.read_csv("../input/athlete_events.csv")
len(df)
df.head()
df.info()
print("Number of Nationalities -->",len(df.NOC.unique()))
print("Number of Teams -->",len(df.Team.unique()))
df.Team.value_counts().tail() #Look at the odd Team names
df.isnull().sum()
df.Medal.value_counts() 
print("total number of Unique players participated in Olympics are -->",len(df.ID.unique()))
print("SUMMER OLYMPICS WERE CONDUCTED IN ",np.array(sorted(df[df['Season'] == 'Summer']['Year'].unique())))
print("WINTER OLYMPICS WERE CONDUCTED IN ",np.array(sorted(df[df['Season'] == 'Winter']['Year'].unique())))
groupedYearID = df.groupby(['Year','ID'],as_index=False).count()[['Year','ID']]
groupedYearID = groupedYearID.groupby('Year',as_index=False).count()
groupedYearID.head()
l = []
for i in [1994,1998,2002,2006,2010,2014]: #The year of winter olympics
    l.append(groupedYearID[groupedYearID.Year == i].index[0])
for i in l:
    groupedYearID.loc[i,'Year'] = groupedYearID.loc[i,'Year'] +2
groupedYearID = groupedYearID.groupby('Year',as_index=False).sum()
import matplotlib.pyplot as pyplot
sns.set(rc={'figure.figsize':(18,12)})
plot1 = sns.barplot('Year','ID',data=groupedYearID).set_xticklabels(groupedYearID.Year,rotation=82)
#plot1.set(xlabel='YEAR',ylabel='Number of people')
pyplot.xlabel("YEAR")
pyplot.ylabel("PARTICIPANTS")
groupedGender = pd.concat([df,pd.get_dummies(df.Sex)],axis=1).groupby(['Year','ID'],as_index = False).sum()
groupedGender[['Year','ID','F','M']].head()
groupedGender.F = groupedGender.F.apply(lambda x: 0 if x==0 else 1)
groupedGender.M = groupedGender.M.apply(lambda x: 0 if x==0 else 1)
groupedGender[['Year','ID','F','M']].head()
groupedGender = groupedGender.groupby('Year',as_index=False).sum()
# same code as mentioned above some where, the years of winter olympics after 1994 to be clubbed to their next summer olympics.
for i in l:
    groupedGender.loc[i,'Year'] = groupedGender.loc[i,'Year'] +2
groupedGender = groupedGender.groupby('Year',as_index=False).sum()
plt.plot(groupedGender.Year,groupedGender.M)
plt.plot(groupedGender.Year,groupedGender.F,color='red')

plt.plot(groupedGender.Year,groupedGender.M,'bo')
plt.plot(groupedGender.Year,groupedGender.F,'bo',color ='red')

plt.legend(['Male','Female'])
plt.xlabel("YEAR")
plt.ylabel("PARTICIPANTS")
df = pd.concat([df,pd.get_dummies(df.Medal)],axis=1)
df['allmedals'] = df['allmedals'] = df['Bronze'] + df['Gold'] + df['Silver'] 
#Obviously it would be either 1 or 0. Added this column to make analysis easier
df.head()
groupcountry = df.groupby(by=['NOC'],as_index= False).sum()
top50 = groupcountry.sort_values(by=['allmedals'],ascending = False).head(50)
plot2 = sns.barplot('NOC','allmedals',data=top50).set_xticklabels(top50.NOC,rotation=82)
groupYearNOC = df.groupby(by=['Year','NOC'],as_index=False).sum()
l1 = []
for i in [1994,1998,2002,2006,2010,2014]: #The year of winter olympics
    l1.append(np.array(groupYearNOC[groupYearNOC.Year == i].index))
for i in l1:
    groupYearNOC.loc[i,'Year'] = groupYearNOC.loc[i,'Year'] +2
groupYearNOC = groupYearNOC.groupby(by=['Year','NOC'],as_index=False).sum()
yeartop = pd.DataFrame() 
y = df.Year.unique() #Gets the Year numbers
for i in y:
    yeartop = pd.concat([yeartop,groupYearNOC[groupYearNOC['Year'] == i].sort_values(by=['allmedals'],ascending= False).head(1)])
    
import pylab as plt
fig, ax = plt.subplots()

sns.barplot('Year','allmedals',hue='NOC',data = yeartop,ax=ax)

# CODE COPIED FROM STACK OVERFLOW: THE CODE FOR CHANGING THE BAR'S WIDTH IN BARPLOT
def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

change_width(ax, .50)
plt.show()
groupsport = df.groupby(by=['Sport','NOC'],as_index=False).sum()
spotop = pd.DataFrame()
sp = df.Sport.unique()
for i in sp:
    spotop = pd.concat([spotop,groupsport[groupsport['Sport'] == i].sort_values(by=['allmedals'],ascending= False).head(1)])

spotop = spotop[['Sport','NOC','Bronze','Gold','Silver','allmedals']]
spotop
set(spotop[spotop['NOC'] == 'USA']['Sport'])
spotop[spotop['Sport']=='Baseball']
df.Sport.unique()
dfage = df.groupby(['Year','ID'],as_index=False).mean()
dfage = dfage[np.isfinite(dfage['Age'])]
dfage = dfage.groupby('Year',as_index= False).mean()
plt.plot(dfage.Year,dfage.Age)
plt.xticks(range(1896,2024,4))
plt.show()
dfagesport = df.groupby(['Sport','ID'],as_index=False).mean()
dfagesport = dfagesport.groupby(['Sport'],as_index=False).mean()
plot4 = sns.barplot('Sport','Age',data = dfagesport.sort_values('Age')).set_xticklabels(dfagesport.sort_values('Age').Sport,rotation=82)
dfheight = dfagesport[np.isfinite(dfagesport['Height'])]
plot4 = sns.barplot('Sport','Height',data = dfheight.sort_values('Height')).set_xticklabels(dfheight.sort_values('Height').Sport,rotation=82)
dfweight = dfagesport[np.isfinite(dfagesport['Weight'])]
plot4 = sns.barplot('Sport','Weight',data = dfweight.sort_values('Weight')).set_xticklabels(dfweight.sort_values('Weight').Sport,rotation=82)
df.Name.value_counts()[:15]
df[(df.Name == 'Robert Tait McKenzie') & (df.allmedals == 1)]
medals = df.groupby('Name',as_index=False).sum()
mostmed = medals.sort_values(by=['allmedals'],ascending=False)
mostmed.head(10)[['Name','Bronze','Gold','Silver','allmedals']]
print("Number of players with more than 10 Medals", len(mostmed[mostmed.allmedals >= 10]))
mostgold = medals.sort_values(by=['Gold'],ascending=False)
mostgold.head(10)[['Name','Gold']]
mostmed[(mostmed.Silver) > (mostmed.Gold + 3)][['Name','Bronze','Gold','Silver','allmedals']]
print("Number of people who won Medal but no Gold -->",len(mostmed[((mostmed.Silver) +( mostmed.Bronze) > 1) & (mostmed.Gold == 0)]))
mostmed[((mostmed.Silver) +( mostmed.Bronze) >= 5 ) & (mostmed.Gold == 0)][['Name','Bronze','Gold','Silver','allmedals']]
df[df.Name ==  'Franziska van Almsick'].head(1)[['Name','Sex','Age','Team','Sport']]
df[df.Age ==df.Age.min()]
df[df.Age ==df.Age.max()]
dfnoart = df[df.Sport !='Art Competitions']
dfnoart[dfnoart.Age ==dfnoart.Age.max()]
df['countnum'] = 1
succgrp = df.groupby('Name',as_index=False).sum()
succgrp['rate']  = succgrp['allmedals']/succgrp['countnum']
succgrp =succgrp.sort_values('rate',ascending = False)
succgrp1 = succgrp[succgrp['countnum'] > 5]
set(succgrp1[succgrp1['rate'] == 1.0]['Name'])
succgrp2 = succgrp[succgrp['countnum'] > 4]
for i in range(len(succgrp2)):
    if (succgrp2.iloc[i].Gold ==  succgrp2.iloc[i]['countnum']):
        print(succgrp2.iloc[i].Name)
dfindia = df[df.NOC == 'IND']
sorted(dfindia.Year.unique())
print("Total number of all Medals India won", dfindia['allmedals'].sum())
dfindia.groupby(['Year','Event'],as_index=False).max()['allmedals'].sum()
dfindia[(dfindia.Year== 1924) & (dfindia.Gold ==1)]
set(dfindia[dfindia.allmedals == 1].Sport.unique())
dfindia[dfindia.Gold == 1].Sport.unique()
dfindia[(dfindia.Gold==1) & (dfindia.Sport == 'Shooting')]
dfindia[(dfindia.Gold==1) & (dfindia.Sport == 'Hockey')]['Year'].unique()
dfindyear = dfindia.groupby(['Year','Event'],as_index=False).max()
dfindyear = dfindyear.groupby(['Year'],as_index=False).sum()
plt.plot(dfindyear.Year,dfindyear.allmedals)
plt.plot(dfindyear.Year,dfindyear.allmedals,'bo')
plt.yticks(range(0,8))
plt.xticks(range(1900,2018,4))
plt.show()