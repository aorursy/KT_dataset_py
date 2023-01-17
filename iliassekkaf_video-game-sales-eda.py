import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import itertools

%matplotlib inline

plt.style.use('fivethirtyeight')



#Imported the libraries
games=pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')
games.head() #Just checking what vaariables are available for use
games.info() #Cheking Data types , missing data etc...
#Year_of_Release is a float, that's Ugly, let's make it int.

games=games[games['Year_of_Release'].notnull()] #remove null years

games['Year_of_Release']=games['Year_of_Release'].apply(int)
plt.figure(figsize=(12,6))

games['Platform'].value_counts(sort=True,ascending=True).plot.bar()

plt.title('Game Releases by Platform')

plt.ylabel('Nb of Games')

plt.xlabel('Platform')
drop_year=games['Year_of_Release'].value_counts()[games['Year_of_Release'].value_counts()<50].index
for i,v in enumerate(drop_year):

    games=games[games['Year_of_Release'] != v ]
byYear=games.groupby('Year_of_Release').sum()
byYear.plot.bar(x=byYear.index,y='Global_Sales',figsize=(12,6))

plt.title('Global Games sales per Year in Million Units')

plt.ylabel('Sales in Millions')
JPN=games.groupby('Year_of_Release').sum()['JP_Sales']

NA=games.groupby('Year_of_Release').sum()['NA_Sales']

EU=games.groupby('Year_of_Release').sum()['EU_Sales']

Other=games.groupby('Year_of_Release').sum()['Other_Sales']



for df,(i,j) in zip([JPN,NA,EU,Other],itertools.product(range(2),range(2))):

    ax1=plt.subplot2grid((2,2),(i,j))

    plt.subplots_adjust(hspace=0.8)

    df.plot.bar(ax=ax1,figsize=(20,8))

    ax1.set_title(df.name)
byPlatform=games.groupby('Platform').sum()
byPlatform.head()
byPlatform.reset_index().sort_values(by='Global_Sales')[['Global_Sales','Platform']].plot(kind='bar',x='Platform',figsize=(12,6))

plt.title('Global Sales By Platform in Millions')
byPlatform.drop(['Critic_Score','Critic_Count','User_Count'],axis=1,inplace=True)
games['Year_of_Release']=pd.to_datetime(games['Year_of_Release'],format='%Y').dt.year #Convert int to Datetime

df=games[['Platform','Year_of_Release']].groupby('Platform').max()-games[['Platform','Year_of_Release']].groupby('Platform').min()
df.columns=['Age']



byPlatform.drop(['Year_of_Release'],axis=1,inplace=True) #we don't need that anymore all we need is the "Age" of the console.
byPlatform=pd.DataFrame(byPlatform).join(df)
byPlatform['Sales per Year']=byPlatform['Global_Sales']/byPlatform['Age']



byPlatform.head()
plt.figure(figsize=(12,6))

byPlatform['Sales per Year'].sort_values().plot.bar()

plt.title('Mean Sales per Year by Platform')
Gen=games.groupby(['Platform','Year_of_Release'])['Global_Sales'].sum()
#Just checking my data

Gen[['PS4','XOne']]
Gen6=pd.DataFrame(Gen[['GC','PS2','XB']])

Gen6=Gen6.reset_index().pivot('Year_of_Release','Platform','Global_Sales')

Gen7=pd.DataFrame(Gen[['PS3','Wii','X360']])

Gen7=Gen7.reset_index().pivot('Year_of_Release','Platform','Global_Sales')

Gen8=pd.DataFrame(Gen[['PS4','XOne','WiiU']])

Gen8=Gen8.reset_index().pivot('Year_of_Release','Platform','Global_Sales')
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,4),sharey=True)

Gen6.plot(ax=ax1,color=['#FC4F30','#30A2DA','#E5AE38'] )

Gen7.plot(ax=ax2 )

Gen8.plot(ax=ax3 )

for ax,df in zip([ax1,ax2,ax3],['Gen6','Gen7','Gen8']):

    ax.legend(frameon=False,bbox_to_anchor=[0,0,1.1,1.05])

    ax.set_title('Sales by Year for : \n'+ df +' Consoles',fontdict=dict(fontsize=15))
Gen_Release=games.groupby(['Platform','Year_of_Release'])['Name'].count()
Gen_Release

Gen6=pd.DataFrame(Gen_Release[['GC','PS2','XB']])

Gen6=Gen6.reset_index().pivot('Year_of_Release','Platform','Name')

Gen7=pd.DataFrame(Gen_Release[['PS3','Wii','X360']])

Gen7=Gen7.reset_index().pivot('Year_of_Release','Platform','Name')

Gen8=pd.DataFrame(Gen_Release[['PS4','XOne','WiiU']])

Gen8=Gen8.reset_index().pivot('Year_of_Release','Platform','Name')
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,4),sharey=True)

Gen6.plot(ax=ax1,color=['#FC4F30','#30A2DA','#E5AE38'] )

Gen7.plot(ax=ax2 )

Gen8.plot(ax=ax3 )

for ax,df in zip([ax1,ax2,ax3],['Gen6','Gen7','Gen8']):

    ax.legend(frameon=False,bbox_to_anchor=[0,0,1.1,1.05])

    ax.set_title('Releases by Year for : \n'+ df +' Consoles',fontdict=dict(fontsize=15))

    ax.set_ylabel('Number of Releases')
Sales_Release=pd.DataFrame(Gen).join(Gen_Release)
Sales_Release['Sales_per_Relase']=Sales_Release['Global_Sales']/Sales_Release['Name'];

Gen6=pd.DataFrame(Sales_Release.loc[['GC','PS2','XB']])

Gen6=Gen6.reset_index().pivot('Year_of_Release','Platform','Sales_per_Relase')

Gen7=pd.DataFrame(Sales_Release.loc[['PS3','Wii','X360']])

Gen7=Gen7.reset_index().pivot('Year_of_Release','Platform','Sales_per_Relase')

Gen8=pd.DataFrame(Sales_Release.loc[['PS4','XOne','WiiU']])

Gen8=Gen8.reset_index().pivot('Year_of_Release','Platform','Sales_per_Relase')
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,4),sharey=True)

Gen6.plot(ax=ax1,color=['#FC4F30','#30A2DA','#E5AE38'] )

Gen7.plot(ax=ax2 )

Gen8.plot(ax=ax3 )

for ax,df in zip([ax1,ax2,ax3],['Gen6','Gen7','Gen8']):

    ax.legend(frameon=False,bbox_to_anchor=[0,0,1.1,1.05])

    ax.set_title('Sales per Game Realeased for: \n'+ df +' Consoles',fontdict=dict(fontsize=15))

    ax.set_ylabel('Sales in Mlillion units')
games.head()
games['Publisher'].value_counts().sort_values().head()
byPub=games.groupby(['Publisher']).count().sort_values('Name',ascending=False)

byPub=byPub[byPub['Name']>50]
plt.figure(figsize=(12,6))

byPub['Name'].sort_values(ascending=True).plot.bar()

plt.title('Games Releases by Publisher')
games.head()
byPlatform=games.groupby(['Publisher'])['Name'].count()

byPubSales=games.groupby(['Publisher'])[['Global_Sales','NA_Sales','JP_Sales','EU_Sales','Other_Sales']].sum()
byPub=pd.DataFrame(byPlatform).join(byPubSales)
byPub.head()
byPub['Sales per Release']=byPub['Global_Sales']/byPub['Name']
byPub.nlargest(n=60,columns='Sales per Release').plot.bar(y='Sales per Release',figsize=(15,6))

plt.legend(frameon=False)

plt.title('Mean Sales per Game Released: \n All Publishers')
byPub[byPub['Name']>50].nlargest(n=60,columns='Sales per Release').plot.bar(y='Sales per Release',figsize=(15,6))

plt.legend(frameon=False)

plt.title('Sales per Game Released :  \n Publishers with at Least 50 Games')
byReg=games.groupby(['Genre']).sum().drop(['Critic_Score','Critic_Count','User_Count','Year_of_Release','Global_Sales'],axis=1)
byReg
byReg.plot.bar(stacked=True,figsize=(15,8))

plt.title('Game Sales by Region and Genre')
Index=['Action',

 'Adventure',

 'Fighting',

 'Misc',

 'Platform',

 'Puzzle',

 'Racing',

 'Role-Playing',

 'Shooter',

 'Simulation',

 'Sports',

 'Strategy',

 'Total']
byReg=byReg.append(byReg.sum(),ignore_index=True)

byReg.index=Index
byReg_Pct=pd.DataFrame([]) #Create empty df

for col in byReg.columns:  #Calculating the percentages for each Col alone(as Series)

    byReg_Pct[col]=byReg[col].apply(lambda x:(x/byReg[col]['Total'])*100) 
byReg_Pct.stack().unstack(level=0).drop('Total',axis=1).plot.bar(stacked=True,figsize=(15,8),cmap='Paired')

plt.legend(frameon=False,bbox_to_anchor=[0,0,1.1,0.95])

plt.title('Sales by Region and Genre ( in %)')
Major_Genre=['Action','Role-Playing','Shooter','Sports']



byPub=games.groupby(['Publisher','Genre']).count().sort_values('Name',ascending=False)

byPub=pd.DataFrame(byPub[byPub['Name']>50]['Name'])



byPub.head()



plt.figure(figsize=(30,25))



for i,genre,j in zip(itertools.product(range(3),range(4)),Major_Genre,range(4)):

    Genre=byPub.xs(genre,level=1).nlargest(columns='Name',n=10)

    ax1=plt.subplot2grid((3,4),i)

    for i,v in enumerate(Genre.index):

        Genre.index.values[i]=Genre.index[i].split()[0]

    ax1.pie(x=Genre['Name'],autopct='%1.1f%%',labels=Genre.index,labeldistance=1.05,pctdistance=0.7,textprops=dict(fontsize=15))

    plt.title('Most Release by Publisher for : \n' + Major_Genre[j])
games.head()
byGenre=games.groupby(['Genre','Platform'])['Name'].count()

byGenre=pd.DataFrame(byGenre)

byGenre.head()

byGenre.reset_index(level=0).head()



byPlatform=games.groupby(['Genre','Platform'])['Name'].count()

plt.figure(figsize=(30,25))

for i,genre,j in zip(itertools.product(range(3),range(4)),Major_Genre ,range(4)):

    genre=byGenre.xs(genre).nlargest(columns='Name',n=10)

    ax1=plt.subplot2grid((3,4),(i))

    ax1.pie(genre['Name'],autopct='%1.1f%%',labels=genre.index,textprops=dict(fontsize=15,color='k'),pctdistance=0.75)

    plt.title('Most release by Platform for : \n' +Major_Genre[j])
