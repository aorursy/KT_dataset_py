import math
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))
#import data
path = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv', index_col=0)
datas = pd.DataFrame(path)
simply_name = {'Sony Computer Entertainment':'Sony',
    'Konami Digital Entertainment': 'Konami',
    'Namco Bandai Games': 'Namco Bandai',
    'Nippon Ichi Software': 'Nippon Ichi'}
datas.User_Score.map(lambda p:  'NaN' if p == 'tbd'else p)
datas.loc[datas['User_Score'] == 'tbd', 'User_Score'] = 'NaN'
data_7 = datas.loc[:, ['Year_of_Release','Genre', 'Publisher','Global_Sales']].dropna().reset_index().groupby(['Name', 'Genre']).apply(lambda df: df.loc[df.Year_of_Release.idxmin()])
# the following line makes the background of normal plots full with grey grids
# just like seaborn plots, I am not sure why this works but I like it.
sns.set(rc={'figure.figsize':(16.7,8.27)})
data_3 = datas.loc[:, ['NA_Sales','EU_Sales','JP_Sales','Other_Sales']]
plot_3 = data_3.head(20).plot.bar(
    figsize=(16.18, 8),
    fontsize=14,
    stacked=True,
)
plot_3.set_xlabel('')
plot_3.set_ylabel('Sales volume(millions)')
plot_3.set_title('Top Game Sales', fontsize = 23)
plot_3.legend(['North America','EU','Japan','Other Region'], prop={'size': 15})
fig_1, axarr_1 = plt.subplots(2, 1, figsize=(22, 27))
plt.figure(0)
plot_2 = data_7['Publisher'].value_counts().sort_values(ascending=False).head(15).rename(simply_name).plot.bar(
    fontsize=20,
    ax = axarr_1[0],
)
plot_2.set_title(" Top publishers by game releases", fontsize=40)
plot_2.set_ylabel('Number of releases(only count first time)', fontsize = 25)
plt.figure(1)
data_9 = datas.loc[:, ['Global_Sales','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Publisher']].groupby('Publisher').sum()
data_9 = data_9.rename(simply_name).sort_values('Global_Sales',ascending = False).drop(['Global_Sales'], axis=1).head(15)
plot_9 = data_9.plot.bar(stacked = True,
                        fontsize=20,
                        ax = axarr_1[1])
plot_9.set_title("Top publishers by game sales", fontsize=40)
plot_9.set_ylabel('Sales volume(millions)',fontsize = 25)
plot_9.legend(['North America','EU','Japan','Other Region'], prop={'size': 20})
plot_9.set_xlabel('')
plt.tight_layout()
plt.show()
fig_2, axarr_2 = plt.subplots(1, 2, figsize=(22, 10))
plt.figure(0)
data_14 = data_7.loc[:, ['Global_Sales','Publisher']]
data_14a = data_14.groupby('Publisher').sum()
data_14a['Number_of_Games_Released'] = data_14['Publisher'].value_counts()
plot_14a = data_14a.plot.scatter(x='Number_of_Games_Released', y='Global_Sales', ax= axarr_2[0],fontsize=14)
plot_14a.set_title("Global Sales VS Number of game released", fontsize=25)
plot_14a.set_ylabel('Global sales(million)',fontsize = 20)
plot_14a.set_xlabel('No. of game released',fontsize = 20)
data_14b = data_14a.copy()
data_14b['log_(Number_of_Games_Released)'] = data_14b.Number_of_Games_Released.map(lambda p: math.log(p))
data_14b['log_(Global_Sales)'] = data_14b.Global_Sales.map(lambda p: math.log(p))
plt.figure(1)
plot_14b = data_14b.plot.scatter(x='log_(Number_of_Games_Released)', y='log_(Global_Sales)', ax = axarr_2[1],fontsize=14)
plot_14b.set_title("log (Global Sales) VS log (Number of game released)", fontsize=25)
plot_14b.set_ylabel('log (Global sales(million))',fontsize = 20)
plot_14b.set_xlabel('log (No. of game released)',fontsize = 20)


plt.show()
data_8 = datas.loc[:, ['Year_of_Release','Genre']]

dummy_genre = pd.get_dummies(data_8['Genre'])
data_8 = pd.concat([data_8, dummy_genre], axis=1)
data_8 = data_8.groupby('Year_of_Release').sum().drop([2017,2020], axis=0)

colors = [plt.cm.Spectral(i) for i in np.linspace(0, 1, 12)]
plot_8 = data_8.plot.line(figsize=(19, 8), color = colors,linewidth= 2,fontsize=14)
plot_8.set_ylabel('Number of game released', fontsize = 17)
plot_8.set_xlabel('Year', fontsize = 17)
plot_8.set_title("Games Released between 1980-2016 by genres", fontsize=25)
data_5 = datas.loc[:, ['Genre','NA_Sales','EU_Sales','JP_Sales','Other_Sales']].groupby('Genre').sum()
plot_5 = data_5.apply(lambda c: 100 * c / c.sum(), axis=1).plot.bar(
    figsize=(16, 8),
    stacked=True,
    fontsize=16
)
plot_5.set_xlabel('Genre',fontsize = 16)
plot_5.set_ylabel('Percentage(%)',fontsize = 16)
plot_5.set_title("% Sales by Players in different regions",fontsize = 23)
plt.legend(['North America','EU','Japan','Other Region'],loc = (1,0.8),prop={'size': 13.5})
data_6 = data_7.loc[:, ['Publisher','Genre']]
num_genre = len(data_6['Genre'].value_counts().index)
fig, axarr = plt.subplots(4, 3, figsize=(22, 27))
simply_name = {'Sony Computer Entertainment':'Sony',
    'Konami Digital Entertainment': 'Konami',
    'Electronic Arts': 'EA',
    'Namco Bandai Games': 'Namco Bandai',
    'Nippon Ichi Software': 'Nippon Ichi'}
for n in range(num_genre):
    g = data_6['Genre'].value_counts().index[n]
    data_6a = data_6[data_6['Genre'] == g]['Publisher'].value_counts()
    data_6b = data_6a.head(5).append(pd.Series(0, index = ['others']))
    data_6b['others'] = sum(data_6a.iloc[5:])
    data_6b = data_6b.rename(simply_name)
    plt.figure(n)
    piec = data_6b.plot.pie(fontsize=18, autopct='%1.0f%%',radius =1.1, ax=axarr[n//3][(n-1)%3], figsize = (30,38))
    piec.set_title(g, fontsize=24)
    piec.set_ylabel('')
    plt.subplots_adjust(top=0.94)
fig.suptitle("% of games released by top publishers according to game genres", fontsize=40)
plt.tight_layout()
plt.show()
data_11 = datas.loc[:, ['Global_Sales','Developer']]
data_12 = datas.loc[:, ['Critic_Score','Developer']].set_index('Developer')
top_publ = data_11.groupby('Developer').sum().sort_values(by = 'Global_Sales',ascending = False).head(12).index
sns.set(rc={'figure.figsize':(20,9)})
plot_12 = sns.boxplot(x= 'Developer',
                         y = 'Critic_Score',
                         data =  data_12.loc[list(top_publ),:].rename(simply_name).reset_index())
plot_12.set_title("Scores of games on Metacritic for top developers (have highest sales)", fontsize=25)
plot_12.set_ylabel('Metascores of Games', fontsize = 18)
plot_12.set_xlabel('Developer', fontsize = 18)
plot_12.tick_params(labelsize=12)
plt.show()
data_4 = datas.loc[:, ['Critic_Score','Critic_Count','User_Score','User_Count']]
data_4a = data_4.loc[(datas['User_Count'] > 100)&(datas['Critic_Count'] > 10)&(datas['User_Score'] != 'NaN')]
data_4a = data_4a.assign(User_Score=data_4a["User_Score"].map(lambda p: pd.to_numeric(p)).values)
plot_4a = sns.jointplot(x='Critic_Score', y='User_Score', data=data_4a[(data_4a['User_Score']>4)&(data_4a['Critic_Score']>40)], kind='hex',
              gridsize=30,height =8.5)
plot_4a.set_axis_labels('Metascore', 'User Score', fontsize=16)
fig = plot_4a.fig
plt.subplots_adjust(top=0.94)
plot_4a.fig.suptitle('Media VS Player', fontsize=20)
plt.show()
plot_1 = datas['Platform'].value_counts().sort_values(ascending=False).plot.bar(
    figsize=(16, 7.25),
    fontsize=16,
    #title='Games published by Platform',
)
plot_1.set_title("Games published by Platform", fontsize=23)
plot_1.set_xlabel("Platforms", fontsize=18)
plot_1.set_ylabel("No. of Game released", fontsize=18)
data_10 = datas.loc[:, ['Year_of_Release','Platform']]
dummy_platform = pd.get_dummies(data_10['Platform'])
data_10 = pd.concat([data_10, dummy_platform], axis=1)
data_10 = data_10.groupby('Year_of_Release').sum().drop([2017,2020], axis=0)
top_platform = data_10.sum().sort_values(ascending = False).head(10).index
plot_10 = data_10.loc[1990:,list(top_platform)].plot.line(figsize = (18,8),linewidth=3,fontsize = 15)
plot_10.set_title("Game Released between 1980-2016 for Top 10 Platforms", fontsize=23)
plot_10.set_ylabel("No. of Game released", fontsize=18)
plot_10.set_xlabel("Year", fontsize=18)
data_13 = datas.loc[:, ['Global_Sales','Platform','Year_of_Release']]
top_platform = data_13.loc[:,['Platform','Global_Sales']].groupby('Platform').sum().sort_values('Global_Sales', ascending = False).head(10).index
flag = False
for g in range(len(top_platform)):
    platform = top_platform[g]
    if flag == True:
        data_13a = pd.merge(data_13a, data_13.set_index('Platform').loc[platform,:].reset_index(), how = 'outer')
    else:
        data_13a = data_13.set_index('Platform').loc[platform,:].reset_index()
        flag = True
dummy_platforms = pd.get_dummies(data_13a.set_index('Year_of_Release')['Platform'])
data_13b = pd.concat([data_13a.set_index('Year_of_Release'), dummy_platforms], axis=1).reset_index().groupby(['Platform','Year_of_Release']).sum().reset_index().set_index('Year_of_Release').drop(['Platform','Global_Sales'], axis = 1).drop([1985,1988, 2017, 2020], axis = 0)
plot_13 = data_13b.groupby('Year_of_Release').sum().plot.area(figsize = (16.18,8),
                                                             fontsize = 14)
plot_13.set_title("Games' sales volume on top 10 Consoles between 1990-2016", fontsize=22)
plot_13.set_ylabel("Sales (millions)", fontsize=18)
plot_13.set_xlabel("Year", fontsize=18)