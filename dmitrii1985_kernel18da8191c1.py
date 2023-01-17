import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
import matplotlib
from matplotlib import rc
import numpy as np
from matplotlib import colors
import seaborn as sns


font = {'family': 'Droid Sans',
        'weight': 'normal'}
rc('font', **font)


def drawGr(xx,yy,fileName,x_label,y_label,tit,h = None ):
    matplotlib.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()
    fig.set_size_inches(20.5, 10.5)
    if h is not None:
        sns.barplot(x=xx, y=yy, hue = h)                                        
    else:
        sns.barplot(x=xx, y=yy) 
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(tit)
    #plt.savefig(fileName)
    plt.show()
    
def point(x,y,fileName,tit):
    fig, ax = plt.subplots(tight_layout=True)
    fig.set_size_inches(18.5, 10.5)
    hist = ax.hist2d(x, y,bins=10)
    plt.title(tit)
    #plt.savefig(fileName)
    plt.show()

import pandas as pn
import numpy as np

#load data
game = pn.read_csv('../input/game.csv')
game['date_time'] =pn.to_datetime(game['date_time'])
game_teams = pn.read_csv('../input/game_teams_stats.csv')
team_info = pn.read_csv('../input/team_info.csv')

#игроки
game_skater_stats = pn.read_csv('../input/game_skater_stats.csv')
#вратари
game_goalie_stats = pn.read_csv('../input/game_goalie_stats.csv')
#инфа о людях
player_info = pn.read_csv('../input/player_info.csv')
player_info['birthDate'] = pn.to_datetime(player_info['birthDate'])
#какие игроки играют
game_playes_players = pn.read_csv('../input/game_plays_players.csv')
def start1():
    val,count = np.unique(game['date_time'].dt.weekday_name, return_counts=True)
    drawGr(val,count,'/home/dima/Machine_Learning/nhl-game-data/img/dayWeek.png','День недели','Кол-во игр','По каким дням играют')

    ct = pn.crosstab(game_teams['HoA'],game_teams['goals'])
    stacked = ct.stack().reset_index().rename(columns={0:'value'})

    drawGr(stacked.goals,stacked.value,'/home/dima/Machine_Learning/nhl-game-data/img/goalsHomeAway.png','Количесво голов','Значение','Как забивают(дома на выезде)',stacked.HoA)

    indexHome = np.where(game_teams['HoA']=='home')[0]
    point(game_teams['goals'][indexHome],game_teams['shots'][indexHome],'/home/dima/Machine_Learning/nhl-game-data/img/shglHome.png','Количество шайб - количество бросков (Дом)')

    indexAway = np.where(game_teams['HoA']=='away')[0]
    point(game_teams['goals'][indexAway],game_teams['shots'][indexAway],'/home/dima/Machine_Learning/nhl-game-data/img/shglAway.png','Количество шайб - количество бросков (Выезд)')
start1()

def start2():
    # тренера которые больше всех выигрывали
    ct = pn.crosstab(game_teams['head_coach'],game_teams['won']).apply(lambda r: r, axis=1,result_type='expand')
    stacked = ct.stack().reset_index().rename(columns={0:'value'})
    names = (stacked.loc[stacked['won']==True]).sort_index(by=['value'], ascending=[False]).head(10)['head_coach'].tolist()
    stacked = (stacked.loc[stacked['head_coach'].isin(names)])
    drawGr(stacked.head_coach,stacked.value,'/home/dima/Machine_Learning/nhl-game-data/img/head_coach.png','Тренер','Значение','Сколько выграл',stacked.won)


start2()

def start3():
    result = game_teams.merge(team_info, left_on='team_id', right_on='team_id', how='outer')

    group = result.groupby(['shortName'])['goals'].mean().reset_index().sort_index(by=['goals'], ascending=[False]).head(15)
    drawGr(group.shortName,group.goals,'/home/dima/Machine_Learning/nhl-game-data/img/avrageGoal.png','Команда','Значение','Сколько забивают за игру')

    group = result.groupby(['shortName'])['hits'].mean().reset_index().sort_index(by=['hits'], ascending=[False]).head(15)
    drawGr(group.shortName,group.hits,'/home/dima/Machine_Learning/nhl-game-data/img/avrageHit.png','Команда','Значение','Сколько столкновений за игру')
start3()
def start4():
    result = game_skater_stats.merge(player_info, left_on='player_id', right_on='player_id', how='outer')

    result = result[['goals','firstName','lastName']]
    result['name'] = result['firstName'] +' ' + result['lastName']
    result = result[['goals','name']]

    group = result.groupby(['name'])['goals'].sum().reset_index().sort_index(by=['goals'], ascending=[False]).head(10)
    drawGr(group.name,group.goals,'/home/dima/Machine_Learning/nhl-game-data/img/leader.png','Имя игрока','Количество забитых шайб','Лутшие игроки NHL')

    group = result.groupby(['name'])['goals'].mean().reset_index().sort_index(by=['goals'], ascending=[False]).head(10)
    drawGr(group.name,group.goals,'/home/dima/Machine_Learning/nhl-game-data/img/avrleader.png','Имя игрока','Среднее количество шайб за матч','Игроки которые чаще всего забивают в матче')
start4()
def start5():
    grp = player_info.groupby(['nationality'])['player_id'].count().reset_index()
    drawGr(grp.nationality,grp.player_id,'/home/dima/Machine_Learning/nhl-game-data/img/nation.png','Страна','Количество игроков','Кто играет в NHL')
    playerTmp = player_info.copy()
    playerTmp['birthDate'] = playerTmp['birthDate'].dt.year
    grp = playerTmp.groupby(['birthDate'])['player_id'].count().reset_index()
    drawGr(grp.birthDate,grp.player_id,'/home/dima/Machine_Learning/nhl-game-data/img/ageplayers.png','Год','Количество игроков','Возрастной состав в NHL')
start5()
def start6():
    result = pn.merge(game_skater_stats[['player_id','goals']],player_info[['player_id','nationality']], left_on='player_id', right_on='player_id', how='outer')
    result = result.groupby(['nationality'])['goals'].mean().reset_index()
    #inter_rater_variability(range(len(result.nationality)))
    drawGr(result.nationality,result.goals,'/home/dima/Machine_Learning/nhl-game-data/img/country.png','Страна','Частота забивания','Какая национальность чаще всего забивает')
start6()
