# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
datasets = ['../input/season-0809_csv.csv','../input/season-0910_csv.csv','../input/season-1011_csv.csv',

     '../input/season-1112_csv.csv','../input/season-1213_csv.csv','../input/season-1314_csv.csv',

     '../input/season-1415_csv.csv','../input/season-1516_csv.csv','../input/season-1617_csv.csv',

     '../input/season-1718_csv.csv']

season = ['2008/2009','2009/2010','2010/2011','2011/2012','2012/2013','2013/2014',

          '2014/2015','2015/2016','2016/2017','2017/2018']

df = pd.DataFrame()

for i in range(10):

    df_temp = pd.read_csv(datasets[i])

    df_temp['Date'] = season[i]

    df = pd.concat([df,df_temp])

df.loc[df.FTR == 'H','WinTeam'] = df.loc[df.FTR == 'H','HomeTeam']

df.loc[df.FTR == 'A','WinTeam'] = df.loc[df.FTR == 'A','AwayTeam']

df.loc[df.FTR == 'D','WinTeam'] = 'Draw'



#Specify Managers for each season

manager1 = ['2008/2009', '2009/2010', '2010/2011','2011/2012']

manager2 = ['2012/2013']

manager3 = ['2013/2014']

manager4 = ['2014/2015','2015/2016','2016/2017']

manager5 = ['2017/2018']

managers = ['Guardiola', 'Vilanova', 'Martino', 'Enrique', 'Valverde']

df.loc[df.Date.isin(manager1), 'Manager'] = 'Guardiola'

df.loc[df.Date.isin(manager2), 'Manager'] = 'Vilanova'

df.loc[df.Date.isin(manager3), 'Manager'] = 'Martino'

df.loc[df.Date.isin(manager4), 'Manager'] = 'Enrique'

df.loc[df.Date.isin(manager5), 'Manager'] = 'Valverde'



# Extract Barca Match

df_barca = df.loc[(df['HomeTeam'] == 'Barcelona') | (df['AwayTeam'] == 'Barcelona')]

df_barca.head()
win_rate = pd.DataFrame()

for i in managers:

    manager_total = df_barca[df_barca['Manager'] == i]['Date'].count()

    manager_win = df_barca[(df_barca.WinTeam == 'Barcelona') & (df_barca.Manager == i)]['Date'].count()

    manager_draw = df_barca[(df_barca.WinTeam == 'Draw') & (df_barca.Manager == i)]['Date'].count()

    manager_lose = manager_total - manager_win - manager_draw

    manager_perform = (manager_win * 3 + manager_draw) / manager_total

    

    # Win rate for Home/Away

    manager_home = df_barca[(df_barca.HomeTeam == 'Barcelona') & (df_barca.Manager==i)]['Date'].count()

    manager_away = df_barca[(df_barca.HomeTeam != 'Barcelona') & (df_barca.Manager==i)]['Date'].count()

    manager_home_win = df_barca[(df_barca.HomeTeam == 'Barcelona') & (df_barca.Manager==i) & (df_barca.WinTeam=='Barcelona')]['Date'].count()

    manager_home_lose = df_barca[(df_barca.HomeTeam == 'Barcelona') & (df_barca.Manager==i) & (df_barca.WinTeam!='Barcelona') & (df_barca.WinTeam!='Draw')]['Date'].count()

    manager_home_draw = df_barca[(df_barca.HomeTeam == 'Barcelona') & (df_barca.Manager==i) & (df_barca.WinTeam=='Draw')]['Date'].count()

    

    manager_away_win = df_barca[(df_barca.HomeTeam != 'Barcelona') & (df_barca.Manager==i) & (df_barca.WinTeam=='Barcelona')]['Date'].count()

    manager_away_lose = df_barca[(df_barca.HomeTeam != 'Barcelona') & (df_barca.Manager==i) & (df_barca.WinTeam!='Barcelona') & (df_barca.WinTeam!='Draw')]['Date'].count()

    manager_away_draw = df_barca[(df_barca.HomeTeam != 'Barcelona') & (df_barca.Manager==i) & (df_barca.WinTeam=='Draw')]['Date'].count()

    win_rate = win_rate.append(pd.Series([i,manager_total, manager_win/manager_total, 

                                          manager_lose/manager_total, manager_draw/manager_total, manager_perform, 

                                          manager_home_win/manager_home, manager_home_lose/manager_home, manager_home_draw/manager_home,

                                        manager_away_win/manager_away, manager_away_lose/manager_away, manager_away_draw/manager_away]), ignore_index=True)

    print('{}: {} Total matches, {} wins, {} losses, {} draws, {:.2f} points per match'.format(i, manager_total, manager_win, manager_lose, manager_draw, manager_perform))

    
win_rate.columns = ['Manager', 'Total_Match', 'Win_Rate', 'Lose_Rate', 'Draw_Rate','Points per match',

                    'Win_Rate_Home', 'Lose_Rate_Home','Draw_Rate_Home',

                   'Win_Rate_Away', 'Lose_Rate_Away', 'Draw_Rate_Away']

win_rate
fig, ax =plt.subplots()

fig.set_size_inches(10,7)

index = np.arange(5)

bar_width = 0.25





rects1 = ax.bar(index, win_rate.Win_Rate, bar_width,

               alpha=0.5, color='g', label = 'Win Rate')

rects2 = ax.bar(index+bar_width, win_rate.Lose_Rate, bar_width,

               alpha=0.5, color='r', label = 'Lose Rate')

rects3 = ax.bar(index+2*bar_width, win_rate.Draw_Rate, bar_width,

               alpha=0.5, label = 'Draw Rate')



ax.set_xlabel('Managers')

ax.set_ylabel('Rate')

ax.set_title('Overall Win/Lose/Draw Rate')

ax.set_xticks(index + bar_width / 2)

ax.set_xticklabels(('Guardiola', 'Vilanova', 'Martino', 'Enrique', 'Valverde'))

ax.legend()



plt.show()
fig, ax =plt.subplots()

fig.set_size_inches(20,7)

index = np.arange(5)

bar_width = 0.15





rects1 = ax.bar(index, win_rate.Win_Rate_Home, bar_width,

               alpha=0.5, color='g', label = 'Win Rate(Home)')

rects2 = ax.bar(index+bar_width, win_rate.Win_Rate_Away, bar_width,

               alpha=0.5, color='blue', label = 'Win Rate(Away)')

rects3 = ax.bar(index+2*bar_width, win_rate.Lose_Rate_Home, bar_width,

               alpha=0.5, color='r', label = 'Lose Rate(Home)')

rects4 = ax.bar(index+3*bar_width, win_rate.Lose_Rate_Away, bar_width,

               alpha=0.5, color='black', label = 'Lose Rate(Away)')

rects5 = ax.bar(index+4*bar_width, win_rate.Draw_Rate_Home, bar_width,

               alpha=0.5, label = 'Draw Rate(Home)')

rects6 = ax.bar(index+5*bar_width, win_rate.Draw_Rate_Away, bar_width,

               alpha=0.5, label = 'Draw Rate(Away)')



ax.set_xlabel('Managers')

ax.set_ylabel('Rate')

ax.set_title('Home/Away Win/Lose/Draw Rate')

ax.set_xticks(index + 5*bar_width / 2)

ax.set_xticklabels(('Guardiola', 'Vilanova', 'Martino', 'Enrique', 'Valverde'))

ax.legend()



plt.show()
Barca_Goal = pd.DataFrame()

for i in managers:

    # Goal Counts

    Barca_Home_GS = df_barca.loc[(df_barca.HomeTeam=='Barcelona') & (df_barca.Manager==i)]['FTHG'].mean() #Barca scored as Home

    Barca_Home_GC = df_barca.loc[(df_barca.HomeTeam=='Barcelona') & (df_barca.Manager==i)]['FTAG'].mean() #Barca conceded as Home

    Barca_Away_GS = df_barca.loc[(df_barca.HomeTeam!='Barcelona') & (df_barca.Manager==i)]['FTAG'].mean() #Barca scored as Away

    Barca_Away_GC = df_barca.loc[(df_barca.HomeTeam!='Barcelona') & (df_barca.Manager==i)]['FTHG'].mean() #Barca conceded as Away

    Barca_Total_GS = (Barca_Home_GS + Barca_Away_GS)/2

    Barca_Total_GC = (Barca_Home_GC + Barca_Away_GC)/2

    # Shoot Counts

    Barca_Home_S = df_barca.loc[(df_barca.HomeTeam=='Barcelona') & (df_barca.Manager==i)]['HS'].mean() # Barca Shoots as Home

    Barca_Home_S_ = df_barca.loc[(df_barca.HomeTeam=='Barcelona') & (df_barca.Manager==i)]['AS'].mean() # Barca's opponent Shoots

    Barca_Away_S = df_barca.loc[(df_barca.HomeTeam!='Barcelona') & (df_barca.Manager==i)]['AS'].mean() # Barca Shoots as Away

    Barca_Away_S_ = df_barca.loc[(df_barca.HomeTeam!='Barcelona') & (df_barca.Manager==i)]['HS'].mean() # Barca' opponent Shoots

    Barca_Total_S = (Barca_Home_S + Barca_Away_S)/2

    Barca_Total_S_ = (Barca_Home_S_ + Barca_Away_S_)/2

    # Shoot on Target Counts

    Barca_Home_ST = df_barca.loc[(df_barca.HomeTeam=='Barcelona') & (df_barca.Manager==i)]['HST'].mean() # Barca Shoots on Target as Home

    Barca_Home_ST_ = df_barca.loc[(df_barca.HomeTeam=='Barcelona') & (df_barca.Manager==i)]['AST'].mean() # Barca's opponent Shoots on Target

    Barca_Away_ST = df_barca.loc[(df_barca.HomeTeam!='Barcelona') & (df_barca.Manager==i)]['AST'].mean() # Barca Shoots on Target as Away

    Barca_Away_ST_ = df_barca.loc[(df_barca.HomeTeam!='Barcelona') & (df_barca.Manager==i)]['HST'].mean() # Barca' opponent Shoots on Target

    # Corners

    Barca_Home_C = df_barca.loc[(df_barca.HomeTeam=='Barcelona') & (df_barca.Manager==i)]['HC'].mean() # Barca Corner as Home

    Barca_Home_C_ = df_barca.loc[(df_barca.HomeTeam=='Barcelona') & (df_barca.Manager==i)]['AC'].mean() # Barca's opponent Corner

    Barca_Away_C = df_barca.loc[(df_barca.HomeTeam!='Barcelona') & (df_barca.Manager==i)]['AC'].mean() # Barca Corner as Away

    Barca_Away_C_ = df_barca.loc[(df_barca.HomeTeam!='Barcelona') & (df_barca.Manager==i)]['HC'].mean() # Barca' opponent Corner



    Barca_Goal = Barca_Goal.append(pd.Series([i,Barca_Total_GS, Barca_Total_GC,Barca_Total_S,Barca_Total_S_,

                                              Barca_Home_GS, Barca_Home_GC, Barca_Away_GS, Barca_Away_GC,

                                             Barca_Home_S, Barca_Home_S_, Barca_Away_S, Barca_Away_S_,

                                             Barca_Home_ST, Barca_Home_ST_, Barca_Away_ST, Barca_Away_ST_,

                                             Barca_Home_C, Barca_Home_C_, Barca_Away_C, Barca_Away_C_]),ignore_index=True)

Barca_Goal.columns = ['Manager', 'Barca_Total_Scored', 'Barca_Total_Conceded', 'Barca_Total_Shoot', 'Bara_Total_Shoot_',

                      'Barca_Home_Scored','Barca_Home_Conceded','Barca_Away_Scored', 'Barca_Away_Conceded',

                     'Barca_Home_Shoot','Barca_Home_Shoot_','Barca_Away_Shoot','Barca_Away_Shoot_',

                     'Barca_Home_ST','Barca_Home_ST_','Barca_Away_ST','Barca_Away_ST_',

                     'Barca_Home_C','Barca_Home_C_', 'Barca_Away_C', 'Barca_Away_C_']

Barca_Goal
fig, ax =plt.subplots()

fig.set_size_inches(10,7)

index = np.arange(5)

bar_width = 0.25





rects1 = ax.bar(index, Barca_Goal.Barca_Home_Scored, bar_width,

               alpha=0.5, color='g', label = 'Home Scored')

rects2 = ax.bar(index+bar_width, Barca_Goal.Barca_Away_Scored, bar_width,

               alpha=0.5, color='r', label = 'Away Scored')





ax.set_xlabel('Managers')

ax.set_ylabel('Rate')

ax.set_title('Goals')

ax.set_xticks(index + bar_width / 2)

ax.set_xticklabels(('Guardiola', 'Vilanova', 'Martino', 'Enrique', 'Valverde'))

ax.legend()



plt.show()
goal_by = df_barca[['Manager','HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','FTR']].copy()

goal_by['Result'] = 'Draw'

goal_by.loc[((goal_by['HomeTeam']=='Barcelona')&(goal_by['FTR']=='H')) |

            ((goal_by['AwayTeam']=='Barcelona')&(goal_by['FTR']=='A')),'Result'] = 'Win'

goal_by.loc[((goal_by['HomeTeam']=='Barcelona')&(goal_by['FTR']=='A')) |

            ((goal_by['AwayTeam']=='Barcelona')&(goal_by['FTR']=='H')),'Result'] = 'Lose'

goal_by.reset_index(drop=True,inplace=True)

goal, goalby=[], []

for i in goal_by.index:

    if goal_by.iloc[i]['HomeTeam'] == 'Barcelona':

        goal.append(goal_by.iloc[i]['FTHG'])

        goalby.append(goal_by.iloc[i]['FTAG'])

    else:

        goal.append(goal_by.iloc[i]['FTAG'])

        goalby.append(goal_by.iloc[i]['FTHG'])

goal_by['Barca_Goal'], goal_by['Opponent_Goal']= goal, goalby



#goal_by.loc[goal_by.Manager=='Guardiola'].boxplot(column='Barca_Goal', by='Result')

#goal_by.loc[goal_by.Manager=='Enrique'].boxplot(column='Barca_Goal', by='Result')

#goal_by.loc[(goal_by.Manager=='Enrique') & (goal_by.Result=='Lose')]

#goal_by

plt.show()
df_goal_by = pd.DataFrame({'Manager':managers})

goalW, goalL, goalD, goal_W, goal_L, goal_D = [], [], [], [], [], []



for i in managers:

    goalW.append(goal_by.loc[(goal_by.Manager==i)&(goal_by.Result=='Win')]['Barca_Goal'].mean())

    goalL.append(goal_by.loc[(goal_by.Manager==i)&(goal_by.Result=='Lose')]['Barca_Goal'].mean())

    goalD.append(goal_by.loc[(goal_by.Manager==i)&(goal_by.Result=='Draw')]['Barca_Goal'].mean())

    goal_W.append(goal_by.loc[(goal_by.Manager==i)&(goal_by.Result=='Win')]['Opponent_Goal'].mean())

    goal_L.append(goal_by.loc[(goal_by.Manager==i)&(goal_by.Result=='Lose')]['Opponent_Goal'].mean())

    goal_D.append(goal_by.loc[(goal_by.Manager==i)&(goal_by.Result=='Draw')]['Opponent_Goal'].mean())

    

df_goal_by['Barca_Goal_Winner']=goalW

df_goal_by['Barca_Goal_Loser']=goalL

df_goal_by['Barca_Goal_Draw']=goalD

df_goal_by['Barca_GoalConceded_Winner']=goal_W

df_goal_by['Barca_GoalConceded_Loser']=goal_L

df_goal_by['Barca_GoalConceded_Draw']=goal_D

#goal_average 

df_goal_by
fig, ax =plt.subplots()

fig.set_size_inches(10,7)

index = np.arange(5)

bar_width = 0.25



rects1 = ax.bar(index, df_goal_by.Barca_Goal_Winner, bar_width,

               alpha=0.5, color='g', label = 'Goals as Winner')

rects2 = ax.bar(index+bar_width, df_goal_by.Barca_Goal_Loser, bar_width,

               alpha=0.5, color='r', label = 'Goals as Loser')

rects3 = ax.bar(index+2*bar_width, df_goal_by.Barca_Goal_Draw, bar_width,

               alpha=0.5, label = 'Goals in Draw Games')



ax.set_xlabel('Managers')

ax.set_ylabel('Average Goal')

ax.set_title('Goals in Win/Draw/Lose Game')

ax.set_xticks(index + bar_width / 2)

ax.set_xticklabels(('Guardiola', 'Vilanova', 'Martino', 'Enrique', 'Valverde'))

ax.legend()



plt.show()
#columns = ['Manager', 'WinTeam', 'GoalC', 'OppF', 'OppC', 'BarcaY', 'BarcaR']



#for i in managers:

#    OppC.append(df_defend.loc[(df_defend.HomeTeam=='Bareclona') & (df_defend.Manager==i)]['AC'].mean() +

#                 df_defend.loc[(df_defend.HomeTeam!='Bareclona') & (df_defend.Manager==i)]['HC'].mean())

#    BarcaF.append(df_defend.loc[(df_defend.HomeTeam=='Bareclona') & (df_defend.Manager==i)]['HF'].mean() +

#                 df_defend.loc[(df_defend.HomeTeam!='Bareclona') & (df_defend.Manager==i)]['AF'].mean())
Barca_Goal_Efficiency = pd.DataFrame({'Manager': managers})



Barca_Goal_Efficiency['Total_Score_Efficiency'] = Barca_Goal['Barca_Total_Scored']/Barca_Goal['Barca_Total_Shoot']

Barca_Goal_Efficiency['Home_Score_Efficiency'] = Barca_Goal['Barca_Home_Scored']/Barca_Goal['Barca_Home_Shoot']

Barca_Goal_Efficiency['Away_Score_Efficiency'] = Barca_Goal['Barca_Away_Scored']/Barca_Goal['Barca_Away_Shoot']



Barca_Goal_Efficiency
fig, ax =plt.subplots()

fig.set_size_inches(10,8)

index = np.arange(5)

bar_width = 0.25



rects1 = ax.bar(index, Barca_Goal.Barca_Total_Shoot,bar_width, 

               alpha=0.5, color='g', label = 'Total Shots')

for i, v in enumerate(Barca_Goal['Barca_Total_Shoot'].round(2)):

    ax.text(i-0.1,v+0.3,  str(v), color='g')

rects2 = ax.bar(index+bar_width, Barca_Goal.Barca_Total_Scored, bar_width,

               alpha=0.5, color='r', label = 'Total Goal Scored')



for i, v in enumerate(Barca_Goal['Barca_Total_Scored'].round(2)):

    ax.text(i+0.15,v+0.3,  str(v), color='r')



ax.set_xlabel('Managers')

ax.set_ylabel('Rate')

ax.set_title('Goal & Shots')

ax.set_xticks(index + bar_width / 2)

ax.set_xticklabels(('Guardiola', 'Vilanova', 'Martino', 'Enrique', 'Valverde'))

ax.legend(loc='center')



plt.show()
fig, ax =plt.subplots()

fig.set_size_inches(10,7)

index = np.arange(5)

bar_width = 0.25





rects1 = ax.bar(index, Barca_Goal_Efficiency.Total_Score_Efficiency, bar_width,

               alpha=0.5, color='g', label = 'Total Score Efficiency')





line1 = ax.plot(index, Barca_Goal_Efficiency.Home_Score_Efficiency)

line2 = ax.plot(index, Barca_Goal_Efficiency.Away_Score_Efficiency)



ax.set_xlabel('Managers')

ax.set_ylabel('Rate')

ax.set_title('Goal Efficiency')

ax.set_xticks(index + bar_width / 2)

ax.set_xticklabels(('Guardiola', 'Vilanova', 'Martino', 'Enrique', 'Valverde'))

ax.legend()



plt.show()
plt.plot(managers,Barca_Goal.Barca_Total_Conceded)

plt.show()
df_defend = df_barca[['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR','WinTeam', 'Manager']]



df_defend.head()
df_goal_by.head()
fig, ax =plt.subplots()

fig.set_size_inches(10,7)

index = np.arange(5)

bar_width = 0.25



rects1 = ax.bar(index, df_goal_by.Barca_GoalConceded_Winner, bar_width,

               alpha=0.5, color='g', label = 'Goals Conceded as Winner')

rects2 = ax.bar(index+bar_width, df_goal_by.Barca_GoalConceded_Loser, bar_width,

               alpha=0.5, color='r', label = 'Goals Conceded as Loser')

rects3 = ax.bar(index+2*bar_width, df_goal_by.Barca_GoalConceded_Draw, bar_width,

               alpha=0.5, label = 'Goals Conceded in Draw Games')



ax.set_xlabel('Managers')

ax.set_ylabel('Average Goals Conceded')

ax.set_title('Goals Conceded in Win/Draw/Lose Game')

ax.set_xticks(index + bar_width / 2)

ax.set_xticklabels(('Guardiola', 'Vilanova', 'Martino', 'Enrique', 'Valverde'))

ax.legend()



plt.show()
df_vs = pd.DataFrame()

df_vs = df_barca.loc[((df_barca.HomeTeam =='Barcelona') & (df_barca.AwayTeam.isin(['Real Madrid', 'Ath Madrid']))) |

              ((df_barca.AwayTeam=='Barcelona') & (df_barca.HomeTeam.isin(['Real Madrid', 'Ath Madrid'])))]

df_vs.head()
df_vs_stat = pd.DataFrame()

for i in managers:

    

    win = df_vs.loc[(df_vs.WinTeam == 'Barcelona')& (df_vs.Manager == i)]['Date'].count()

    lose = df_vs.loc[(df_vs.WinTeam.isin(['Ath Madrid', 'Real Madrid']))& (df_vs.Manager == i)]['Date'].count()

    draw = df_vs.loc[(df_vs.WinTeam == 'Draw')& (df_vs.Manager == i)]['Date'].count()

    goal = (df_vs.loc[((df_vs.HomeTeam=='Barcelona') & (df_vs.Manager==i))]['FTHG'].mean() + df_vs.loc[((df_vs.AwayTeam=='Barcelona')&(df_vs.Manager==i))]['FTAG'].mean())/2

    goal_ = (df_vs.loc[((df_vs.HomeTeam=='Barcelona') & (df_vs.Manager==i))]['FTAG'].mean() + df_vs.loc[((df_vs.AwayTeam=='Barcelona')&(df_vs.Manager==i))]['FTHG'].mean())/2

    

    total_match = df_vs.loc[df_vs.Manager==i]['Date'].count()

    perform = (3*win + draw) / total_match

    

    df_vs_stat = df_vs_stat.append(pd.Series([i, total_match, win, lose, draw, win/total_match, lose/total_match, draw/total_match, perform, goal, goal_]), ignore_index=True)

    

df_vs_stat.columns=['Manager', 'Total_Match', 'Win', 'Lose', 'Draw', 'Win_Rate', 'Lose_Rate', 'Draw_Rate','Points_per_Match','Goal_Scored', 'Goal_Conceded']

df_vs_stat
win_rate, lose_rate, draw_rate= df_vs_stat['Win_Rate'], df_vs_stat['Lose_Rate'], df_vs_stat['Draw_Rate']



plt.figure(1, figsize=(15,3))

plt.subplot(131)

plt.bar(managers,win_rate,color='g',alpha=0.5)

plt.subplot(132)

plt.bar(managers, lose_rate,color='r',alpha=0.5)

plt.subplot(133)

plt.bar(managers, draw_rate,color='b',alpha=0.5)

plt.suptitle('Win/Lose/Draw Rate Against Ath Mardird and Real Mardrid')

plt.show()
fig, ax =plt.subplots()

fig.set_size_inches(10,7)

index = np.arange(5)

bar_width = 0.25





rects1 = ax.bar(index, df_vs_stat.Goal_Scored, bar_width,

               alpha=0.5, color='g', label = 'Barca Goals Scored')

rects2 = ax.bar(index+bar_width, df_vs_stat.Goal_Conceded, bar_width,

               alpha=0.5, color='r', label = 'Barca Goals Conceded')





ax.set_xlabel('Managers')

ax.set_ylabel('Goals per Match')

ax.set_title('Goals Scored/Conceded Against Ath Mardird and Real Mardrid')

ax.set_xticks(index + bar_width / 2)

ax.set_xticklabels(('Guardiola', 'Vilanova', 'Martino', 'Enrique', 'Valverde'))

ax.legend()



plt.show()