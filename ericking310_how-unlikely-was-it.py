import pandas as pd 

import random as rnd

import numpy as np 

import matplotlib.pyplot as plt
rapgl = pd.read_csv('../input/raptors-playoff-run-20182019/rap_allgames.csv')



wargl = pd.read_csv('../input/raptors-playoff-run-20182019/warriors_allgames.csv')



maggl = pd.read_csv('../input/raptors-playoff-run-20182019/magic_allgames.csv')



sixgl = pd.read_csv('../input/raptors-playoff-run-20182019/76ers_allgames.csv')



buckgl = pd.read_csv('../input/raptors-playoff-run-20182019/bucks_allgames.csv')
rapgl.Tm.hist()

rapgl.Opp.hist()
wargl.Tm.hist()

wargl.Opp.hist()
maggl.Tm.hist()

maggl.Opp.hist()
sixgl.Tm.hist()

sixgl.Opp.hist()
buckgl.Tm.hist()

buckgl.Opp.hist()
# Now, let's find the average points scored, and allowed,(and standard deviations) per game for each team

rap_ppg=rapgl.Tm.mean() 

rap_oppppg=rapgl.Opp.mean()

rap_ppgstd=rapgl.Tm.std() 

rap_oppppgstd=rapgl.Opp.std()



warr_ppg=wargl.Tm.mean()

warr_oppppg=wargl.Opp.mean()

warr_ppgstd=wargl.Tm.std()

warr_oppppgstd=wargl.Opp.std()



mag_ppg=maggl.Tm.mean()

mag_oppppg=maggl.Opp.mean()

mag_ppgstd=maggl.Tm.std()

mag_oppppgstd=maggl.Opp.std()



six_ppg=sixgl.Tm.mean()

six_oppppg=sixgl.Opp.mean()

six_ppgstd=sixgl.Tm.std()

six_oppppgstd=sixgl.Opp.std()



buck_ppg=buckgl.Tm.mean()

buck_oppppg=buckgl.Opp.mean()

buck_ppgstd=buckgl.Tm.std()

buck_oppppgstd=buckgl.Opp.std()
#First round simulation between Raptors and Orlando

def RvOgameSim():

    RapScore = (rnd.gauss(rap_ppg,rap_ppgstd)+ rnd.gauss(mag_oppppg,mag_oppppgstd))/2

    OrlScore = (rnd.gauss(mag_ppg,mag_ppgstd)+ rnd.gauss(rap_oppppg,rap_oppppgstd))/2

    if int(RapScore) > int((OrlScore)):

        return 1

    elif int(RapScore) < int(OrlScore):

        return -1



    return 0



def RvOgamesSim(ns):

    gamesout = []

    team1win = 0

    team2win = 0

    tie = 0

    for i in range(ns):

        gm = RvOgameSim()

        gamesout.append(gm)

        if gm == 1:

            team1win +=1 

        elif gm == -1:

            team2win +=1

        else: tie +=1 

    print('Raptors Win ', (team1win/(team1win+team2win+tie))*100,'%')

    print('Magic Win ', (team2win/(team1win+team2win+tie))*100,'%')

    print('Tie ', (tie/(team1win+team2win+tie))*100, '%')
RvOgamesSim(10000)
# Second Round matchup vs the 76ers

def RvSgameSim():

    RapScore = (rnd.gauss(rap_ppg,rap_ppgstd)+ rnd.gauss(six_oppppg,six_oppppgstd))/2

    SixScore = (rnd.gauss(six_ppg,six_ppgstd)+ rnd.gauss(rap_oppppg,rap_oppppgstd))/2

    if int(RapScore) > int(SixScore):

        return 1

    elif int(RapScore) < int(SixScore):

        return -1

    return 0
def RvSgamesSim(ns):

    gamesout = []

    team1win = 0

    team2win = 0

    tie = 0

    for i in range(ns):

        gm = RvSgameSim()

        gamesout.append(gm)

        if gm == 1:

            team1win +=1 

        elif gm == -1:

            team2win +=1

        else: tie +=1 

    print('Raptors Win ', 100*team1win/(team1win+team2win+tie),'%')

    print('Sixers Win ', 100*team2win/(team1win+team2win+tie),'%')

    print('Tie ', 100*tie/(team1win+team2win+tie), '%')
RvSgamesSim(10000)
# In 10000 simulated games, the Raptors win about 54% of the games.
def RvBgameSim():

    RapScore = (rnd.gauss(rap_ppg,rap_ppgstd)+ rnd.gauss(buck_oppppg,buck_oppppgstd))/2

    BuckScore = (rnd.gauss(buck_ppg,buck_ppgstd)+ rnd.gauss(rap_oppppg,rap_oppppgstd))/2

    if int(RapScore) > int(BuckScore):

        return 1

    elif int(RapScore) < int(BuckScore):

        return -1

    return 0
def RvBgamesSim(ns):

    gamesout = []

    team1win = 0

    team2win = 0

    tie = 0

    for i in range(ns):

        gm = RvBgameSim()

        gamesout.append(gm)

        if gm == 1:

            team1win +=1 

        elif gm == -1:

            team2win +=1

        else: tie +=1 

    print('Raptors Win ', 100*team1win/(team1win+team2win+tie),'%')

    print('Bucks Win ', 100*team2win/(team1win+team2win+tie),'%')

    print('Tie ', 100*tie/(team1win+team2win+tie), '%')

    
RvBgamesSim(10000)
def RvWgameSim():

    RapScore = (rnd.gauss(rap_ppg,rap_ppgstd)+ rnd.gauss(warr_oppppg,warr_oppppgstd))/2

    WarrScore = (rnd.gauss(warr_ppg,warr_ppgstd)+ rnd.gauss(rap_oppppg,rap_oppppgstd))/2

    if int(RapScore) > int(WarrScore):

        return 1

    elif int(RapScore) < int(WarrScore):

        return -1

    return 0
def RvWgamesSim(ns):

    gamesout = []

    team1win = 0

    team2win = 0

    tie = 0

    for i in range(ns):

        gm = RvWgameSim()

        gamesout.append(gm)

        if gm == 1:

            team1win +=1 

        elif gm == -1:

            team2win +=1

        else: tie +=1 

    print('Raptors Win ', 100*team1win/(team1win+team2win+tie),'%')

    print('Warriors Win ', 100*team2win/(team1win+team2win+tie),'%')

    print('Tie ', 100*tie/(team1win+team2win+tie), '%')

    
RvWgamesSim(10000)
RvWgamesSim(1000000)
a= (0.5818 * 0.5443  * 0.4287  * 0.4704 )*100
a