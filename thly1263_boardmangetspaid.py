#Thomas Lynch

#CSCI 3022 Final Project

#last editted on july 11, 2020

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

file_path = '../input/nba-games/games.csv'  #change path accordingly

df = pd.read_csv(file_path)

df.head()

hreb= df["REB_home"]

areb=df["REB_away"]

h3pct= df["FG3_PCT_home"]

a3pct=df["FG3_PCT_away"]

has= df["AST_home"]

aas=df["AST_away"]

hfg= df["FG_PCT_home"]

afg = df["FG_PCT_away"]

print("assist to fg%",has.corr(hfg))

homewon=df["HOME_TEAM_WINS"]

win3=0

winreb=0

wina=0

winfg=0

#best combos (who to pay)

win3reb=0 #3 and reb

winreba= 0 #reb and assists (C/PF and PG)

win3a=0 #3 and assists

winfga=0 # fg and assists

win3fg=0 #shooters shoot

winfgreb=0 # fg and reb

#dream team

win3reba=0 # 3 and reb and assists

winall=0 # all four stats

win3fg=0 # threes and fg

win3a=0 # threes and assists

winreba=0 # rebounds and assists

winrebfg=0 # rebounds and fieldgoals

worse3=0 #worst 3pct won

onlyreb=0

only3=0

onlyfg=0

onlya=0

noreband3=0

norebandfg=0

norebanda=0

gamecount=df.size

print("Compounding stats from ",df.size, "NBA games from 2004 to 2020")

howmany=afg.size -1

for a in range(0,howmany):

    #print(a)

   # print(has[a])

    betterreb= 0

    better3 = 0

    bettera=0

    betterfg=0

    #print(homewon[a])

   # print(homewon[a]==0)

    if homewon[a]>0:

        betterreb= hreb[a] > areb[a]

        better3 = h3pct[a] > a3pct[a]

        bettera = has[a] > aas[a]

        betterfg = hfg[a] > afg[a]

    else:

        betterreb= hreb[a] < areb[a]

        better3 = h3pct[a] < a3pct[a]

        bettera = has[a] < aas[a]

        betterfg = hfg[a] < afg[a]

    

    if betterreb:

        winreb=winreb + 1

    if better3:

        win3=win3 + 1

    if bettera:

        wina=wina+1

    if betterfg:

        winfg=winfg+1

    if (better3==0):

        worse3=worse3+1

    if (bettera==0) and (betterreb==0):

        winreba = winreba + 1

    #best duos

    if (betterreb==1) and (better3==1) and (betterfg==0) and (bettera==0):

        win3reb=win3reb + 1

    if (betterreb==1) and (better3==0) and (betterfg==0) and (bettera==1):

        winreba=winreba + 1

    if (betterreb==0) and (better3==1) and (betterfg==1) and (bettera==0):

        win3fg=win3fg + 1

    if (betterreb==0) and (better3==1) and (betterfg==0) and (bettera==1):

        win3a=win3a + 1

    if (betterreb==1) and (better3==1) and (betterfg==0) and (bettera==0):

        win3reb=win3reb + 1

    #without rebounds

    if (betterreb==0) and better3:

        noreband3=noreband3 +1

    if (betterreb==0) and betterfg:

        norebandfg=norebandfg +1

    if (betterreb==0) and bettera:

        norebanda=norebanda +1

    #only stats

    if (betterreb==1) and (better3==0) and (betterfg==0) and (bettera==0):

        onlyreb=onlyreb + 1

    if (betterreb==0) and (better3==1) and (betterfg==0) and (bettera==0):

        only3=only3 + 1

    if (betterreb==0) and (better3==0) and (betterfg==1) and (bettera==0):

        onlyfg=onlyfg + 1

    if (betterreb==0) and (better3==0) and (betterfg==0) and (bettera==1):

        onlya=onlya + 1

print("The team with better 3's had a ",win3/howmany," chance of winning")

print("The team with better assists had a ",wina/howmany," chance of winning")

print("The team with better rebounds had a ",winreb/howmany," chance of winning")

print("The team with better field goal had a ",winfg/howmany," chance of winning")

print("The team with worse 3's had a ",worse3/howmany," chance of winning")

print("The team with more assists and rebounds had a ",1 - winreba/howmany," chance of winning")

print("The team with only better rebounds had a ",onlyreb/howmany," chance of winning")

print("The team with only better 3's had a ",only3/howmany," chance of winning")

print("The team with only better field goal pct had a ",onlyfg/howmany," chance of winning")

print("The team with only better assists had a ",onlya/howmany," chance of winning")

print("The team losing rebounds but winning 3 pct had a ", noreband3/howmany," chance of winning")

print("The team losing rebounds but winning fg pct had a ", norebandfg/howmany," chance of winning")

print("The team losing rebounds but winning assists had a ", norebanda/howmany," chance of winning")

print("Deciding which two max salaries to get...")

print("With rebounds and assists you get ", winreba/howmany)

print("With three's and field goals ", win3fg/howmany)

print("With three and assists", win3a/howmany)

print("With rebounds and threes",win3reb/howmany)

print("Concluding statisitics....")

print("Board man gets paid")



#game details (indivudal)

file_path = '../input/nba-games/games_details.csv'  #change path accordingly

df = pd.read_csv(file_path)

df = df[pd.notnull(df['PLUS_MINUS'])];

player3pct=df["FG3_PCT"]

playerboards=df["REB"]

plusminus=df["PLUS_MINUS"]

blocks=df["BLK"]

steals=df["STL"]

assists=df["AST"]

defense=blocks+steals

#print(defense)

fgpct=df["FG_PCT"]

correlationreb = playerboards.corr(plusminus)

correlation3 = player3pct.corr(plusminus)

correlationdef = defense.corr(plusminus)

correlationassists = assists.corr(plusminus)

correlationfg=fgpct.corr(plusminus)

print(correlationreb)

print(correlation3)

print(correlationdef)

print(correlationassists)

print(correlationfg)
