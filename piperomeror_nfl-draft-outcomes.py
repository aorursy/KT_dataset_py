import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import os

#print(os.listdir("../input"))
df = pd.read_csv("../input/nfl_draft.csv")

print(df.info())

print('--------------------------------------')

print('Número de Registros:',df.shape[0])

print('Número de Variables:',df.shape[1])
df.head(5)
print('Hay',df.Player.duplicated().sum(),'registros duplicados en la variable Player')

print('Hay',df.Player_Id.duplicated().sum(),'registros duplicados en la variable Player_Id')

print('Hay',df[['Player_Id','Player']].duplicated().sum(),'registros duplicados en la variables Player y Player_Id')
compl = pd.DataFrame({'Porcentaje de Completitud':df.count()*100/len(df)}).round(2)

compl['indexx'] = compl.index.values.tolist()

print(compl)
compl = compl.sort_values(['Porcentaje de Completitud'],ascending=False).reset_index(drop=True)

plt.figure(figsize=(15,15))

sns.barplot(x=compl["Porcentaje de Completitud"],y=compl.indexx,palette='Blues_d')

plt.axvline(x=compl.mean()[0],linestyle='--',color='firebrick',label='Completitud Promedio: NFL Draft Outcomes')

plt.xlabel("Porcentaje de Completitud",fontsize=15)

plt.ylabel("Variable",fontsize=15)

plt.title("Completitud de la Base: NFL Draft Outcomes",fontsize=15)

plt.show()
sns.clustermap(df.corr(), center=0, cmap="vlag",

               linewidths=.75, figsize=(13, 13))
df = df[compl[compl['Porcentaje de Completitud'] > compl.mean()[0]].indexx.tolist()]

print('Variables con Completitud Superior a la Media:',df.shape[1])

print('Estas son:')

for i in df.columns.tolist():

    print(i)
df.describe()
sns.clustermap(df.corr(), center=0, cmap="vlag",

               linewidths=.75, figsize=(10, 10))
sns.set(style="ticks")

sns.jointplot(df.First4AV, df.G, kind="hex", color="#4CB391")

sns.jointplot(df.First4AV, df.CarAV, kind="hex", color="#4CB391")

sns.jointplot(df.DrAV, df.G, kind="hex", color="#4CB391")

sns.jointplot(df.DrAV, df.CarAV, kind="hex", color="#4CB391")
team = ["Tampa Bay Buccaneers", "Tennessee Titans", "Jacksonville Jaguars", "Oakland Raiders", "Washington Redskins", "New York Jets","Chicago Bears", "Atlanta Falcons", "New York Giants", "St. Louis Rams", "Minnesota Vikings", "Cleveland Browns", "New Orleans Saints", "Miami Dolphins", "San Diego Chargers", "Houston Texans", "San Franciso 49ers", "Kansas City Chiefs", "Phildelphia   Eagles", "Cincinatti Bengals", "Pittsburg Steelers", "Denver Broncos", "Arizona Cardinals", "Carolina Panthers", "Baltimore Ravens", "Dallas   Cowboys", "Detroit Lions", "Indianapolis Colts", "Green Bay Packers", "New England Patriots", "Buffalo Bills", "Seattle Seahawks", "Los   Angeles/St. Louis Rams", "Los Angeles Raiders/Oakland Raiders", "Phoenix/Arizona Cardinals"]

Tm = ["TAM", "TEN", "JAX", "OAK", "WAS", "NYJ", "CHI", "ATL", "NYG", "STL", "MIN", "CLE", "NOR", "MIA", "SDG", "HOU", "SFO", "KAN",  "PHI", "CIN", "PIT", "DEN", "ARI", "CAR", "BAL", "DAL", "DET",  "IND", "GNB", "NWE", "BUF", "SEA", "RAM", "RAI", "PHO"]



team = pd.DataFrame({'Team': team, 'Tm':Tm})

df = pd.merge(df, team,'left','Tm')

del team

df.head(4)
pac12 = ["Stanford", "California", "Arizona St.", "Arizona", "Washington", 

          "Washington St.", "Oregon", "Oregon St.", "USC", "UCLA", "Utah", "Colorado"]

    

big12 = ["Oklahoma", "Oklahoma St.", "TCU", "Baylor", "Iowa St.", "Texas", "Kansas", 

        "Kansas St.", "West Virginia", "Texas Tech"]



b1g = ["Northwestern", "Michigan", "Michigan St.", "Iowa", "Ohio St.", "Purdue", 

      "Indiana", "Rutgers", "Illinois", "Minnesota", "Penn St.", "Nebraska", "Maryland", 

      "Wisconsin"]



acc = ["Florida St.", "Syracuse", "Miami", "North Carolina", "North Carolina St.", 

      "Duke", "Virginia", "Virginia Tech", "Boston College", "Clemson", "Wake Forest",

      "Pittsburgh", "Louisville", "Louisville", "Georgia Tech"]



sec = ["Alabama", "Georgia", "Vanderbilt", "Kentucky", "Florida", "Missouri", 

      "Mississippi", "Mississippi St.", "Texas A&M", "Louisiana St.", "Arkansas", 

      "Auburn", "South Carolina", "Tennessee"]

conf = []

for i in range(len(df)): 

        if df.loc[i]["College/Univ"] in pac12: 

            conf.append("Pac 12")

            

        elif df.loc[i]["College/Univ"] in big12:

            conf.append("Big 12")

            

        elif df.loc[i]["College/Univ"] in b1g:

            conf.append("Big 10")

            

        elif df.loc[i]["College/Univ"] in acc:

            conf.append("ACC")

            

        elif df.loc[i]["College/Univ"] in sec: 

            conf.append("SEC")

        

        else: 

            conf.append("Not Power 5")

            

df['CFB_Conference'] = conf

df.head(5)
sns.set(style ="dark")

ax = sns.countplot(x = conf, palette='Blues_d')

ax.set_title(label='Frecuencia', fontsize=20)
pos = ["Quarterback", "Linebacker", "Wide Receiver", "Tackle", "Defensive End", "Running Back", "Defensive Back", "Defensive Tackle", "Center", "Guard", "Tight End", "Fullback", "Punter", "Long Snapper", "Kicker"]

pos1 = ["QB", "LB", "WR", "T", "DE", "RB", "DB", "DT", "C", "G", "TE", "FB", "P", "LS", "K"]

line = ["Offense", "Defense", "Offense", "Offense", "Defense", "Offense", "Defense", "Defense", "Offense", "Offense", "Offense", "Offense", "Special Teams", "Special Teams", "Special Teams"]

score = ["Yes", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "Yes"]

print('Los tres vectores tienen el mismo largo?    ---->   ',len(pos) == len(pos1) == len(line) == len(score))
posi = pd.DataFrame({'Pos': pos1, 'Position':pos,'Line':line,'Score':score})

df = pd.merge(df, posi,'left','Pos')

del posi

df = df[pd.notnull(df['Position'])]

df.head(5)
df = df[pd.notnull(df['First4AV'])]

df = df[pd.notnull(df['Rnd'])]

df = df[pd.notnull(df['Age'])]

df = df[pd.notnull(df['CarAV'])]

df = df[pd.notnull(df['G'])]

df = df[pd.notnull(df['DrAV'])]



f, axes = plt.subplots(2,3, figsize=(20, 12))

sns.distplot( df["First4AV"], ax=axes[0,0])

sns.distplot( df["Rnd"], ax=axes[0,1])

sns.distplot( df["Age"], ax=axes[0,2])

sns.distplot( df["CarAV"], ax=axes[1,0])

sns.distplot( df["G"], ax=axes[1,1])

sns.distplot( df["DrAV"], ax=axes[1,2])
def make_scatter(df):

    feats = ('G','CarAV','DrAV','First4AV')

    

    for index, feat in enumerate(feats):

        plt.subplot(len(feats)/4+1, 4, index+1)

        ax = sns.regplot(x = 'Age', y = feat, data = df)



plt.figure(figsize = (20, 20))

plt.subplots_adjust(hspace = 0.4)



make_scatter(df)
compl = pd.DataFrame({'Frecuencia':pd.value_counts(df.Team),'indexx':pd.value_counts(df.Team).index.values.tolist()})

compl = compl.sort_values(['Frecuencia'],ascending=False).reset_index(drop=True)

plt.figure(figsize=(15,15))

sns.barplot(x=compl["Frecuencia"],y=compl.indexx,palette='Blues_d')

plt.axvline(x=compl.mean()[0],linestyle='--',color='firebrick',label='Frecuencia Media de Fichajes por Equipo en la NFL')

plt.xlabel("Frecuencia",fontsize=15)

plt.ylabel("Equipo",fontsize=15)

plt.title("Frecuencia de Fichajes por Equipo",fontsize=15)

plt.show()
compl = compl[:5]

compl    
df = pd.read_csv("../input/nfl_draft.csv")

pos = ["Quarterback", "Linebacker", "Wide Receiver", "Tackle", "Defensive End", "Running Back", "Defensive Back", "Defensive Tackle", "Center", "Guard", "Tight End", "Fullback", "Punter", "Long Snapper", "Kicker"]

pos1 = ["QB", "LB", "WR", "T", "DE", "RB", "DB", "DT", "C", "G", "TE", "FB", "P", "LS", "K"]

bla1 = ["Offense", "Defense", "Offense", "Offense", "Defense", "Offense", "Defense", "Defense", "Offense", "Offense", "Offense", "Offense", "Special Teams", "Special Teams", "Special Teams"]

bla = ["Yes", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "Yes"]



df_pos = pd.DataFrame({'Posición':pos, 'Pos':pos1,'ataque':bla1, 'Offense':bla})



dfdf = pd.merge(df,df_pos,'left','Pos')



dfdf = dfdf[dfdf.Posición.astype(str) != 'nan']



del df, df_pos, pos, pos1



pos = ["Tampa Bay Buccaneers", "Tennessee Titans", "Jacksonville Jaguars", "Oakland Raiders", "Washington Redskins", "New York Jets","Chicago Bears", "Atlanta Falcons", "New York Giants", "St. Louis Rams", "Minnesota Vikings", "Cleveland Browns", "New Orleans Saints", "Miami Dolphins", "San Diego Chargers", "Houston Texans", "San Franciso 49ers", "Kansas City Chiefs", "Phildelphia   Eagles", "Cincinatti Bengals", "Pittsburg Steelers", "Denver Broncos", "Arizona Cardinals", "Carolina Panthers", "Baltimore Ravens", "Dallas   Cowboys", "Detroit Lions", "Indianapolis Colts", "Green Bay Packers", "New England Patriots", "Buffalo Bills", "Seattle Seahawks", "Los   Angeles/St. Louis Rams", "Los Angeles Raiders/Oakland Raiders", "Phoenix/Arizona Cardinals"]

pos1 = ["TAM", "TEN", "JAX", "OAK", "WAS", "NYJ", "CHI", "ATL", "NYG", "STL", "MIN", "CLE", "NOR", "MIA", "SDG", "HOU", "SFO", "KAN",  "PHI", "CIN", "PIT", "DEN", "ARI", "CAR", "BAL", "DAL", "DET",  "IND", "GNB", "NWE", "BUF", "SEA", "RAM", "RAI", "PHO"]



df_pos = pd.DataFrame({'Team':pos, 'Tm':pos1})

dfdf = pd.merge(dfdf,df_pos,'left','Tm')



dfdf = dfdf[dfdf.Team.astype(str) != 'nan']



del df_pos, pos, pos1









dfdf = dfdf[dfdf.To.astype(str) != 'nan']

dfdf['Career_Length'] = abs(dfdf.Year - dfdf.To)+1



a = ['CIN','GNB','BUF','NWE','PIT','CHI']   # Equipos seleccionados



df0 = dfdf[dfdf.Tm == a[0]]

df1 = dfdf[dfdf.Tm == a[1]]

df2 = dfdf[dfdf.Tm == a[2]]

df3= dfdf[dfdf.Tm == a[3]]

df4 = dfdf[dfdf.Tm == a[4]]

df5 = dfdf[dfdf.Tm == a[5]]





dfdf = df0.append(df1.append(df2.append(df3.append(df4.append(df5)))))



variables = dfdf.columns.values.tolist()



dfdf = dfdf[dfdf.ataque != 'Special Teams']

dfdf.Offense = dfdf.ataque       



a = []

for i in dfdf.Offense:

    if i == 'Offense':

        a.append('Ofensivo')

    else:

        a.append('Defensivo')



dfdf.Offense = a







fig, ax = plt.subplots()

fig.set_size_inches(17, 9)

ax = sns.violinplot(y="Career_Length", x="Team",hue='Offense' , data=dfdf,palette='Blues_d',split=True,linewidth=3);

ax.set_title(label='Distribución de la Duración de la Carrera en la NFL por Equipo y Tipo de Jugador (Draft Picks 1985-2015)', fontsize=20);

plt.ylabel('Duración de la Carrera Profesional en la NFL (Años)',fontsize=15)

plt.xlabel('Equipo',fontsize=15)

plt.axhline(y=4.45, label='Años Promedio de Carrera en la NFL', linestyle='--', color='firebrick')   #3,45 en promedio mas 1 año porque en promedio son mas jovenes que los fichajes normales

ax.legend()
print('Fin...')