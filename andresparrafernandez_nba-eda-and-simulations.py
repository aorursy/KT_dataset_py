# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split



from IPython.display import display, Markdown



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
games = pd.read_csv("/kaggle/input/nba-games/games.csv")

teams = pd.read_csv("/kaggle/input/nba-games/teams.csv")

games_details = pd.read_csv("/kaggle/input/nba-games/games_details.csv")

players = pd.read_csv("/kaggle/input/nba-games/players.csv")

ranking = pd.read_csv("/kaggle/input/nba-games/ranking.csv")
games.columns
teams.head()
games.info()
games = games.dropna()



games.info()
games.describe()
winning_teams = np.where(games['HOME_TEAM_WINS'] == 1, games['HOME_TEAM_ID'], games['VISITOR_TEAM_ID'])

winning_teams = pd.DataFrame(winning_teams, columns=['TEAM_ID'])

winning_teams = winning_teams.merge(teams[['TEAM_ID', 'NICKNAME']], on='TEAM_ID')['NICKNAME'].value_counts().to_frame().reset_index()

winning_teams.columns = ['TEAM NAME', 'Number of wins']



sns.barplot(winning_teams['Number of wins'], winning_teams['TEAM NAME'])







## credits to https://www.kaggle.com/nathanlauga/nba-games-eda-let-s-dive-into-the-data
sns.set_palette("rocket")



plt.hist(games["PTS_home"],bins = int(np.sqrt(len(games["PTS_home"]))))

plt.xlabel("Number of points scored by HOME TEAM")

plt.ylabel("Number of games ")

plt.show()

mean_ptsh = np.mean(games["PTS_home"])

std_ptsh=np.std(games["PTS_home"])



print ("mean:",mean_ptsh,"std:",std_ptsh)



plt.hist(games["PTS_away"],bins = int(np.sqrt(len(games["PTS_away"]))))

plt.xlabel("Number of points scored by AWAY TEAM")

plt.ylabel("Number of games ")

plt.show()

mean_ptsa = np.mean(games["PTS_away"])

std_ptsa=np.std(games["PTS_away"])



print ("mean:",mean_ptsa,"std:",std_ptsa)
plt.hist(games["AST_home"],bins = int(np.sqrt(len(games["AST_home"]))))

plt.xlabel("Number of assist scored by HOME TEAM")

plt.ylabel("Number of games ")

plt.show()



mean_asth = np.mean(games["AST_home"])

std_asth=np.std(games["AST_home"])



print ("mean:",mean_asth,"std:",std_asth)



plt.hist(games["AST_away"],bins = int(np.sqrt(len(games["AST_away"]))))

plt.xlabel("Number of assist scored by AWAY TEAM")

plt.ylabel("Number of games ")

plt.show()



mean_asta = np.mean(games["AST_away"])

std_asta=np.std(games["AST_away"])



print ("mean:",mean_asta,"std:",std_asta)
plt.hist(games["REB_home"],bins = int(np.sqrt(len(games["REB_home"]))))

plt.xlabel("Number of rebounds taken by HOME TEAM")

plt.ylabel("Number of games ")

plt.show()



mean_rebh = np.mean(games["REB_home"])

std_rebh=np.std(games["REB_home"])



print ("mean:",mean_rebh,"std:",std_rebh)



plt.hist(games["REB_away"],bins = int(np.sqrt(len(games["REB_away"]))))

plt.xlabel("Number of rebounds taken by AWAY TEAM")

plt.ylabel("Number of games ")

plt.show()



mean_reba = np.mean(games["REB_away"])

std_reba=np.std(games["REB_away"])



print ("mean:",mean_reba,"std:",std_reba)
def ecdf(data):

    

    n = len(data)



    

    x = np.sort(data)



    

    y = np.arange(1, 1+n) / n



    return x, y

## credits to datacamp
x_ptsh, y_ptsh = ecdf(games["PTS_home"])

x_ptsh1, y_ptsh1 = ecdf(np.random.normal(mean_ptsh,std_ptsh,size=100000))

plt.plot(x_ptsh,y_ptsh,marker=".",linestyle="none")

plt.plot(x_ptsh1,y_ptsh1,marker=".",linestyle="none")

plt.xlabel("Points scored by HOME TEAM")

plt.ylabel("ECDF")

plt.show()





x_ptsa, y_ptsa = ecdf(games["PTS_away"])

x_ptsa1, y_ptsa1 = ecdf(np.random.normal(mean_ptsa,std_ptsa,size=100000))

plt.plot(x_ptsa,y_ptsa,marker=".",linestyle="none")

plt.plot(x_ptsa1,y_ptsa1,marker=".",linestyle="none")

plt.xlabel("Points scored by AWAY TEAM")

plt.ylabel("ECDF")

plt.show()







x_asth, y_asth = ecdf(games["AST_home"])

x_asth1, y_asth1 = ecdf(np.random.normal(mean_asth,std_asth,size=100000))

plt.plot(x_asth,y_asth,marker=".",linestyle="none")

plt.plot(x_asth1,y_asth1,marker=".",linestyle="none")

plt.xlabel("Assist scored by HOME TEAM")

plt.ylabel("ECDF")

plt.show()





x_asta, y_asta = ecdf(games["AST_away"])

x_asta1, y_asta1 = ecdf(np.random.normal(mean_asta,std_asta,size=100000))

plt.plot(x_asta,y_asta,marker=".",linestyle="none")

plt.plot(x_asta1,y_asta1,marker=".",linestyle="none")

plt.xlabel("Assist scored by AWAY TEAM")

plt.ylabel("ECDF")

plt.show()

x_rebh, y_rebh = ecdf(games["REB_home"])

x_rebh1, y_rebh1 = ecdf(np.random.normal(mean_rebh,std_rebh,size=100000))

plt.plot(x_rebh,y_rebh,marker=".",linestyle="none")

plt.plot(x_rebh1,y_rebh1,marker=".",linestyle="none")

plt.xlabel("Rebounds taken by HOME TEAM")

plt.ylabel("ECDF")

plt.show()





x_reba, y_reba = ecdf(games["REB_away"])

x_reba1, y_reba1 = ecdf(np.random.normal(mean_reba,std_reba,size=100000))

plt.plot(x_reba,y_reba,marker=".",linestyle="none")

plt.plot(x_reba1,y_reba1,marker=".",linestyle="none")

plt.xlabel("Rebounds taken by AWAY TEAM")

plt.ylabel("ECDF")

plt.show()

years = [2018,2019]



gamestt = games[games["SEASON"].isin(years)]



winner = gamestt["HOME_TEAM_WINS"]



gamestt = gamestt.drop(columns=["TEAM_ID_home","TEAM_ID_away","GAME_STATUS_TEXT"])





gamestt["PTS_home"] = (gamestt["PTS_home"]).astype(int)

gamestt["PTS_away"] = (gamestt["PTS_away"]).astype(int)

gamestt["AST_home"] = (gamestt["AST_home"]).astype(int)

gamestt["AST_away"] = (gamestt["AST_away"]).astype(int)

gamestt["REB_home"] = (gamestt["REB_home"]).astype(int)

gamestt["REB_away"] = (gamestt["REB_away"]).astype(int)



gamestt.info()

forrep = teams.set_index("TEAM_ID")["ABBREVIATION"].to_dict()



print(forrep)





gamestt["HOME_TEAM_ID"] = gamestt["HOME_TEAM_ID"].replace(forrep)

gamestt["VISITOR_TEAM_ID"] = gamestt["VISITOR_TEAM_ID"].replace(forrep)







(gamestt.head())
gamestt["GAME_DATE_EST"]=pd.to_datetime(gamestt["GAME_DATE_EST"])

gamestts = gamestt.set_index(["GAME_ID"])

gamestts = gamestts.sort_index(axis=0)

gamestts
fig, ax = plt.subplots()



fig.set_size_inches(11.7, 8.27)



sns.boxplot(x="HOME_TEAM_ID",y="PTS_home",data=gamestts)







plt.xlabel("HOME TEAM")

plt.xticks(rotation = 90)

plt.ylabel("PTS SCORED ")





plt.show()



fig, ax = plt.subplots()



fig.set_size_inches(11.7, 8.27)



sns.boxplot(x="VISITOR_TEAM_ID",y="PTS_away",data=gamestts)







plt.xlabel("AWAY TEAM")

plt.xticks(rotation = 90)

plt.ylabel("PTS SCORED ")





plt.show()
fig, ax = plt.subplots()



fig.set_size_inches(11.7, 8.27)



sns.boxplot(x="HOME_TEAM_ID",y="AST_home",data=gamestts)







plt.xlabel("HOME TEAM")

plt.xticks(rotation = 90)

plt.ylabel("AST SCORED ")





plt.show()



fig, ax = plt.subplots()



fig.set_size_inches(11.7, 8.27)



sns.boxplot(x="VISITOR_TEAM_ID",y="AST_away",data=gamestts)







plt.xlabel("AWAY TEAM")

plt.xticks(rotation = 90)

plt.ylabel("AST SCORED ")





plt.show()
fig, ax = plt.subplots()



fig.set_size_inches(11.7, 8.27)



sns.boxplot(x="HOME_TEAM_ID",y="REB_home",data=gamestts)







plt.xlabel("HOME TEAM")

plt.xticks(rotation = 90)

plt.ylabel("REB GRABBED ")





plt.show()



fig, ax = plt.subplots()



fig.set_size_inches(11.7, 8.27)



sns.boxplot(x="VISITOR_TEAM_ID",y="REB_away",data=gamestts)







plt.xlabel("AWAY TEAM")

plt.xticks(rotation = 90)

plt.ylabel("REB GRABBED ")





plt.show()
def pearson_r(x, y):

    """Compute Pearson correlation coefficient between two arrays."""

    # Compute correlation matrix: corr_mat

    corr_mat=np.corrcoef(x,y)



    # Return entry [0,1]

    return corr_mat[0,1]

sns.scatterplot(x="PTS_home",y="AST_home",data=gamestts,alpha=0.5)



plt.xlabel("POINTS SCORED (HOME TEAMS)")

plt.xticks(rotation = 90)

plt.ylabel("ASSIST SCORED (HOME TEAMS)")



plt.show()



print("Pearson correlation coefficient;",pearson_r(gamestts["PTS_home"],gamestts["AST_home"]))





sns.scatterplot(x="PTS_away",y="AST_away",data=gamestts,alpha=0.5)



plt.xlabel("POINTS SCORED (AWAY TEAMS)")

plt.xticks(rotation = 90)

plt.ylabel("ASSIST SCORED (AWAY TEAMS)")



plt.show()

print("Pearson correlation coefficient;",pearson_r(gamestts["PTS_away"],gamestts["AST_away"]))
sns.scatterplot(x="PTS_home",y="REB_home",data=gamestts,alpha=0.5)



plt.xlabel("POINTS SCORED (HOME TEAMS)")

plt.xticks(rotation = 90)

plt.ylabel("REBOUNDS GRABBED (HOME TEAMS)")



plt.show()



print("Pearson correlation coefficient;",pearson_r(gamestts["PTS_home"],gamestts["REB_home"]))





sns.scatterplot(x="PTS_away",y="REB_away",data=gamestts,alpha=0.5)



plt.xlabel("POINTS SCORED (AWAY TEAMS)")

plt.xticks(rotation = 90)

plt.ylabel("REBOUNDS GRABBED (AWAY TEAMS)")



plt.show()

print("Pearson correlation coefficient;",pearson_r(gamestts["PTS_away"],gamestts["REB_away"]))
sns.scatterplot(x="AST_home",y="REB_home",data=gamestts,alpha=0.5)



plt.xlabel("ASSIST SCORED (HOME TEAMS)")

plt.xticks(rotation = 90)

plt.ylabel("REBOUNDS GRABBED (HOME TEAMS)")



plt.show()



print("Pearson correlation coefficient;",pearson_r(gamestts["AST_home"],gamestts["REB_home"]))





sns.scatterplot(x="AST_away",y="REB_away",data=gamestts,alpha=0.5)



plt.xlabel("ASSIST SCORED (AWAY TEAMS)")

plt.xticks(rotation = 90)

plt.ylabel("REBOUNDS GRABBED (AWAY TEAMS)")



plt.show()

print("Pearson correlation coefficient;",pearson_r(gamestts["AST_away"],gamestts["REB_away"]))
team_list = teams["ABBREVIATION"]

results_dic = {}

for i in team_list:

 results_dic[str(i)] = []







for i in range(len(gamestts)) : 

  for j in team_list:

    

    

    if (gamestts.iloc[i,1])==j:

        results_dic[j].append(gamestts.iloc[i,:])

    elif (gamestts.iloc[i,2])==j:

        results_dic[j].append(gamestts.iloc[i,:])
results = {}



for i in team_list : 

    results[i]=pd.DataFrame(results_dic[i])  

  

    

(results["LAL"])

sns.set(style="darkgrid")



plot_list=["PTS_home","AST_home","REB_home","PTS_away","AST_away","REB_away"]



fig, axes =plt.subplots(2,3, figsize=(25,10), sharex=True)



for j in range(2):

    for i,ax in enumerate(axes.flat):

        

        sns.boxplot(x="HOME_TEAM_WINS",y=plot_list[i], data=gamestts, ax=ax)
for i in range(6):

    print("Pcorrcoef between HOME_TEAM_WINS and", plot_list[i],pearson_r(gamestts["HOME_TEAM_WINS"],gamestts[plot_list[i]]) )



    

print("percentage of times where the HOME TEAM WON between 2016 and 2018 ?: A://", np.mean(gamestts["HOME_TEAM_WINS"]))    
results_home = {}

results_away = {}



for i in team_list:

 results_home[str(i)] =[]

 results_away[str(i)] =[]



for i in team_list : 

    for j in range(len(results[i])):

        if results[i].iloc[j,1]== i :

            results_home[i].append(results[i].iloc[j,:])

        elif results[i].iloc[j,2]== i :

            results_away[i].append(results[i].iloc[j,:])
results_homedf = {}

results_awaydf = {}



for i in team_list : 

    results_homedf[i]=pd.DataFrame(results_home[i])  

    results_awaydf[i]=pd.DataFrame(results_away[i])

  
sns.set_palette("Paired")



fig, axes =plt.subplots(1,2, figsize=(10,4), sharex=True)



sns.boxplot( y="PTS_home", data=results_homedf["LAL"], ax=axes[0])

sns.boxplot( y="PTS_away", data=results_homedf["LAL"], ax=axes[1])



fig, axes =plt.subplots(1,2, figsize=(10,4), sharex=True)



sns.boxplot( y="AST_home", data=results_homedf["LAL"], ax=axes[0])

sns.boxplot( y="AST_away", data=results_homedf["LAL"], ax=axes[1])



fig, axes =plt.subplots(1,2, figsize=(10,4), sharex=True)



sns.boxplot(y="REB_home", data=results_homedf["LAL"], ax=axes[0])

sns.boxplot(y="REB_away", data=results_homedf["LAL"], ax=axes[1])
fig, axes =plt.subplots(1,2, figsize=(10,4), sharex=True)



sns.boxplot( y="PTS_home", data=results_awaydf["TOR"], ax=axes[0])

sns.boxplot( y="PTS_away", data=results_awaydf["TOR"], ax=axes[1])



fig, axes =plt.subplots(1,2, figsize=(10,4), sharex=True)



sns.boxplot( y="AST_home", data=results_awaydf["TOR"], ax=axes[0])

sns.boxplot( y="AST_away", data=results_awaydf["TOR"], ax=axes[1])



fig, axes =plt.subplots(1,2, figsize=(10,4), sharex=True)



sns.boxplot(y="REB_home", data=results_awaydf["TOR"], ax=axes[0])

sns.boxplot(y="REB_away", data=results_awaydf["TOR"], ax=axes[1])
sns.set_palette("rocket")



sns.kdeplot(results_homedf["LAL"]["PTS_home"])

plt.xlabel("Number of points scored by LAL WHEN HOME")

plt.ylabel("Number of games ")

plt.show()

mean_ptsh = np.mean(results_homedf["LAL"]["PTS_home"])

std_ptsh=np.std(results_homedf["LAL"]["PTS_home"])



print ("mean:",mean_ptsh,"std:",std_ptsh)



sns.kdeplot(results_awaydf["TOR"]["PTS_away"])

plt.xlabel("Number of points scored by TOR WHEN AWAY")

plt.ylabel("Number of games ")

plt.show()

mean_ptsa = np.mean(results_awaydf["TOR"]["PTS_away"])

std_ptsa=np.std(results_awaydf["TOR"]["PTS_away"])



print ("mean:",mean_ptsa,"std:",std_ptsa)
x_ptsh, y_ptsh = ecdf(results_homedf["LAL"]["PTS_home"])

x_ptsh1, y_ptsh1 = ecdf(np.random.normal(mean_ptsh,std_ptsh,size=100000))

plt.plot(x_ptsh,y_ptsh,marker=".",linestyle="none")

plt.plot(x_ptsh1,y_ptsh1,marker=".",linestyle="none")

plt.xlabel("Points scored by LAL WHEN HOME")

plt.ylabel("ECDF")

plt.show()





x_ptsa, y_ptsa = ecdf(results_awaydf["TOR"]["PTS_away"])

x_ptsa1, y_ptsa1 = ecdf(np.random.normal(mean_ptsa,std_ptsa,size=100000))

plt.plot(x_ptsa,y_ptsa,marker=".",linestyle="none")

plt.plot(x_ptsa1,y_ptsa1,marker=".",linestyle="none")

plt.xlabel("Points scored by TOR WHEN AWAY")

plt.ylabel("ECDF")

plt.show()
LAL_WINS = 0



mean_ptsh = np.mean(results_homedf["LAL"]["PTS_home"])

std_ptsh = np.std(results_homedf["LAL"]["PTS_home"])

mean_ptsh1 = np.mean(results_homedf["LAL"]["PTS_away"])

std_ptsh1 = np.std(results_homedf["LAL"]["PTS_away"])

mean_ptsa = np.mean(results_awaydf["TOR"]["PTS_away"])

std_ptsa=np.std(results_awaydf["TOR"]["PTS_away"])

mean_ptsa1 = np.mean(results_awaydf["TOR"]["PTS_home"])

std_ptsa1=np.std(results_awaydf["TOR"]["PTS_home"])





Points_LAL = (np.random.normal(mean_ptsh,std_ptsh,size=100000)+np.random.normal(mean_ptsh1,std_ptsh1,size=100000))/2

Points_TOR = (np.random.normal(mean_ptsa,std_ptsa,size=100000)+np.random.normal(mean_ptsa1,std_ptsa1,size=100000))/2



for i in range(100000):

    

    if Points_LAL[i] > Points_TOR[i]:

        

        LAL_WINS = LAL_WINS + 1

    

    

LAL_WINPER = LAL_WINS/100000  



print(LAL_WINPER*100)

def gamesim(HOMEID,AWAYID):



 H_WINS = 0



 mean_ptsh = np.mean(results_homedf[HOMEID]["PTS_home"])

 std_ptsh = np.std(results_homedf[HOMEID]["PTS_home"])

 mean_ptsh1 = np.mean(results_homedf[HOMEID]["PTS_away"])

 std_ptsh1 = np.std(results_homedf[HOMEID]["PTS_away"])

 mean_ptsa = np.mean(results_awaydf[AWAYID]["PTS_away"])

 std_ptsa=np.std(results_awaydf[AWAYID]["PTS_away"])

 mean_ptsa1 = np.mean(results_awaydf[AWAYID]["PTS_home"])

 std_ptsa1=np.std(results_awaydf[AWAYID]["PTS_home"])





 Points_H = (np.random.normal(mean_ptsh,std_ptsh,size=100000)+np.random.normal(mean_ptsh1,std_ptsh1,size=100000))/2

 Points_A = (np.random.normal(mean_ptsa,std_ptsa,size=100000)+np.random.normal(mean_ptsa1,std_ptsa1,size=100000))/2



 for i in range(100000):

    

      if Points_H[i] > Points_A[i]:

        

          H_WINS = H_WINS + 1

    

    

 H_WINPER = (H_WINS/100000)*100  



 return H_WINPER
gamesim("GSW","MEM")