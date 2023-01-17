import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/international-football-results-from-1872-to-2017/results.csv')

data.head()
data.info()
data["date"] = pd.to_datetime(data["date"])

data["date_month"] = data["date"].dt.month

data["date_month"].value_counts().sort_index()
plt.title("Number of game days per month")

sns.countplot(data=data,x="date_month" )

plt.show()
june = data[data["date_month"]==6]

june = june[june["tournament"].isin(june["tournament"].value_counts().head(7).index)]

plt.figure(figsize=(20,5))

sns.countplot(data=june,x="tournament")

plt.show()
def find_win(row):

    if row["home_score"]> row["away_score"]:

        return row["home_team"]

    elif row["home_score"]< row["away_score"]:

        return row["away_team"]

    else:

        return "draw"



def find_homeaway(row):

    if row["home_team"] == row["winning_team"]:

        return "Home"

    elif row["away_team"] == row["winning_team"]:

        return "Away"

    else:

        return "Draw"
data["winning_team"] = data.apply(find_win,axis=1)

data["winning_team2"] = data.apply(find_homeaway,axis=1)

data.head()
table = pd.pivot_table(data,index="winning_team2",values="tournament",aggfunc="count").reset_index()

P = table["tournament"].unique()
plt.title("Home away winning percentage")

plt.pie(P, 

        explode = (0, 0, 0.1),

        labels=["Away","Draw","Home"],

        shadow=True,

        autopct='%1.1f%%',

         )

plt.show()
data["total_score"] = data["home_score"] + data["away_score"]

data = data[["date","home_team","away_team","home_score","away_score","total_score","winning_team","winning_team2",

            "tournament","city","country","neutral"]]

data.head()
home_mean = data["home_score"].mean()

away_mean = data["away_score"].mean()

print("The average score on the home team is",home_mean)

print("The average score on the away team is",away_mean)
sns.scatterplot(x="home_score", y="away_score", data=data)

plt.show()
away_team = pd.DataFrame(data.groupby('away_team')['away_score'].sum().index)

away_score = pd.DataFrame(data.groupby('away_team')['away_score'].sum().values,columns=['Score'])

away_score_team=pd.concat([away_team,away_score],axis=1)

away_score_team = away_score_team.sort_values(by="Score",ascending = False).head(15)

home_team = pd.DataFrame(data.groupby('home_team')['home_score'].sum().index)

home_score = pd.DataFrame(data.groupby('home_team')['home_score'].sum().values,columns=['Score'])

home_score_team=pd.concat([home_team,home_score],axis=1)

home_score_team = home_score_team.sort_values(by="Score",ascending = False).head(15)
figure,((ax1),(ax2)) = plt.subplots(nrows=2,ncols=1)

figure.set_size_inches([20,10])

sns.barplot(data = away_score_team, x="away_team" , y="Score" , ax=ax2)

sns.barplot(data = home_score_team, x="home_team" , y="Score" , ax=ax1)

plt.show()
away_win = pd.DataFrame(data.groupby('away_team')['away_score'].count().values,columns=['win_count'])

home_win = pd.DataFrame(data.groupby('home_team')['home_score'].count().values,columns=['win_count'])



away_win_team=pd.concat([away_team,away_win],axis=1)

home_win_team=pd.concat([home_team,home_win],axis=1)



away_win_team = away_win_team.sort_values(by="win_count",ascending = False).head(15)

home_win_team = home_win_team.sort_values(by="win_count",ascending = False).head(15)
figure,((ax1),(ax2)) = plt.subplots(nrows=2,ncols=1)

figure.set_size_inches([20,10])



sns.barplot(data = away_win_team, x="away_team" , y="win_count" , ax=ax2)

sns.barplot(data = home_win_team, x="home_team" , y="win_count" , ax=ax1)

plt.show()
worldcup = data[data["tournament"].isin(["FIFA World Cup"])]

print(worldcup.shape)

worldcup.head()
home_team = pd.DataFrame(worldcup.groupby('home_team')['home_score'].sum().index)

home_score = pd.DataFrame(worldcup.groupby('home_team')['home_score'].sum().values,columns=['score'])

home_score_team= pd.concat([home_team,home_score],axis=1)



away_team = pd.DataFrame(worldcup.groupby('away_team')['away_score'].sum().index)

away_score = pd.DataFrame(worldcup.groupby('away_team')['away_score'].sum().values,columns=['score'])

away_score_team= pd.concat([away_team,away_score],axis=1)
home_team = pd.DataFrame(worldcup.groupby('home_team')['winning_team'].count().index)

home_win = pd.DataFrame(worldcup.groupby('home_team')['winning_team'].count().values,columns=['win'])

home_win_team= pd.concat([home_team,home_win],axis=1)



away_team = pd.DataFrame(worldcup.groupby('away_team')['winning_team'].count().index)

away_win = pd.DataFrame(worldcup.groupby('away_team')['winning_team'].count().values,columns=['win'])

away_win_team= pd.concat([away_team,away_win],axis=1)
worldcup_score = home_score_team.merge(away_score_team,left_on="home_team",right_on="away_team").drop("away_team",axis=1)

worldcup_score["total_score"] = worldcup_score["score_x"] + worldcup_score["score_y"]

worldcup_score = worldcup_score.sort_values(by="total_score",ascending=False).head(15)

worldcup_score
worldcup_win= home_win_team.merge(away_win_team,left_on="home_team",right_on="away_team").drop("away_team",axis=1)

worldcup_win["total_win"] = worldcup_win["win_x"] + worldcup_win["win_y"]

worldcup_win = worldcup_win.sort_values(by="total_win",ascending=False).head(15)

worldcup_win
plt.figure(figsize=(15,5))

plt.title('Number of scores by country')

sns.barplot(data = worldcup_score , x="home_team", y="total_score")

plt.show()
plt.figure(figsize=(15,5))

plt.title('Number of wins by country')

sns.barplot(data = worldcup_win , x="home_team", y="total_win")

plt.show()
merged = pd.merge(worldcup_win,worldcup_score)

merged = merged[["home_team","total_win","total_score"]]

merged.corr()
sns.heatmap(merged.corr(),annot=True)

plt.show()
tour = pd.DataFrame(data.groupby("tournament")["total_score"].mean()).reset_index().sort_values(by="total_score",ascending=False).head(10)

tour
plt.figure(figsize=(23,5))

plt.title("Average score by tournament")

sns.barplot(data=tour,x="tournament",y="total_score")

plt.show()
neu = pd.DataFrame(data.groupby("neutral")["total_score"].mean()).reset_index()
plt.title("Average score by neutral")

sns.barplot(data=neu,x="neutral",y="total_score")

plt.show()
country = pd.DataFrame(data["country"].value_counts()).head(15).reset_index()

country.columns = ["country","count"]
plt.figure(figsize=(23,7))

plt.title("Competition Places (National)")

sns.barplot(data=country,x="country",y="count")

plt.show()