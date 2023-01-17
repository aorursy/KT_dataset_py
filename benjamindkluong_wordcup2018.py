# Import libraries

import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from IPython.display import Image



from plotly.offline import iplot, init_notebook_mode

from geopy.geocoders import Nominatim

import plotly.plotly as py



import os

print(os.listdir("../input"))

df = pd.read_csv("../input/fifa-18-demo-player-dataset/CompleteDataset.csv", index_col=0)

df.head()
columns = ['Name','Age','Nationality','Overall','Value']

data = pd.DataFrame(df,columns=columns)

data.head()
data.info()
# Supporting function for converting string values into numbers

def str2number(amount):

    if amount[-1] == 'M':

        return float(amount[1:-1])*1000000

    elif amount[-1] == 'K':

        return float(amount[1:-1])*1000

    else:

        return float(amount[1:])

    

data['Value'] = data['Value'].apply(lambda x: str2number(x))

plt.figure(figsize=(16,8))

sns.set_style("whitegrid")

plt.title('Grouping players by Age', fontsize=30, fontweight='bold', y=1.05,)

plt.xlabel('Age',fontsize=25)

plt.ylabel('Count',fontsize=25)

sns.countplot(x="Age", data=data, palette="hls");

plt.show()
plt.figure(figsize=(16,8))

sns.set_style("whitegrid")

plt.title('Grouping players by Overall', fontsize=30, fontweight='bold', y=1.05,)

plt.xlabel('Number of players', fontsize=25)

plt.ylabel('Players Age', fontsize=25)

sns.countplot(x="Overall", data=data, palette="hls");

plt.show()
data.Overall.describe()
player_over_80 = len(data[data.Overall > 80])*100/len(data)

print('The percentage of players who have overall score greater than 80 is only {:.1f}%'.format(player_over_80))

print('Most players who play in Worldcup games have overall score obove 80.')
'''

a = data.Nationality.value_counts().reset_index()

a.columns=['Nationality','Count']

a[:20]

'''

top20_nation = data.groupby('Nationality').size().reset_index(name='Count').sort_values('Count',ascending = False)[:20]

top20_nation
plt.figure(figsize=(16,20))



countries = list(top20_nation.loc[::-1,'Nationality'])

pos = np.arange(len(countries))

count = list(top20_nation.loc[::-1,'Count'])



plt.barh(pos, count, align='center', alpha=.8)

plt.yticks(pos, countries, fontsize=25)

plt.xlabel('Count', fontsize=25)

plt.title('Number players by Countries', fontsize=30, fontweight='bold')

 

plt.show()
plt.figure(figsize=(16,16))

sns.set_style("whitegrid")

plt.title('Relationship between Value, Overall and Age', fontsize=30, fontweight='bold', y=1.05,)

plt.xlabel('Age', fontsize=25)

plt.ylabel('Overall', fontsize=25)



age = data["Age"].values

overall = data["Overall"].values

value = data["Value"].values



plt.scatter(age, overall, s = value/50000, edgecolors='black', color="red")

plt.show()
sns.pairplot(data[['Age','Overall','Value']])
data['AgeRange'] = pd.cut(data.Age, bins = [0,23,33,45],labels = ['Young','Mature','Old'])
# Use the 'hue' argument to provide a factor variable

sns.lmplot( x="Overall", y="Value", data=data, fit_reg=False, hue='AgeRange', legend=False)

 

# Move the legend to an empty part of the plot

plt.legend(loc='upper left')

plt.figure(figsize=(16,8))

plt.title("Percentages of Young, Mature, and Old Players", fontsize = 20, fontweight = 'bold')

labels = 'Mature','Young','Old'

plt.rcParams['font.size'] = 20.0

plt.pie(data.AgeRange.value_counts(), labels = labels, autopct='%1.1f%%', startangle=0)

plt.axis('equal')

plt.show()
def get_top20_players(country):

    top20players = data[data.Nationality == country].sort_values('Value',ascending = False)[:20]

    return top20players



top10_nation_list = top20_nation.Nationality[:10].tolist()



frames = []

for i in range(len(top10_nation_list)):

    temp_df = get_top20_players(top10_nation_list[i])

    frames.append(temp_df)

top_players_in_top10 = pd.concat(frames)

top_players_in_top10
plt.figure(figsize=(20,14))

sns.boxplot(x="Nationality",y = 'Value',data = top_players_in_top10)

plt.title("Value of Top 20 players in each Top 10 Countries", fontsize = 30, fontweight = 'bold')

plt.xlabel('Countries', fontsize=25)

plt.ylabel('Value', fontsize=25)

plt.show()
# define a function that get location of each players from a name list

def get_location(player_list,data):

    location=[]

    for idx,s in enumerate(data.Name):

        for player in player_list:

            if player in s:

                location.append(idx)

    return location

# Players who are younger than 20 will have 95% of their overall score

# Players who are older than 29 will have 98% of their overall score

def overall_adjusted_score(input_data):

    data = input_data.copy()

    data.loc[data.index[(data.Age < 20)],"Overall"]=data.loc[data.index[(data.Age < 20)],"Overall"]*0.95

    data.loc[data.index[(data.Age >29)],"Overall"]=data.loc[data.index[(data.Age >29)],"Overall"]*0.98

    return data.Overall.mean()
Image("../input/worldcup2018/FranceVsBelgium.png")
FrancePlayers = ["H. Lloris","B. Pavard","R. Varane","S. Umtiti","L. Hernandez","N. Kante","P. Pogba","K. Mbappe","A. Griezmann","B. Matuidi","O. Giroud"]

BelgiumPlayers = ["R. Lukaku","E. Hazard","M. Fellaini","K. De Bruyne","M. Dembele","A. Witsel","J. Vertonghen","V. Kompany","T. Alderweireld","N. Chadli","T. Courtois"]
all_france_players =data[data.Nationality == "France"]

France_lineups = all_france_players.iloc[get_location(FrancePlayers,all_france_players),:]

France_lineups

# Add missing players to France_lineups dataframe

France_lineups=France_lineups.append({'Name' : 'K. Mbappe' , 'Age' : 20, 'Nationality': 'France', 'Overall' : 88, 'Value' : 81000000} , ignore_index=True)

France_lineups=France_lineups.append({'Name' : 'L. Hernandez' , 'Age' : 22, 'Nationality': 'France', 'Overall' : 83, 'Value' : 29500000} , ignore_index=True)

France_lineups=France_lineups.append({'Name' : 'N. Kante' , 'Age' : 27, 'Nationality': 'France', 'Overall' : 89, 'Value' : 63000000} , ignore_index=True)

# complete France Lineups

France_lineups
# Belgium

all_belgium_players =data[data.Nationality == "Belgium"]

Belgium_lineups = all_belgium_players.iloc[get_location(BelgiumPlayers,all_belgium_players),:]

Belgium_lineups
# Add missing players to Belgium_lineups dataframe

Belgium_lineups=Belgium_lineups.append({'Name' : 'M. Dembele' , 'Age' : 30, 'Nationality': 'Belgium', 'Overall' : 82, 'Value' : 0} , ignore_index=True)

Belgium_lineups
print("Here is France's average age: {:.2f},comparing against Belgium's average age: {:.2f}.".format(France_lineups.Age.mean(),Belgium_lineups.Age.mean()))

print("We can see France players are younger than Belgium players on average. It means Belgium have more experiences than France. France players are youngers, so they have more strength.")
plt.figure(figsize=(16,8))

sns.set_style("whitegrid")

plt.title('France Players Age', fontsize=30, fontweight='bold', y=1.05,)

plt.xlabel('x', fontsize=25)

plt.ylabel('y', fontsize=25)

sns.countplot(x="Age", data=France_lineups, palette="hls");

plt.show()
plt.figure(figsize=(16,8))

sns.set_style("whitegrid")

plt.title('Belgium Players Age', fontsize=30, fontweight='bold', y=1.05,)

plt.xlabel('x', fontsize=25)

plt.ylabel('y', fontsize=25)

sns.countplot(x="Age", data=Belgium_lineups, palette="hls");

plt.show()
names = ['A. Witsel','M. Dembele']

mean = Belgium_lineups.Value.mean()

Belgium_lineups.loc[get_location(names,Belgium_lineups),'Value'] = mean

Belgium_lineups
print('Here is total value of France: €{:.2f}M, \nand here is total value of Belgium: €{:.2f}M'.format(France_lineups.Value.sum()/1000000,Belgium_lineups.Value.sum()/1000000))

France_Forward_Players = ["K. Mbappe","A. Griezmann","B. Matuidi","O. Giroud"]

Belgium_Forward_Players = ["R. Lukaku","E. Hazard","M. Fellaini","K. De Bruyne"]
France_Forward_Lineups=France_lineups.iloc[get_location(France_Forward_Players,France_lineups),:]

Belgium_Forward_Lineups=Belgium_lineups.iloc[get_location(Belgium_Forward_Players,Belgium_lineups),:]
print("With adjusted overall score for forward lineups, France's average Overall score: {:.2f},comparing against Belgium's average Overall score: {:.2f}.".format(overall_adjusted_score(France_Forward_Lineups),overall_adjusted_score(Belgium_Forward_Lineups)))
France_Midfielder_Players = ["N. Kante","P. Pogba"]

Belgium_Midfielder_Players = ["M. Dembele","A. Witsel"]
France_Midfielder_Lineups=France_lineups.iloc[get_location(France_Midfielder_Players,France_lineups),:]

Belgium_Midfielder_Lineups=Belgium_lineups.iloc[get_location(Belgium_Midfielder_Players,Belgium_lineups),:]
print("With adjusted overall score for midfielder lineups, France's average Overall score: {:.2f},comparing against Belgium's average Overall score: {:.2f}.".format(overall_adjusted_score(France_Midfielder_Lineups),overall_adjusted_score(Belgium_Midfielder_Lineups)))
France_Defender_Players = ["B. Pavard","R. Varane","S. Umtiti","L. Hernandez"]

Belgium_Defender_Players = ["J. Vertonghen","V. Kompany","T. Alderweireld","N. Chadli"]
France_Defender_Lineups=France_lineups.iloc[get_location(France_Defender_Players,France_lineups),:]

Belgium_Defender_Lineups=Belgium_lineups.iloc[get_location(Belgium_Defender_Players,Belgium_lineups),:]
print("With adjusted overall score for defender lineups, France's average Overall score: {:.2f},comparing against Belgium's average Overall score: {:.2f}.".format(overall_adjusted_score(France_Defender_Lineups),overall_adjusted_score(Belgium_Defender_Lineups)))
France_Goalkeeper_Players = ["H. Lloris"]

Belgium_Goalkeeper_Players = ["T. Courtois"]
France_Goalkeeper_Lineups=France_lineups.iloc[get_location(France_Goalkeeper_Players,France_lineups),:]

Belgium_Goalkeeper_Lineups=Belgium_lineups.iloc[get_location(Belgium_Goalkeeper_Players,Belgium_lineups),:]
print(France_Goalkeeper_Lineups)

print(Belgium_Goalkeeper_Lineups)
print("With unadjusted overall score, France's average Overall score: {:.2f},comparing against Belgium's average Overall score: {:.2f}.".format(France_lineups.Overall.mean(),Belgium_lineups.Overall.mean()))

print("Belgium's overall score is higher than France's overall score.")
print("With adjusted overall score, France's average Overall score: {:.2f},comparing against Belgium's average Overall score: {:.2f}.".format(overall_adjusted_score(France_lineups),overall_adjusted_score(Belgium_lineups)))

print("Belgium's overall score is higher than France's overall score.")
print("Total games: {}".format(24+19+30))

print("France winrate: {:.2f}%".format(24*100/73))

print("Belgium winrate: {:.2f}%".format(30*100/73))

print("Belgium has more chance to win than France.")
Image("../input/worldcup2018/FranceVsBelgiumResult.png")
Image("../input/worldcup2018/CroatiaVsEngland.png")
CroatiaPlayers = ["D. Subašić","S. Vrsaljko","D. Lovren","D. Vida","I. Strinić","M. Brozović","A. Rebić","L. Modrić","I. Rakitić","I. Perišić", "M. Mandžukić"]

EnglandPlayers = ["R. Sterling","H. Kane","D. Alli","J. Lingard","A. Young","K. Trippier","J. Henderson", "H. Maguire","J. Stones","K. Walker","J. Pickford"]
all_croatia_players =data[data.Nationality == "Croatia"]

Croatia_lineups = all_croatia_players.iloc[get_location(CroatiaPlayers,all_croatia_players),:]

Croatia_lineups
# Add missing players to Croatia_lineups dataframe

Croatia_lineups=Croatia_lineups.append({'Name' : 'D. Vida' , 'Age' : 29, 'Nationality': 'Croatia', 'Overall' : 80, 'Value' : 11500000} , ignore_index=True)

Croatia_lineups
# Remove D. Lovren who has overall score 58.  I will keep D. Lovren with overall score at 81.

Croatia_lineups.drop(index = 10)
all_england_players =data[data.Nationality == "England"]

England_lineups = all_england_players.iloc[get_location(EnglandPlayers,all_england_players),:]

England_lineups
England_lineups.drop(England_lineups.index[[7,12]])
print("Here is Croatia's average age: {:.2f},comparing against England's average age: {:.2f}.".format(Croatia_lineups.Age.mean(),England_lineups.Age.mean()))

print("We can see England players are younger than Croatia players on average. It means Croatia have more experiences than France. England players are youngers, so they have more strength.")
Image("../input/worldcup2018/FranceVsCroatia.png")