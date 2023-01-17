# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/fifa19/data.csv")
data.head()
data.shape
def country_list_player(country_name):
    country = data[data["Nationality"]==country_name][["Name", "Age", "Overall", "Potential", "Position", "Club", "Jersey Number"]]
    return country
print(country_list_player("Greece").shape)
country_list_player("Greece")
import matplotlib.pyplot as plt
import seaborn as sns
nation_ = data["Nationality"].value_counts()
nation_
index = data.index
percentage_of_diff_country = (nation_ / len(index)) * 100
sns.set(style="darkgrid", palette='Set1')
percentage_of_diff_country[:20].plot(kind="bar", figsize=(20,5), color='green')
plt.ylabel("Percentage");
col_name = []
for col in data.columns:
    col_name.append(col)
print(col_name)
def club_stat(name_of_club):
    club = data[data["Club"]==name_of_club][["Name", "Age", "Value", "Wage", "Special", "International Reputation"]].sort_values("International Reputation", ascending=False)
    return club
club_stat("FC Barcelona")
club_stat("Paris Saint-Germain")
def change_form_of_value(Value):
    change = Value.replace("€", "")
    if "M" in change:
        change = float(change.replace("M",""))*1000000
    elif "K" in change:
        change = float(change.replace("K", "")) * 1000
    return float(change)
data["Value"] = data["Value"].apply(lambda x: change_form_of_value(x))
data["Wage"] = data["Wage"].apply(lambda x: change_form_of_value(x))
print(data["Wage"].head())
print(data["Value"].head())
data["International Reputation"].fillna(data["International Reputation"].mean(), inplace=True)
data["Curve"].fillna(data["Curve"].mean(), inplace=True)
data["ShotPower"].fillna(data["ShotPower"].mean(), inplace=True)
data["Jumping"].fillna(data["Jumping"].mean(), inplace=True)
data["Stamina"].fillna(data["Stamina"].mean(), inplace=True)
data["Strength"].fillna(data["Strength"].mean(), inplace=True)

data["LongShots"].fillna(data["LongShots"].mean(), inplace=True)
data["Aggression"].fillna(data["Aggression"].mean(), inplace=True)
data["Interceptions"].fillna(data["Interceptions"].mean(), inplace=True)
data["Positioning"].fillna(data["Positioning"].mean(), inplace=True)
data["Vision"].fillna(data["Vision"].mean(), inplace=True)
data["Penalties"].fillna(data["Penalties"].mean(), inplace=True)
data["Composure"].fillna(data["Composure"].mean(), inplace=True)
data["Marking"].fillna(data["Marking"].mean(), inplace=True)
data["StandingTackle"].fillna(data["StandingTackle"].mean(), inplace=True)
data["SlidingTackle"].fillna(data["SlidingTackle"].mean(), inplace=True)
data["GKDiving"].fillna(data["GKDiving"].mean(), inplace=True)
data["GKHandling"].fillna(data["GKHandling"].mean(), inplace=True)
data["GKKicking"].fillna(data["GKKicking"].mean(), inplace=True)
data["GKPositioning"].fillna(data["GKPositioning"].mean(), inplace=True)
data["GKReflexes"].fillna(data["GKReflexes"].mean(), inplace=True)

data["Crossing"].fillna(data["Crossing"].mean(), inplace=True)
data["Finishing"].fillna(data["Finishing"].mean(), inplace=True)
data["HeadingAccuracy"].fillna(data["HeadingAccuracy"].mean(), inplace=True)
data["ShortPassing"].fillna(data["ShortPassing"].mean(), inplace=True)
data["Volleys"].fillna(data["Volleys"].mean(), inplace=True)
data["Dribbling"].fillna(data["Dribbling"].mean(), inplace=True)
data["FKAccuracy"].fillna(data["FKAccuracy"].mean(), inplace=True)
data["LongPassing"].fillna(data["LongPassing"].mean(), inplace=True)
data["BallControl"].fillna(data["BallControl"].mean(), inplace=True)
data["Acceleration"].fillna(data["Acceleration"].mean(), inplace=True)
data["SprintSpeed"].fillna(data["SprintSpeed"].mean(), inplace=True)
data["Agility"].fillna(data["Agility"].mean(), inplace=True)
data["Reactions"].fillna(data["Reactions"].mean(), inplace=True)
data["Balance"].fillna(data["Balance"].mean(), inplace=True)

data["Jersey Number"].fillna(8, inplace=True)
data["Joined"].fillna('Jan 1, 2004', inplace=True)
data["Contract Valid Until"].fillna('Jan 1, 2019', inplace=True)
data.columns[data.isnull().any()]
sns.countplot(x="Preferred Foot", data=data)
plt.title("Player count of Left and Right foot");
plt.subplots(figsize=(16,8))
sns.countplot(x="Height", data=data)
plt.title("Player count of various Height");
sns.lmplot(data = data, x = 'Age', y = 'SprintSpeed', lowess=True,line_kws={'color':'black'}, scatter_kws={'alpha':0.01})
df = ['Name', 'Age', 'Nationality', 'Club', 'Overall', 'Potential', 'Value', 'Wage', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot', 'Skill Moves', 'Work Rate', 'Body Type', 'Real Face', 'Position', 'Jersey Number', 'Joined', 'Loaned From', 'Contract Valid Until', 'Height', 'Weight', 'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause']
new_dataset = pd.DataFrame(data, columns=df)
new_dataset.head()
print(new_dataset.shape)
new_dataset.iloc[new_dataset.groupby(new_dataset['Position'])['Potential'].idxmax()][['Position', 'Name', 'Age', 'Value', 'Club', 'Nationality']]
country_list = ["Brazil" , "Argentina", "Italy", "Germany", "Spain", "Columbia", "France", "Uruguay", "Prtugal", "Denmark"]
country_vs_age = new_dataset.loc[new_dataset["Nationality"].isin(country_list) & data["Age"]]
plt.subplots(figsize=(10,6))
sns.barplot(x=country_vs_age["Nationality"], y=country_vs_age["Age"], data=new_dataset)
plt.title("Some countries and their player age");
club_list = ["FC Barcelona" , "Real Madrid", "Juventus", "Paris Saint-Germain", "Manchester City", "Manchester United", "Chelsea", "Napoli", "Arsenal", "Liverpool"]
club_vs_wage = new_dataset.loc[new_dataset["Club"].isin(club_list) & new_dataset["Wage"]]
plt.subplots(figsize=(10,10))
sns.barplot(x=club_vs_wage["Club"], y=club_vs_wage["Wage"], data=new_dataset)
plt.title("Some club and their player age");
plt.xticks(rotation=-40);
# df = new_dataset.groupby([new_dataset["Name", "Age", "Value", "Club"])[]

df2 = new_dataset.groupby(["Name", "Value", "Club", "Nationality"])["Age"].sum()\
.groupby(["Name", "Club", "Nationality"]).max().sort_values()
# .groupby(["Age"]).sum().sort_values(ascending=False)
young_club = pd.DataFrame(df2)
young_club.head(20)
df2 = new_dataset.groupby(["Name", "Value", "Club", "Nationality"])["Age"].sum()\
.groupby(["Name", "Club", "Nationality"]).max().sort_values(ascending=False)
eldest_age_club = pd.DataFrame(df2)
eldest_age_club.head(20)
def ComparePlayer(*argv):
#     Stamina
#     Finishing
#     Dribbling
#     Balance
#     ShotPower
#     Strength
    name_1 = argv[0]
    name_2 = argv[1]
    Name_1_data = new_dataset.loc[(new_dataset["Name"]==name_1)].reset_index(drop=True)
    Name_2_data = new_dataset.loc[(new_dataset["Name"]==name_2)].reset_index(drop=True)
    print(Name_1_data.Overall)
    print(Name_2_data.Overall)
    
    
    
ComparePlayer("K. De Bruyne", "L. Messi")
def player(name):
    club_nem = new_dataset.loc[(new_dataset["Name"]==name)]  
    pd.set_option('display.max_columns', 200)
    return club_nem
player("Neymar Jr").T
new_dataset.columns
import datetime
now = datetime.datetime.now()
new_dataset['Joining_year'] = new_dataset.Joined.dropna().map(lambda x:x.split(",")[1].split(" ")[1])
new_dataset["Memebership_year"] = (new_dataset.Joining_year.dropna().map(lambda x:now.year - int(x)).astype(int))
year_of_membership = new_dataset[["Name", "Club", "Memebership_year"]].sort_values("Memebership_year", ascending=False).head(20)
year_of_membership.set_index("Name", inplace=True)
year_of_membership
player_feature = new_dataset[['Position','Acceleration', 'Agility', 'Balance', 'BallControl',
       'ShotPower', 'Jumping', 'Crossing', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Penalties',
       'Composure', 'StandingTackle', 'SlidingTackle', 'GKDiving',
       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']]
player_feature.head()
feature_of_player = ('Acceleration', 'Agility', 'Balance', 'BallControl',
       'ShotPower', 'Jumping', 'Crossing', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Penalties',
       'Composure', 'StandingTackle', 'SlidingTackle', 'GKDiving',
       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes', 'Crossing')
i = 0
while i < len(feature_of_player):
    print('Best {0} : {1}'.format(feature_of_player[i],new_dataset.loc[new_dataset[feature_of_player[i]].idxmax()][0]))
    i+=1
for i, val in player_feature.groupby(player_feature["Position"])[feature_of_player].mean().iterrows():
    print("Position Of Player in {}: {}, {}, {}, {}".format(i, *tuple(val.nlargest(4).index)))
from math import pi
idx = 1
plt.figure(figsize=(15,45))
for position_name, features in new_dataset.groupby(new_dataset["Position"])[feature_of_player].mean().iterrows():
    top_features = dict(features.nlargest(5))
    
    categories = top_features.keys()
    N = len(categories)
    
    values = list(top_features.values())
    values += values[:1]
    
    angles =[n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    ax = plt.subplot(9, 3, idx, polar=True)
    
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    
    ax.set_rlabel_position(0)
    plt.yticks([25,50,75], ["25", "50", "75"], color="grey", size=7)
    plt.ylim(0, 100)
    
    plt.subplots_adjust(hspace = 0.5)
    
    ax.plot(angles, values, linewidth=1, linestyle="solid")
    
    ax.fill(angles, values, 'b', alpha=0.1)
    
    plt.title(position_name, size=11, y=1.1)
    
    idx += 1
new_dataset.groupby(new_dataset["Overall"])["Name", "Age"].max().sort_values("Overall", ascending=False).head(10)
