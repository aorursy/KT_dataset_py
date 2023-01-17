import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime
df_nba=pd.read_csv('../input/nba2k20-player-dataset/nba2k20-full.csv')
df_nba.head(10)
df_nba.info()
def from_date_to_age(date):

    born=datetime.datetime.strptime(date, '%m/%d/%y')

    today = datetime.date.today()

    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

df_nba["current_age"]=df_nba["b_day"].apply(lambda x: from_date_to_age(x))





def int_weight(weight):

    return weight.split("/")[1].split(" ")[1]



def int_height(height):

    return height.split("/")[1].split(" ")[1]



df_nba["height"]=df_nba["height"].apply(lambda x: int_height(x)).astype("float")

df_nba["weight"]=df_nba["weight"].apply(lambda x: int_weight(x)).astype("float")



df_nba.rename({"height":"height_in_m","weight":"weight_in_kg"},axis='columns',inplace=True)

df_nba["salary"] = df_nba["salary"].str[1:].astype("int64")

df_nba["year_played"] = df_nba["current_age"] - (df_nba["draft_year"] - pd.to_datetime(df_nba["b_day"]).dt.year)



df_nba["draft_round"] = df_nba["draft_round"].replace({"Undrafted": 0}).astype("int8")

df_nba["draft_peak"] = df_nba["draft_peak"].replace({"Undrafted": 0}).astype("int8")



df_nba.drop(columns=['b_day'])



df_nba["body_mass_index"] = np.round(df_nba["weight_in_kg"] / ((df_nba["height_in_m"])**2),1)

df_nba.loc[(df_nba["body_mass_index"]>=18.5) & (df_nba["body_mass_index"]<=24.9),"bmi_class"] = "Normal"

df_nba.loc[(df_nba["body_mass_index"]>=25) & (df_nba["body_mass_index"]<=29.9),"bmi_class"] = "Overweight"

df_nba.loc[df_nba["body_mass_index"]>=30,"bmi_class"] = "Obese"



df_nba['college'].isna().astype('int').value_counts() 

df_nba["attended_college"] = df_nba['college'].isna().astype('int')



df_nba.head(10)
plt.figure(figsize=(15,8))

sns.heatmap(df_nba.corr(), annot=True, linewidths=0.5, linecolor='black', cmap='coolwarm')

plt.show()
df_nba["country"].value_counts()
country = df_nba[['country','full_name']].groupby('country').count().sort_values(by='full_name', ascending=False)

fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(x=country.index, y=country.full_name)

plt.xticks(rotation=90)

plt.xlabel('Country')

plt.ylabel('Players count')

plt.title('Players\' Country')

plt.show()
plt.rcParams['figure.figsize'] = (10, 8)

sns.distplot(df_nba['current_age'], color = 'blue')

plt.xlabel('Ages range for players')

plt.ylabel('Count of players')

plt.title('Players\' Weight Distribution')

plt.xticks()

plt.show()

print(f"{df_nba[df_nba.current_age == df_nba.current_age.max()].full_name.values[0]} is the oldest nba player and has ({df_nba.current_age.max()} years old)")

print(f"{df_nba[df_nba.current_age == df_nba.current_age.min()].full_name.values[0]} is the youngest nba players and has ({df_nba.current_age.min()} years old)")
plt.rcParams['figure.figsize'] = (10, 8)

sns.distplot(df_nba['height_in_m'], color = 'blue')

plt.xlabel('Height range for players')

plt.ylabel('Count of players')

plt.title('Players\' Height Distribution')

plt.xticks()

plt.show()

print(f"{df_nba[df_nba.height_in_m == df_nba.height_in_m.max()].full_name.values[0]} is the tallest nba player and is ({df_nba.height_in_m.max()} m)")

print(f"{df_nba[df_nba.height_in_m == df_nba.height_in_m.min()].full_name.values[0]} is the smallest nba players and is ({df_nba.height_in_m.min()} m)")
plt.figure(figsize=(15, 8))

plt.title("Salary distribution based on players teams", fontsize=18)

x = sns.boxplot(x="team", y="salary", data=df_nba)

x.set_xticklabels(x.get_xticklabels(), rotation=90);
plt.figure(figsize=(15, 8))

plt.title("Salary distribution based on players jerseys", fontsize=18)

x = sns.boxplot(x="jersey", y="salary", data=df_nba)

x.set_xticklabels(x.get_xticklabels(), rotation=90);
plt.figure(figsize=(15,10))

ax = sns.barplot(data = df_nba, x = 'year_played', y = 'salary')

plt.xlabel('Years after draft', fontsize=15)

plt.ylabel('Salary', fontsize=15)

plt.show()
df_nba.loc[df_nba['position'] == 'C-F', 'position'] = 'F-C'

df_nba.loc[df_nba['position'] == 'F-G', 'position'] = 'G-F'
plt.figure(figsize=(15, 8))

plt.title("Salary distribution based on players postions", fontsize=18)

x = sns.boxplot(x="position", y="salary", data=df_nba)

x.set_xticklabels(x.get_xticklabels(), rotation=90);
p_r = df_nba[['position','rating']].groupby('position').mean().sort_values(by='rating', ascending=False)

fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(x=p_r.index, y=p_r.rating)

plt.xticks(rotation=90)

plt.xlabel('Position')

plt.ylabel('Players rating mean')

plt.title('Players\' position by mean rating')

plt.show()
labels = ['G', 'F','C','F-C','G-F'] 

size = df_nba['position'].value_counts()

colors = plt.cm.RdYlBu(np.linspace(0, 1, 8))

explode = [0, 0, 0, 0, 0]



plt.rcParams['figure.figsize'] = (10, 10)

plt.pie(size, labels = labels, colors = colors, explode = explode, shadow = True, startangle = 0)

plt.title('Distribution of Players\' Position')

plt.legend()

plt.show()



team_ret = df_nba[['team','rating']].groupby('team').mean().sort_values(by='rating', ascending=False)

#country = df_nba[['country','full_name']].groupby('country').count().sort_values(by='full_name', ascending=False)

fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(x=team_ret.index, y=team_ret.rating)

plt.xticks(rotation=90)

plt.xlabel('Teams')

plt.ylabel('Average rating')

plt.title('Players\' average rating over all teams')

plt.show()
#plt.rcParams['figure.figsize'] = (10, 8)

sns.pairplot(df_nba, x_vars=["salary", "height_in_m"], y_vars=["rating"],

             height=10, aspect=.8, kind="reg");

plt.xticks()

plt.show()



#sns.lmplot(x="total_bill", y="tip", hue="attended_college", data=df_nba);



#plt.rcParams['figure.figsize'] = (10, 8)

sns.pairplot(df_nba, x_vars=["salary", "height_in_m"], y_vars=["rating"], hue="attended_college",

             height=10, aspect=.8, kind="reg");

plt.xticks()

plt.show()

df_nba["bmi_class"].value_counts()
g = sns.catplot(x="bmi_class", y="height_in_m", kind="violin", inner=None, data=df_nba)

sns.swarmplot(x="bmi_class", y="height_in_m", color="k", size=3, data=df_nba, ax=g.ax);
g = sns.PairGrid(df_nba[["rating","salary","height_in_m","weight_in_kg","bmi_class"]], hue="bmi_class", palette="magma")

g.map(plt.scatter, s=50, edgecolor="white")

g.add_legend()