import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.image as mpimg

import folium

import os

import warnings
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/fifa19/data.csv")

df.head()
df.columns
features = df.drop(columns=["Unnamed: 0", "ID", "Photo", "Flag", "Potential", "Club Logo", "Special", "Preferred Foot", "International Reputation", "Weak Foot", "Skill Moves", "Work Rate", "Body Type", "Real Face", "Jersey Number","Height", "Weight", "LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM", "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB", "CB", "RCB", "RB"], inplace=False)

features.head()
def clean_currency(x):

    """If the value is in thousands or millions, clear the "K" or "M", respectively, and add the corresponding zeros

    """

    if x.find("K") > -1 and x.find(".") == -1:

        return x.replace("K","000").replace("€","")

    elif x.find("K") > -1 and x.find(".") > -1:

        return x.replace("K","00").replace(".","").replace("€","")

    elif x.find("M") > -1 and x.find(".") == -1:

        return x.replace("M","000000").replace("€","")

    elif x.find("M") > -1 and x.find(".") > -1:

        return x.replace("M","00000").replace(".","").replace("€","")

    else:

        return x.replace("€","")

    return(x)

columns_in_euros = ["Value", "Wage", "Release Clause"]

for column in columns_in_euros:

    features[column] = features[column].astype("str")

    features[column] = features[column].apply(clean_currency).astype("float64")

features.head()
missing_lst = []

for col in features.columns:

    column_name = col

    missing_values = int(features[col].isnull().sum())

    missing_lst.append([column_name,missing_values])

missing_df = pd.DataFrame(missing_lst,columns=["Column name","Missing values"])

missing_df.sort_values("Missing values",ascending=False).head()
features = features.dropna(subset=["Joined","Release Clause"])
missing_lst = []

for col in features.columns:

    column_name = col

    missing_values = int(features[col].isnull().sum())

    missing_lst.append([column_name,missing_values])

missing_df = pd.DataFrame(missing_lst,columns=["Column name","Missing values"])

missing_df.sort_values("Missing values",ascending=False).head()
features["Joined"] = features["Joined"].astype("datetime64[ns]")

features["Contract Valid Until"] = features["Contract Valid Until"].astype("datetime64[ns]")
for column in features.columns[11:-1]:

    features[column] = features[column].astype("int64")
features.head()
features["Position"].unique()
# goalkeeping positions

goal_pos = ["GK"]

# defensive positions

def_pos = ["CB", "LB", "RB", "LWB", "RWB", "RCB", "LCB"]

# midfield positions

mid_pos = ["CDM", "RDM", "LDM", "CAM", "LAM", "RAM", "CM", "LM", "RM", "RW", "LW", "RCM", "LCM"]

# attacking positions

att_pos = ["CF", "ST", "LF", "RF", "LS", "RS"]
def graph_top_by_position(position, column1, column2, num_of_players):

    """Graph the top players by position.

        Variable position must be a list and num_players must be an integer"""

    # create a dictionary to retrive the corresponding value from position argument

    lists = {"goalkeepers":goal_pos, "defenders":def_pos, "midfielders":mid_pos, "attackers":att_pos}

    # DataFrame of top players

    top_players = features[features["Position"].isin(lists.get(position))].head(num_of_players)

    # DataFrame columns converted to lists

    names = top_players["Name"].to_list()

    rating = top_players[column1].to_list()

    value = top_players[column2].to_list()

    names.reverse()

    rating.reverse()

    value.reverse()

    # Plotting

    fig, axs = plt.subplots(2)

    fig.suptitle("Top "+str(num_of_players)+" "+position)

    axs[0].set_ylabel("Overall rating")

    axs[0].set_xlabel("Player name")

    axs[1].set_ylabel("Market value (by hundreds of millions)")

    axs[1].set_xlabel("Player name")

    ax1 = axs[0].barh(names, width=rating)

    ax2 = axs[1].barh(names, width=value)

    colors = ['#80ffaa', '#4dff88', '#1aff66', '#00cc44', '#009933']

    for x in range(5):

        ax1[x].set_color(colors[x])

        ax2[x].set_color(colors[x])

    plt.show()
graph_top_by_position("goalkeepers","Overall", "Value", 5)
graph_top_by_position("defenders","Overall","Value",5)
graph_top_by_position("midfielders","Overall","Value",5)
graph_top_by_position("attackers","Overall","Value",5)
# Create a new DataFrame of top clubs with the best average overall ratings

top_clubs_overall = features.groupby("Club",as_index=False)["Overall"].mean().sort_values("Overall", ascending=False).head(10)

# Plotting

colors = ["#003606","#003B06","#00610A","#00790D","#008C0E","#00A411","#00C915","#00ED19","#16FB2D","#5EFE6F"]

plt.barh("Club",width="Overall",height=0.5, data=top_clubs_overall,align="edge", color=colors)

plt.gca().invert_yaxis()

plt.title("Clubs with the best average overall ratings")

plt.ylabel("Clubs")

plt.xlabel("Average overall rating");
# Create a new DataFrame of top clubs with most expensive players

top_clubs_expensive = features.groupby("Club",as_index=False)["Value"].sum().sort_values("Value", ascending=False).head(10)

# Plotting

colors = ["#003606","#003B06","#00610A","#00790D","#008C0E","#00A411","#00C915","#00ED19","#16FB2D","#5EFE6F"]

plt.barh("Club",width="Value",height=0.4, data=top_clubs_expensive,align="edge", color=colors)

plt.gca().invert_yaxis()

plt.title("Most expensive clubs")

plt.ylabel("Clubs")

plt.xlabel("Values in hundreds of millions of Euros");
# Create a new DataFrame of top clubs with the most expensive release clauses

top_clubs_clauses = features.groupby("Club",as_index=False)["Release Clause"].sum().sort_values("Release Clause", ascending=False).head(10)

# Plotting

colors = ["#003606","#003B06","#00610A","#00790D","#008C0E","#00A411","#00C915","#00ED19","#16FB2D","#5EFE6F"]

plt.barh("Club",width="Release Clause",height=0.4, data=top_clubs_clauses,align="edge", color=colors)

plt.gca().invert_yaxis()

plt.title("Clubs with most expensive clauses")

plt.ylabel("Clubs")

plt.xlabel("Values in billions of Euros");
# New dataframe consisting of player market values, ratings, age and wage

features_col = features.loc[:,["Value","Overall","Age","Wage"]]

# Plotting

sns.scatterplot(features_col["Value"], features_col["Overall"], hue=features_col["Age"],palette="YlGn",size=features["Wage"],sizes=(40,400));

fig = plt.gcf()

# Change fig size

fig.set_size_inches(12, 8)

# Change axis labels

plt.xlabel("Market Value in Hundreds of millions of Euros");

plt.ylabel("Overall ratings");

plt.show()
def plot_avg_worldmap(by_column):

    # New dataframe consisting of top countries by market value

    fil_features = features.loc[:,["Nationality",by_column]]

    top_countries = fil_features.groupby("Nationality",as_index=False)[by_column].mean().sort_values(by_column, ascending=False)

    # Map plotting

    m = folium.Map(location=[0,0], zoom_start=1.5)

    legend = {"Overall":"Overall mean rating by country","Value":"Mean market values of players by country"}

    folium.Choropleth(

        geo_data="/kaggle/input/geojson/custom.geo.json",

        name='choropleth',

        data=top_countries,

        columns=['Nationality', by_column],

        key_on='feature.properties.sovereignt',

        fill_color='YlGn',

        fill_opacity=0.9,

        line_opacity=0.2,

        legend_name=legend.get(by_column)

    ).add_to(m)



    folium.LayerControl().add_to(m)



    return m
plot_avg_worldmap("Value")
plot_avg_worldmap("Overall")
def dream_team(formation):

    """Creates a scatter map that assembles a football pitch and plots the dream team for the determined formation. The argument "formation" can only accept one the following values: 4-4-2, 4-3-3, 4-2-3-1"""

    lst = []

    # create a base dataframe consisting of top players by position attribute

    for p in features["Position"].unique():

        top_gk = features[features["Position"].isin([p])].sort_values("Overall", ascending=False).head(1)

        gk = top_gk["Name"].astype("str").tolist()

        pos = p

        gk.append(pos)

        lst.append(gk)

    players = pd.DataFrame(lst, columns=["Player", "Position"])

    # include only relevant players in 4-4-2 and modify dataframe axis values

    if formation=="4-4-2":

        filters = ["GK","RCB","LCB","RB","LB","RDM","LDM","RAM","LAM","RF","LF"]

        df_team = players[players["Position"].isin(filters)]

        df_team["Y"] = [1.5,2,2.5,1.5,2.5,2.5,3.5,1.5,0.5,0.5,3.5]

        df_team["Position"] = df_team["Position"].apply(lambda x: 0.2 if x=="GK" else 0.8 if x in ["RCB","LCB"] else 0.9 if x in ["RB","LB"] else 2 if x in ["RDM","LDM"] else 2.1 if x in ["RAM","LAM"] else 3.2)

    # include only relevant players in 4-4-3 and modify dataframe axis values

    elif formation=="4-3-3":

        filters = ["GK","RCB","LCB","RB","LB","RCM","LCM","CM","RW","LW","ST"]

        df_team = players[players["Position"].isin(filters)]

        df_team["Y"] = [2,3.5,2,0.5,1.5,3.5,2.5,3.5,0.5,2,0.5]

        df_team["Position"] = df_team["Position"].apply(lambda x: 0.2 if x=="GK" else 0.8 if x in ["RCB","LCB"] else 0.9 if x in ["RB","LB"] else 2 if x in ["RCM","LCM","CM"] else 3.0 if x in ["RW","LW"] else 3.2)

        df_team

    # include only relevant players in 4-2-3-1 and modify dataframe axis values

    elif formation=="4-2-3-1":

        filters = ["GK","RCB","LCB","RB","LB","RDM","LDM","CAM","RM","LM","CF"]

        df_team = players[players["Position"].isin(filters)]

        df_team["Y"] = [2,1.5,2.5,2,2.5,0.7,3.3,3.5,1.5,0.5,2]

        df_team["Position"] = df_team["Position"].apply(lambda x: 0.2 if x=="GK" else 0.8 if x in ["RCB","LCB"] else 0.9 if x in ["RB","LB"] else 1.5 if x in ["RDM","LDM"] else 2.3 if x in ["CAM"] else 2.7 if x in ["RM","LM"] else 3.2)

    # Plotting

    img = mpimg.imread("/kaggle/input/imagefile/pitch1.png")

    fig, ax = plt.subplots()

    ax.grid()

    ax.scatter("Position","Y", data=df_team)

    ax.set_title("Dream team "+formation)

    ax.set_ylabel("Clubs")

    ax.axis("off")

    axes = plt.gca()

    axes.set_ylim(bottom=0, top=4.3)

    axes.set_xlim(left=0, right=4)

    fig.figimage(img, 0,-10, resize=False, alpha=0.3)

    # Create a list of coordinates

    x = df_team["Position"].tolist()

    y = df_team["Y"].tolist()

    labels = df_team["Player"].tolist()

    coord = []

    for ele in range(11):

        coord.append((x[ele],y[ele]))

    # Use the list of coordinates to generate labels

    for i, label in enumerate(labels):

        x, y = coord[i]

        ax.annotate(label, xy=(x,y), xytext= (30,7),textcoords = "offset points", ha='right', va='bottom')

    plt.show()
warnings.filterwarnings("ignore")

dream_team("4-4-2")
dream_team("4-3-3")
dream_team("4-2-3-1")
heatmap_df = features.loc[:,['Name','Overall','Crossing',

       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

       'GKKicking', 'GKPositioning', 'GKReflexes']].sort_values("Overall",ascending=False).set_index("Name").head(20)

plt.title("Top 20 players attribute map")

sns.heatmap(heatmap_df, fmt="d", annot=True, cmap="Greens");

fig = plt.gcf()

# Change fig size

fig.set_size_inches(12, 8)