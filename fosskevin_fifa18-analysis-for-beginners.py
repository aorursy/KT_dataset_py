# this statement will import the pandas library into our notebook and give it a nickname "pd"

import pandas as pd
# here we use the nickname "pd" to access the read_csv function

df = pd.read_csv("../input/CompleteDataset.csv")

# this invokes the linux "ls" command which lists files in the particular folder

!ls ../input
print(df.columns)
players = df[["Name", "Age", "Club"]]
columnsRequired = ["Name", "Age", "Club"]

players = df[columnsRequired]
# the original DataFrame has a lot of columns!

print(df.shape)
# our simplified DataFrame has 3 columns. Yay!

print(players.shape)
# or you can just type players and Kaggle notebook will print the contents with nicer formatting

print(players)
# no need for print statement

players
players.head()
players.tail(10)
players.isnull().head()
players.isnull().head().sum()
players.isnull().sum()
players = players.dropna()
players.isnull().sum()
players.sort_values("Age").head(20)
players.sort_values("Age", ascending=False).head()
players.groupby("Club").mean()
players.groupby("Club").mean().sort_values("Age")
realMadridPlayers = players[players["Club"] == "Real Madrid CF"]

realMadridPlayers.sort_values("Age", ascending=False).head(10)
players["Club"].unique()
players = df[["Name", "Age", "Club", "Wage"]].dropna()
players.head()
players.dtypes
str = "Hello world"

str = str.replace("l", "L")

str
str = str.replace("L", "")

str
# removes €

wages = players['Wage'].map(lambda x: x.replace("€", ""))

wages.head()
# removes K

wages = wages.map(lambda x: x.replace("K", ""))

wages.head()
wages = wages.astype("int")*1000
# I don't understand what copy() does, but it was needed for correct working

players = players.copy()

players.loc[:, "Wage"] = wages.values

players.dtypes
players.groupby("Club")["Wage"].mean().sort_values(ascending=False).head(10)
players.sort_values("Wage", ascending=False).head(20)
players = df[["Name", "Age", "Club", "Nationality"]].dropna()

players.groupby("Nationality").Name.count().sort_values(ascending=False).head(10).plot(kind="bar");

players.groupby("Nationality").Name.count()
players.groupby("Nationality").Name.count().sort_values(ascending=False).head(10)
players.groupby("Nationality").Name.count().sort_values(ascending=False).head(10).plot()
players.groupby("Nationality").Name.count().sort_values(ascending=False).head(10).plot(kind="bar")
# horizontal bar plot

players.groupby("Nationality").Name.count().sort_values(ascending=False).head(10).plot(kind="barh")
# pie chart

players.groupby("Nationality").Name.count().sort_values(ascending=False).head(10).plot(kind="pie");
df.describe()
#with pd.option_context('display.max_rows', -1, 'display.max_columns', -1):

#    df.describe()
#with pd.option_context('display.max_rows', -1, 'display.max_columns', -1):

#    desc = df.describe()

#    print(desc)
df = df.drop("Unnamed: 0", axis=1)

df.columns