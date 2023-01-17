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
data = pd.read_csv("../input/nba2k20-player-dataset/nba2k20-full.csv")
data.head(2) #data.head(n) will show first n rows
data.tail(2) #data.tail(n) will show last n rows 
len(data)
type(data)
list(data)
dict(data)
max(data)
min(data)
data.values
data.index
data.shape
data.columns
data.axes
1 in [1,2,3,4,5]
50 in data.index
"Messi" in data.index
data["team"]
data[["full_name", "jersey"]]
data["League"] = "NBA"
data["Sport"] = "Basketball"
data = pd.read_csv("../input/nba2k20-player-dataset/nba2k20-full.csv")
data.insert(3, column = "Sport", value = "Basketball")
data = pd.read_csv("../input/nba2k20-player-dataset/nba2k20-full.csv")
data["rating"].add(10)
data["rating"] + 5

data["draft_year"].sub(2)
data["draft_year"] - 2

data["draft_peak"].mul(1)
data["draft_peak"] * 1

data["rating"].div(1)
data["rating"] / 1
data["position"].value_counts()
data.dropna(how = "all", inplace = True)
data.dropna(subset = ["position", "country"])
data = pd.read_csv("../input/nba2k20-player-dataset/nba2k20-full.csv")
data.info()
data["team"] = data["team"].astype("category")
data["country"] = data["country"].astype("category")
data.info() #memory usage decreased, it is usefull for large datasets
data.sort_values("rating", ascending = False, inplace = True)
data.sort_values(["team", "rating"], ascending=[True, False], inplace = True)
data.head()
data = pd.read_csv("../input/nba2k20-player-dataset/nba2k20-full.csv")
data.set_index("full_name", inplace = True)
data.head(2)
data.reset_index(drop = False, inplace = True)
data.tail(2)
data = pd.read_csv("../input/nba2k20-player-dataset/nba2k20-full.csv", index_col="full_name")
data.sample(3)
data.loc["Kawhi Leonard"]
data.loc["Kawhi Leonard":"Malcolm Brogdon"]
data.loc[:"Malcolm Brogdon"]
data.loc[["Kawhi Leonard", "Malcolm Brogdon"]]
data = pd.read_csv("../input/nba2k20-player-dataset/nba2k20-full.csv")
data.iloc[3]
data.iloc[[4,10]]
data.iloc[4:10]
data.iloc[:10]
# & --> and
# | --> or
data[data["position"] == "F"]
data[(data["team"] == "Los Angeles Lakers")& (data["rating"] > 85)] 
data[(data["country"] != "USA") | (data["rating"] > 85)] 
data[data["rating"].between(75, 80)]
data["team"].unique()
data["team"].nunique()
# Adding +5 for rating column
[x+5 for x in data["rating"]] # not inplaced, to inplace data["rating"] = 
data = pd.read_csv("../input/nba2k20-player-dataset/nba2k20-full.csv")
def add_plus_five(x):
    return x+5

data["rating"].apply(add_plus_five)
data = pd.read_csv("../input/nba2k20-player-dataset/nba2k20-full.csv")
data["rating"].apply(lambda x: x+5)
data = pd.read_csv("../input/nba2k20-player-dataset/nba2k20-full.csv")
data["team"].str.upper() #LOS ANGELES LAKERS
data["team"].str.lower() #los angeles lakers
data["team"].str.title() #Los Angeles Lakers
data["full_name"].str.split(" ").str[0]
data["first_name"] = data["full_name"].str.split(" ").str[0]
data["last_name"] = data["full_name"].str.split(" ").str[1]
data.head(3)
data.columns
data = data.reindex(columns = ['full_name', 'first_name', 'last_name', 'rating', 'jersey', 'team', 'position', 'b_day', 'height',
       'weight', 'salary', 'country', 'draft_year', 'draft_round',
       'draft_peak', 'college'])
data.head(3)
data = pd.read_csv("../input/nba2k20-player-dataset/nba2k20-full.csv")
data["salary"].str.replace("$", "").astype("int")
data["height in metres"] = data["height"].str.split("/").str[1].astype(float)
data.head(3)