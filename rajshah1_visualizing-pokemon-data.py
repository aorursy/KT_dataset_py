# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

%matplotlib inline



import plotly as py

import plotly.graph_objs as go

py.offline.init_notebook_mode(connected=True)
df = pd.read_csv("../input/Pokemon.csv") # Imported the data from pokemon.csv file and stored in df dataframe
df.columns = df.columns.str.upper().str.replace("_", " ") # we are capitalizing the headers of the each column.

# So all the headers of the columns are capitalized.
df = df.set_index("NAME") # Setting the index of the dataframe to the name of pokemon.
df = df.drop(["#"], axis=1) # Dropping the # column in the dataframe which of no use to us.
# Removed all the text before "Mega" |For Ex. In the fourth row of dataframe 

                                # we removed extra name of the pokemon- VenusaurMega Venusaur to just

                                # Mega Venusaur. It applies for all the rows in dataframe.

df.index = df.index.str.replace(".*(?=Mega)", "")
# Replacing all the null values in the TYPE 2 column with their values of TYPE 1 in dataframe. 

df["TYPE 2"].fillna(df["TYPE 1"], inplace=True)

df.head()
# Summary of our dataset

df_summary = df.describe()

print(df_summary)
bins = range(0, 200, 20)  # Using matplotlib Here

plt.hist(df["ATTACK"], bins, histtype="bar", rwidth=1, color="c")

plt.xlabel("Attack Power")

plt.ylabel("No. of Pokemons")

plt.plot()

plt.axvline(df["ATTACK"].mean(), linestyle="dashed", color="r")

plt.show()
dragon = df[(df["TYPE 1"] == "Dragon") | (df["TYPE 2"] == "Dragon")] 

# These are customized dataframe which have only dragon type pokemons used OR operations here.

psychic = df[(df["TYPE 1"] == "Psychic") | (df["TYPE 2"] == "Psychic")] 

# These are customized dataframe which have only psychic type pokemons used OR operations here.



# It is the layout for the graph.

layout = go.Layout(title="Psychic vs Dragon",

                   yaxis=dict(title="Defence"),xaxis=dict(title="Attack"))



# This is the trce for dragon type pokemon.

trace0 = go.Scatter(x=dragon.ATTACK, y=dragon.DEFENSE, mode="markers",

    name="Dragon",marker = dict(color="rgb(255,0,0)", size=8), text=dragon.index)



# This is the trce for psychic type pokemon.

trace1 = go.Scatter(x=psychic.ATTACK, y=psychic.DEFENSE, mode="markers",

    name="Psychic",marker = dict(color="rgb(0,100,0)", size=8), text=psychic.index)



# compiling all the data from traces and appling layout.

fig = go.Figure(data=[trace0, trace1], layout=layout)



# Plots the graph.

py.offline.iplot(fig)
print("The total types of Pokemons are : ", df["TYPE 1"].unique())
def Poke_type_compare(type1, type2):# Taking two pokemon type arguments

    # In case of wrong arguments exception will raise and code will not crash.

    try: 

        t1 = type1.capitalize()

        t2 = type2.capitalize()

        title1 = t1 + " vs " + t2 + " "  + "Pokemons"



        poke1 = df[(df["TYPE 1"] == t1) | (df["TYPE 2"] == t1)]

        poke2 = df[(df["TYPE 1"] == t2) | (df["TYPE 2"] == t2)]



        layout = go.Layout(title= title1,

                               yaxis=dict(title="Defence Power"),xaxis=dict(title="Attack Power"))



        trace1 = go.Scatter(x=poke1.ATTACK, y=poke1.DEFENSE, mode="markers",

                name=t1, marker = dict(color="rgb(255,0,0)", size=8), text=poke1.index)



        trace2 = go.Scatter(x=poke2.ATTACK, y=poke2.DEFENSE, mode="markers",

                name=t2, marker = dict(color="rgb(0,100,0)", size=8), text=poke2.index)



        fig = go.Figure(data=[trace1, trace2], layout=layout)

        py.offline.iplot(fig)

    except Exception:

        print("Enter the names of pokemon types in lower.")

        

Poke_type_compare("water", "bug")

Poke_type_compare("grass", "normal")

# you can try various types of combinations here.

# Poke_type_compare("ghost", "electric")

# Poke_type_compare("ground", "dark")
df_test = df.drop(["TYPE 1", "TYPE 2", "TOTAL", "GENERATION", "LEGENDARY"], axis=1)



def poke_compare(poke1, poke2):

    try:    

        df_test.loc[poke1].plot(color="r", marker="o", markersize=10 , label=poke1)

        df_test.loc[poke2].plot(color="c", marker="o", markersize=10, label=poke2)

        plt.xlabel("Features")

        plt.ylabel("Values")

        plt.title(poke1 + " vs " + poke2)

        fig = plt.gcf()

        fig.set_size_inches(12, 6)

        plt.legend()

        plt.show()

    except Exception:

        print("Please enter the correct names of the Pokemons")

    

poke_compare("Charmander", "Mega Venusaur")

# Here also you can type different names of pokemon and compare them.

# To find the names you can just google it and enjoy comparing the pokemons.
def poke_compare(poke1, poke2):

    try:

        fig = plt.figure()

        ax = fig.add_subplot(111)

        width = 0.4

        

        df_test.loc["Charmander"].plot(kind="bar", color="r", ax=ax,

                    width=width,position=1, label=poke1)

        

        df_test.loc["Mega Venusaur"].plot(kind="bar", color="c",

                    ax=ax, width=width, position=0, label=poke2)

        

        plt.xticks(rotation=0)

        plt.xlabel("Features")

        plt.ylabel("Values")

        plt.title(poke1 + " vs " + poke2)

        plt.legend()

        fig = plt.gcf()

        fig.set_size_inches(12, 6)

        plt.show()

    except Exception:

        print("Please enter the correct names of the Pokemons")

        

        

poke_compare("Pikachu", "Squirtle")

# Here also you can type different names of pokemon and compare them.

# To find the names you can just google it and enjoy comparing the pokemons.
labels = ['Water', 'Normal', 'Grass', 'Bug', 'Psychic', 'Fire', 'Electric',

        'Rock', 'Other']

sizes = [112, 98, 70, 69, 57, 52, 44, 44, 175] # These are the no. of pokemons present in respective type.

colors = ['Y', 'B', '#00ff00', 'C', 'R', 'G', 'silver', 'white', 'M']

plt.pie(sizes, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')

plt.title("Percentage of Different Types of Pokemon")

plt.plot()

fig=plt.gcf()

fig.set_size_inches(7,7)

plt.show()