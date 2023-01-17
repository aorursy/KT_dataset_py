%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.style.use("classic")



star_wars = pd.read_csv("/kaggle/input/star-wars-survey-data/star_wars.csv", encoding="Latin-1")

star_wars.head(5)
star_wars.shape
star_wars.columns
star_wars = star_wars[star_wars["RespondentID"].notnull()]
yes_no = {"Yes": True, "No": False}



for col in [

    "Have you seen any of the 6 films in the Star Wars franchise?",

    "Do you consider yourself to be a fan of the Star Wars film franchise?"

    ]:

    star_wars[col] = star_wars[col].map(yes_no)



star_wars.head(3)
star_wars = star_wars.rename(columns = {"Which of the following Star Wars films have you seen? Please select all that apply.": "seen_1",

          "Unnamed: 4": "seen_2",

          "Unnamed: 5": "seen_3", 

          "Unnamed: 6": "seen_4", 

          "Unnamed: 7": "seen_5",

          "Unnamed: 8": "seen_6"

         })





movie_mapping = {"Star Wars: Episode I  The Phantom Menace": True,

           np.nan: False,

           "Star Wars: Episode II  Attack of the Clones": True,

           "Star Wars: Episode III  Revenge of the Sith": True,

           "Star Wars: Episode IV  A New Hope": True,

           "Star Wars: Episode V The Empire Strikes Back": True,

           "Star Wars: Episode VI Return of the Jedi": True

}



for v in star_wars.columns[3:9]:

    star_wars[v] = star_wars[v].map(movie_mapping)

      

star_wars.head()
star_wars = star_wars.rename(columns = {"Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.": "ranking_1",

          "Unnamed: 10":"ranking_2",

          "Unnamed: 11":"ranking_3",

          "Unnamed: 12":"ranking_4",

          "Unnamed: 13":"ranking_5",

          "Unnamed: 14":"ranking_6"

         })



star_wars[star_wars.columns[9:15]].head()
star_wars[star_wars.columns[9:15]] = star_wars[star_wars.columns[9:15]].astype(float)

star_wars[star_wars.columns[9:15]].mean()
fig = plt.figure(figsize = (5,4))

plt.bar(range(6), star_wars[star_wars.columns[9:15]].mean())
star_wars[star_wars.columns[3:9]].sum()
fig = plt.figure(figsize = (5,4))

plt.bar(range(6), star_wars[star_wars.columns[3:9]].sum())
males = star_wars[star_wars["Gender"] == "Male"]

females = star_wars[star_wars["Gender"] == "Female"]
fig = plt.figure(figsize=(5,4))

#Ranked movies by males

plt.bar(range(6), males[males.columns[9:15]].mean())

plt.title("Males")

plt.show()





fig = plt.figure(figsize=(5,4))

#Ranked movies by females

plt.bar(range(6), females[females.columns[9:15]].mean())

plt.title("Females")

plt.show()
fig = plt.figure(figsize=(5,4))

#View count by males

plt.bar(range(6), males[males.columns[3:9]].sum())

plt.title("Males")

plt.show()





fig = plt.figure(figsize=(5,4))

#View count by females

plt.bar(range(6), females[females.columns[3:9]].sum())

plt.title("Females")

plt.show()
star_wars[star_wars.columns[15:29]].head()
star_wars.columns[15:29].value_counts()
star_wars["Unnamed: 17"].value_counts(dropna=False)
star_wars = star_wars.rename(columns = {"Please state whether you view the following characters favorably, unfavorably, or are unfamiliar with him/her.": "Han Solo",

                                        "Unnamed: 16": "Luke Skywalker",

                                        "Unnamed: 17": "Princess Leia Organa",

                                        "Unnamed: 18": "Anakin Skywalker",

                                        "Unnamed: 19": "Obi Wan Kenobi",

                                        "Unnamed: 20": "Emperor Palpatine",

                                        "Unnamed: 21": "Darth Vader",

                                        "Unnamed: 22": "Lando Calrissian",

                                        "Unnamed: 23": "Boba Fett",

                                        "Unnamed: 24": "C-3P0",

                                        "Unnamed: 25": "R2 D2",

                                        "Unnamed: 26": "Jar Jar Binks",

                                        "Unnamed: 27": "Padme Amidala",

                                        "Unnamed: 28": "Yoda"})





#star_wars[star_wars.columns[15:29]]



most_favorable = {"Very favorably": 1,

                   np.nan: 4,

                  "Somewhat favorably": 2,

                  "Neither favorably nor unfavorably (neutral)": 3,

                  "Unfamiliar (N/A)": 4,

                  "Somewhat unfavorably": 5,

                  "Very unfavorably": 6

                 }



for v in star_wars.columns[15:29]:

    star_wars[v] = star_wars[v].map(most_favorable)
    

fav_list = ["Han Solo", 

            "Luke Skywalker", 

            "Princess Leia Organa", 

            "Anakin Skywalker", 

            "Obi Wan Kenobi", 

            "Emperor Palpatine", 

            "Darth Vader", 

            "Lando Calrissian", 

            "Boba Fett", 

            "C-3P0", 

            "R2 D2", 

            "Jar Jar Binks", 

            "Padme Amidala",

            "Yoda"]





favorite_character = star_wars[star_wars[fav_list] == 1]

        

fig = plt.figure(figsize = (7,6))

plt.bar(range (14), favorite_character[favorite_character.columns[15:29]].sum())
star_wars["Luke Skywalker"].value_counts(dropna=False)

#star_wars[star_wars[fav_list] == 1].sum()