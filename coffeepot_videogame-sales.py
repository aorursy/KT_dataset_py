import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt # plotting libs

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
vgsales = pd.read_csv("../input/Video_Games_Sales_as_at_22_Dec_2016.csv")
type(vgsales)
print(vgsales.shape) # we can see the shape of our data by invoking the DataFrame's shape field

vgsales.head(5) # we can see a snapshot of our data by calling the DataFrame's head method
print(vgsales.columns)

print(type(vgsales.columns))
print("Platforms (Unique):")

platforms = pd.unique(vgsales.Platform)

print(platforms)



print("\nYears (Range): " + str(min(vgsales.Year_of_Release)) + " - " + str(max(vgsales.Year_of_Release)))



print("\nGenres (Unique):")

genres = pd.unique(vgsales.Genre)

print(genres)



print("\nNA_Sales (Range): " + str(min(vgsales.NA_Sales)) + " - " + str(max(vgsales.NA_Sales)))

print("EU_Sales (Range): " + str(min(vgsales.EU_Sales)) + " - " + str(max(vgsales.EU_Sales)))

print("JP_Sales (Range): " + str(min(vgsales.JP_Sales)) + " - " + str(max(vgsales.JP_Sales)))

print("Other Sales (Range): " + str(min(vgsales.Other_Sales)) + " - " + str(max(vgsales.Other_Sales)))

print("Global Sales (Range): " + str(min(vgsales.Global_Sales)) + " - " + str(max(vgsales.Global_Sales)))



print("\nCritic Score (Range): " + str(min(vgsales.Critic_Score)) + " - " + str(max(vgsales.Critic_Score)))
# fixing our little dating issue

vgsales = vgsales.set_value(5936, "Year_of_Release", 2009)

vgsales.loc[5936]
nintendo_games = vgsales.loc[vgsales["Publisher"] == "Nintendo"]

nintendo_games.head(5)
_ = plt.pie(nintendo_games.Genre.value_counts(), 

            labels = nintendo_games.Genre.unique(), 

            autopct='%1.1f%%',

            shadow=True)

_ = plt.title("Nintendo by Genre")
lbs = nintendo_games.Rating.dropna().unique()



_ = plt.pie(nintendo_games.Rating.value_counts(),

           labels = lbs,

           autopct='%1.1f%%',

           shadow=True)

_ = plt.title("Nintendo by Rating")
vgsales.loc[(vgsales.Publisher == "Nintendo") & (vgsales.Rating == "M")]
Top_JP = vgsales.loc[(vgsales.JP_Sales > vgsales.NA_Sales) & (vgsales.JP_Sales > vgsales.EU_Sales) & (vgsales.JP_Sales > vgsales.Other_Sales)]

Top_NA = vgsales.loc[(vgsales.NA_Sales > vgsales.JP_Sales) & (vgsales.NA_Sales > vgsales.EU_Sales) & (vgsales.NA_Sales > vgsales.Other_Sales)]

Top_EU = vgsales.loc[(vgsales.EU_Sales > vgsales.NA_Sales) & (vgsales.EU_Sales > vgsales.JP_Sales) & (vgsales.EU_Sales > vgsales.Other_Sales)]

Top_Other = vgsales.loc[(vgsales.Other_Sales > vgsales.EU_Sales) & (vgsales.Other_Sales > vgsales.NA_Sales) & (vgsales.Other_Sales > vgsales.JP_Sales)]

print("Number of games that sold best in Japan: " + str(len(Top_JP)))

print("Number of games that sold best in Europe: " + str(len(Top_EU)))

print("Number of games that sold best in North America: " + str(len(Top_NA)))

print("Number of games that sold best in Other: " + str(len(Top_Other)))
Top_NA.head(3)
Top_JP.head(3)
Top_EU.head(3)
Top_Other.head(3)
cs_counts = vgsales.Critic_Score.groupby(vgsales.Year_of_Release).count()

print(cs_counts)

plt.plot(cs_counts)

plt.title("Number of Games w/Critic Scores 1980 - 2017")

plt.xlabel("Year")

plt.ylabel("# of Games")


print(len(vgsales.Genre.value_counts()))

lbs = vgsales.Genre.dropna().unique()

# lbs = lbs.resize((1,12))

print(lbs)

_ = plt.pie(vgsales.Genre.value_counts(),

           labels = lbs)
lbs = vgsales.Rating.dropna().unique()

_ = plt.pie(vgsales.Rating.value_counts(),

           labels = lbs)
vgsales.loc[(vgsales.Publisher == "Nintendo") & (vgsales.Genre == "Shooter")]