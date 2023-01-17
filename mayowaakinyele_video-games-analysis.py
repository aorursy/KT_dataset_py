from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
video_game_records = pd.read_csv("../input/vgsales/vgsales.csv")
video_game_records
Total_North_America_Sales = round(sum(video_game_records.NA_Sales) * 1000000)
Total_North_America_Sales
Total_Europe_Sales = round(sum(video_game_records.EU_Sales) * 1000000)
Total_Europe_Sales
Total_Japan_Sales = round(sum(video_game_records.JP_Sales) * 1000000)
Total_Japan_Sales
Total_Other_Sales = round(sum(video_game_records.Other_Sales) * 1000000)
Total_Other_Sales
video_game_records.describe()
Genres_total_games = video_game_records.groupby("Genre").size()
Genres_total_games 
Genres_total_games.plot.pie(autopct="%0.0f%%",  radius=2)
plt.show()
                              # AFTER FILTERING THE DATAFRAME ABOVE USING MYSQL

#Genre - Genre of the game

#NORTH_AMERICA SALES - Sales in North America (in millions)

#EUROPE_SALES - Sales in Europe (in millions)

#JAPAN_SALES - Sales in Japan (in millions)

#OTHER_COUNTRY_SALES - Sales in the rest of the world (in millions)

#GLOBAL_SALES - Total worldwide sales.
video_game_data = pd.read_csv("../input/video-game-sales/Video_game_sales.csv")
video_game_data
# GLOBAL SALES PERCENTAGE AMONG GENRES


plt.pie(video_game_data.GLOBAL_SALES, labels = video_game_data.Genre, radius = 2, autopct = "%0.0f%%")
plt.show()
                    #VISUAL REPRESENTATION OF THE TOTAL SALES AMONG FIVE VARIABLES OF DIFERENT GENRE
    



plt.figure(figsize = (15, 10))
plt.plot(video_game_data.Genre, video_game_data.NORTH_AMERICA_SALES, marker = "o")
plt.plot(video_game_data.Genre, video_game_data.EUROPE_SALES, marker = "*")
plt.plot(video_game_data.Genre, video_game_data.JAPAN_SALES, marker = "s")
plt.plot(video_game_data.Genre, video_game_data.OTHER_COUNTRIES_SALES, marker = "v")
plt.plot(video_game_data.Genre, video_game_data.GLOBAL_SALES, marker = "P")
plt.xlabel("Genre Type")
plt.ylabel("Sales(Millions) ")
plt.title("VIDEO GAME SALES TREND")
plt.legend(["NORTH AMERICA","EUROPE","JAPan","OTHER","GLOBAL"])
plt.show()
