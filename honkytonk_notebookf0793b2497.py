import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline
reader = pd.read_csv('../input/Video_Game_Sales_as_of_Jan_2017.csv')
gameData = pd.DataFrame(reader)
gameData
plt.figure(figsize=(10,10))
plt.plot(gameData['Critic_Score'], gameData['Global_Sales'], '.', alpha = 0.5)
plt.xlabel('Critic Score')
plt.ylabel('Gloabl Sales')
plt.title('Does Critic Score affect Global Sales?')
plt.figure(figsize=(10,10))
plt.plot(gameData['User_Score'], gameData['Global_Sales'], '.', alpha = 0.5)
plt.xlabel('User Score')
plt.ylabel('Gloabl Sales')
plt.title('Does User Score affect Global Sales?')
gameData.corr()
fig, ax = plt.subplots()
fig.set_size_inches(6, 6)
ratingCountPlot = sb.countplot(x='Rating',data=gameData)
ratingCountPlot.set_ylabel("Number of Games")
ratingGameData = gameData[gameData['Rating'].isin(['E','E10+','M','T'])]
ratingGameData[['Global_Sales']].groupby(ratingGameData['Rating']).mean()

fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
genreCountPlot = sb.countplot(x='Genre',data=gameData)
genreCountPlot.set_ylabel("Number of Games")
gameData[['Global_Sales']].groupby(gameData['Genre']).mean()
gameData.groupby(gameData['Platform']).count()
platformGameData = gameData[gameData['Platform'].isin(['PC','3DS','PS4','WiiU','XOne'])]
platformGameData[['Global_Sales']].groupby(platformGameData['Platform']).mean()
