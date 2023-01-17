import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

from numpy import mean
plt.style.use('ggplot')

data = pd.read_csv('../input/chess/games.csv')

data.columns



data.dtypes
data.head()
data.nunique()
plt.figure(figsize=(15,10))

sb.countplot(x='winner', data=data)
plt.figure(figsize=(15,10))

sb.countplot(x='winner', hue='victory_status', data=data)
white = data.loc[data['winner']=='white']

black = data.loc[data['winner']=='black']

white['victory_status'].value_counts().plot.pie(autopct="%.1f%%", label="White Wins")

black['victory_status'].value_counts().plot.pie(autopct="%.1f%%", label="Black Wins")
WhiteTop10Openings = white['opening_name'].value_counts().nlargest(10)

plt.figure(figsize=(15,10))

df = white[white['opening_name'].isin(WhiteTop10Openings.index)]

sb.countplot(x='opening_name', data=df)

plt.xticks(rotation=90, fontsize=12)

plt.ylabel("# Games Won as White", fontsize=15)

plt.yticks(fontsize=16)

plt.title("White # Win vs Openings")
plt.figure(figsize=(15,10))

openingSet = data[data['opening_name'].isin(WhiteTop10Openings.index)]

woS = df.groupby(['opening_name']).count()

oS = openingSet.groupby('opening_name').count()

perc = (woS/oS)*100

sb.barplot(x=perc.index , y=perc.id)



plt.xticks(rotation=90, fontsize=12)

plt.ylabel("% Games Won as White", fontsize=15)

plt.yticks(fontsize=16)

plt.title("White % Win vs Openings")
BlackTop10Openings = black['opening_name'].value_counts().nlargest(10)

plt.figure(figsize=(15,10))

df = black[black['opening_name'].isin(BlackTop10Openings.index)]

sb.countplot(x='opening_name', data=df)

plt.xticks(rotation=90, fontsize=12)

plt.ylabel("# Games Won as Black", fontsize=15)

plt.yticks(fontsize=16)

plt.title("Black # Win vs Openings")
plt.figure(figsize=(15,10))

openingSet = data[data['opening_name'].isin(BlackTop10Openings.index)]

boS = df.groupby(['opening_name']).count()

oS = openingSet.groupby('opening_name').count()

perc = (boS/oS)*100

sb.barplot(x=perc.index , y=perc.id)



plt.xticks(rotation=90, fontsize=12)

plt.ylabel("% Games Won as Black", fontsize=15)

plt.yticks(fontsize=16)

plt.title("Black % Win vs Openings")
plt.figure(figsize=(20,10))

data['ratingDiff'] = data['white_rating'] - data['black_rating']

sb.catplot(x='winner', y='ratingDiff', kind='boxen', data=data)

plt.ylabel("Rating Difference (White-Black)")

plt.figure(figsize=(20,15))

sb.scatterplot(x='turns', y="ratingDiff", data=data)

plt.xticks(fontsize=16)

plt.xlabel("# of Turns", fontsize=15)

plt.ylabel("Rating Difference (White-Black)", fontsize=15)

plt.yticks(fontsize=16)

plt.figure(figsize=(20,20))

data['AbsDiff'] = data['ratingDiff'].abs()

sb.regplot(x='turns', y='AbsDiff', x_estimator=mean, ci=False, data=data)

plt.xticks(fontsize=16)

plt.xlabel("# of Turns", fontsize=15)

plt.ylabel("|Rating Difference|", fontsize=15)

plt.yticks(fontsize=16)
plt.figure(figsize=(15,10))

df = data.loc[data['ratingDiff']>250]

blackWins = df.loc[df['winner']=='black']

mostWins = blackWins['opening_name'].value_counts().nlargest(10)

sb.barplot(x=mostWins.index, y=mostWins.values)

plt.xticks(rotation=90, fontsize=16)

plt.ylabel("Games Black Won", fontsize=15)

plt.yticks(fontsize=16)

plt.xlabel("Opening", fontsize=15)

plt.title("Games Black Won With -250 Rating on White")
plt.figure(figsize=(15,10))

df = data.loc[data['ratingDiff']<-250]

whiteWins = df.loc[df['winner']=='white']

mostWins = whiteWins['opening_name'].value_counts().nlargest(10)

sb.barplot(x=mostWins.index, y=mostWins.values)

plt.xticks(rotation=90, fontsize=16)

plt.ylabel("Games White Won", fontsize=15)

plt.yticks(fontsize=16)

plt.xlabel("Opening", fontsize=15)

plt.title("Games White Won With -250 Rating on Black")