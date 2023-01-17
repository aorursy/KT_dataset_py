import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb



import os

print(os.listdir("../input"))
data=pd.read_csv("../input/top2018.csv")
# First, we look at the dataset info

data.info()

# We have 16 columns. 13 of them are float and 3 of them are object variable. We have 100 rows.
# Is there a null observation in our data? Lets look at that.

data.isnull().sum()

# We see that there is no null value in our dataset.
# We look the first 10 rows of dataset.

data.head(10)
# We look last 10 rows of datasets.

data.tail(10)
data.columns

# These are the column names of data.
# Now, we look at our dataset mean,standard deviation, max value and min. value

data.describe()

# When we look at our describtions we can say that most of the variable cumulate maximum values. 

# In order to say that there are few values close to minimum.
# Now we look at the correlations of each other.

data.iloc[:,3:16].corr()

# We can say that energy and loudness is highly positive correlated. Also, energy and acousticness is negatively correlated and so on.
# Now we look at the correlations of each other.

f,ax=plt.subplots(figsize=(20,10))

sb.heatmap(data.iloc[:,3:16].corr(),annot=True,linewidth=.5,fmt=".2f",ax=ax)

plt.title("Correlations of Variables")

plt.show()
# We see that the previous graphy energy and loudness are positively correlated. Then, we make their scatter plots.

plt.rcParams['axes.facecolor'] = 'w'

data.plot(kind="scatter",x="energy",y="loudness",color="black",figsize=(15,8))

plt.title("Scatter Plot of Energy and Loudness")

plt.xlabel("Energy")

plt.ylabel("Loudness")

plt.show()
#We also look the almost uncorrelated variables which are duration_ms and tempo.

data.plot(kind="scatter",x="tempo",y="duration_ms",color="black",figsize=(15,8))

plt.title("Scatter Plot of Mode and Tempo")

plt.xlabel("Duration_ms")

plt.ylabel("Tempo")

plt.show()
# If I increase the tempo, Is tempo more correlated with energy?

data["increased_tempo"]=data.tempo*1.5

data.head()
# Lets find that tempo is more correlated with energy or not.

print(data.iloc[:,[4,13]].corr())

print(data.iloc[:,[4,16]].corr())

# We see that there is no change if we increase the tempo.
# Now, we look at the which artist's songs are in our spotify tracks.

#Firstly, we need to count the values after that we select the top 10 numbers and plot the bar graph.

data['artists'].value_counts()[:10].plot(kind='barh',figsize=(15,6))

plt.title("Top 10 Artists in Spotify Tracks of 2018")

plt.xlabel("Number of Occurence")

plt.gca().invert_yaxis() # inverted the y axis.

plt.show()

# We see that Post Malone and XXTENTACION is the most seeing in the tracks.
plt.rcParams['axes.facecolor'] = 'w'

data["danceability"].plot(kind="line",color="r",label="danceability",linewidth=1,linestyle="--",alpha=.5,grid=False,figsize=(15,7))

data.energy.plot(kind="line",color="black",label="energy",linewidth=1,alpha=.5,grid=False,figsize=(15,7))

legend = plt.legend(loc="best")

for text in legend.get_texts():

    plt.setp(text, color = 'black')

plt.xlabel("Index")

plt.ylabel("Danceability and Energy")

plt.title("Danceability and Energy Line Graph")

plt.show()

# In this plot, we see that danceability and energy are acting differently. When danceability moves up, energy was decreased or vice versa. 