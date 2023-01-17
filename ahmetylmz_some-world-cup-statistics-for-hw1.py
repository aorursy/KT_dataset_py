# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/FIFA 2018 Statistics.csv")
data.head(10)
data[data["Team"] == "Russia"]# Statistics of Russian Team
data.columns = [each.split()[0]+"_"+each.split()[1] if (len(each.split())>1) else each for each in data.columns]# rearranged the name of data columns
dataRussia = data[data["Team"] == "Russia"]
#Bar plot of Russian team's goal scored against its opponent
dataRussia.plot(kind="bar", x="Opponent", y="Goal_Scored" ,color="green",figsize = (10,10), label = "Goal Scored")
plt.xlabel("Opponent")
plt.ylabel("Goal Scored")
plt.title("Goal Statistics of Russia")
plt.show()
dataCorr = data.loc[:,"Goal_Scored":"Red"]# correlation of some columns

f,ax = plt.subplots(figsize=(18, 18))# tablodaki karalerin boyutunu ayarlar
sns.heatmap(dataCorr.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax) # heatmap ise bu correlation ilişkisini güzel bi tabloya dönüştürür normalde excel gibi bi tablo. Annot tablo içindeki sayıları gösterir. fmt ise virgül hassasiyeti
plt.show()
#Scatter plot of attempts and goal scored in the tournament
data.plot(kind="scatter", x="Attempts", y="Goal_Scored", alpha = 0.7, color="blue", figsize = (10,10))
plt.xlabel("Attempts")
plt.ylabel("Goal Scored")
plt.title("Attempts & Goal Scored Scatter Plot")
plt.show()
dataBrazil = data[data["Team"] == "Brazil"] # Statistics of Brazil Team
#Line plot of Pass Accuracy and Ball Possession of Brazil Team
dataBrazil.Pass_Accuracy.plot(kind="line",label = "Pass Accuracy", color="pink", grid=True, figsize = (10,10))
dataBrazil.Ball_Possession.plot(kind="line", label="Ball Possession", color="black", grid=True)
plt.legend(loc='center right') 
plt.xlabel("Pass Accuracy %")
plt.ylabel("Ball Possession %")
plt.title("Ball Possession % & Pass Accuracy % Line Plot of Brazil Team")
plt.show()
# Histogram of Goal Scored in the tournament
data.Goal_Scored.plot(kind = 'hist', bins=20, figsize = (10,10), label ="Goal Scored", color="orange")
plt.xlabel("Goal Scored")
plt.legend(loc="upper right")
plt.title("Goal Scored in the Tournament")
plt.show()