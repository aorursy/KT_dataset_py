# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

pd.set_option('max_columns',None)
pd.options.display.width = 500
data = pd.read_csv("../input/ign.csv")
print("\nSample Data")
print(data.iloc[1:500:3][["title","score","score_phrase","genre","platform","editors_choice"]])
print("\nNumber of platforms")
print(data["platform"].unique().size)

#Q1 Playstation vs Xbox (Which PF has more high rating games)?
data["score_round"] = data["score"].round(0)
ratings = sorted(data["score_round"].unique())
data_pf_ratings = pd.DataFrame({"ratings" : ratings})
data_pf_ratings["playstation_count"] = data[data["platform"].str.contains("PlayStation")]["score_round"].value_counts()
data_pf_ratings["xbox_count"] = data[data["platform"].str.contains("Xbox")]["score_round"].value_counts()
#plot1
plt.figure()
data_pf_ratings.plot(kind="bar", title="PlayStation vs Xbox",y=["playstation_count","xbox_count"],edgecolor="black",linewidth=1.2)
plt.draw()

#Q2 Is rating of a game dependent on editor's choice?
data.loc[data["editors_choice"] == "Y","editors_choice_int"] = 1 # correlation can be calculated between numerical fields, so converting
data.loc[data["editors_choice"] == "N","editors_choice_int"] = 0
print("\nCorrelation between score and editors_choice")
print("Option 1\n")
print(data["score"].corr(data["editors_choice_int"]))
print("Option 2\n")
print(data.loc[1:50,["score","editors_choice_int"]].corr())
print("Option 3\n")
print(data.corr())

#Q3 get all games in 1996 by playstation with score > 7?
print(data.loc[(data["platform"].str.contains("PlayStation")) & (data["release_year"] == 1996) & (data["score"] > 7),["title","score","platform","genre","editors_choice","release_year"]])

#Q4 find change in game genre(Action) with year?
data2 = data.loc[(data["genre"].str.contains("Action", case=False, na=False)) | data["genre"].str.contains("Fight", case=False, na=False)]
df_year_to_genre = data2.groupby("release_year")["title"].agg("count").to_frame("count").reset_index()
plt.figure()
df_year_to_genre.set_index("release_year").plot(kind="bar", title="Action games per year")
plt.draw()

#Q5 what genre gets best ratings(7-10)?
data_7_10 = data.loc[(data["score"] >= 7)]
dataTemp = data_7_10["genre"].value_counts()
plt.figure()
dataTemp[:5].plot(kind="bar", title="Top 5 rated game genres")
plt.draw()

#Q6 Number of games per rating for Action/Fightng genre?
print("\nNumber of Action games per rating score")
print(data.loc[(data["genre"].str.contains("Action", case=False, na=False)) | data["genre"].str.contains("Fight", case=False, na=False)].groupby("score_round")["genre"].agg("count"))

#Q7 How many games got better rating for PS vs How many got better rating for Xbox?
print("\n How many games got different ratings for PS and Xbox")
print(data.groupby("title")["score"].agg(np.ptp))

#Q8 Number of games released for each pf?
data_highest_release_pf = data["platform"].value_counts()
plt.figure()
data_highest_release_pf[:10].plot(kind="bar", title="Highest release platforms")
plt.draw()

#Q9 During what time of year most games are released?
data_month_grp = data["release_month"].value_counts().to_frame("count").reset_index()
plt.figure()
data_month_grp = data_month_grp.sort_values(by="index")
data_month_grp.index = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
data_month_grp.plot(kind="bar", y="count", title="Month wise release count")
plt.draw()

#Q10 Number of games relased each year?
plt.figure()
data["release_year"].value_counts().plot(kind="bar",title="Games released each year")
plt.draw()

plt.show()

# Any results you write to the current directory are saved as output.
