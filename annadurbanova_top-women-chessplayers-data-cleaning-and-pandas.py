import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import glob

import os

%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

data=pd.read_csv('/kaggle/input/top-women-chess-players/top_women_chess_players_aug_2020.csv')
#data=pd.read_csv("Top_Women_Chessplayer.csv")



#--------Filling NA with 0-----------------#



data["Year_of_birth"]=data["Year_of_birth"].fillna(0)

data["Title"]=data["Title"].fillna(0)

data["Rapid_rating"]=data["Rapid_rating"].fillna(0)

data["Blitz_rating"]=data["Blitz_rating"].fillna(0)

data["Inactive_flag"]=data["Inactive_flag"].fillna(0)



#-----------Replacing values-----------------#



title_replacement={"GM":"Grandmaster",

                   "IM":"International Master",

                   "FM": "FIDE Master",

                   "CM": "Candidate Master",

                   "WFM": "Woman FIDE master",

                   "WCM": "Woman Candidate Master", }



data["Title"].replace(title_replacement, inplace=True)

data["Gender"].replace({"F":"Female"}, inplace=True)

data["Inactive_flag"].replace({"wi":"Woman Inactive"}, inplace=True)



#-----------From float to Int-----------------#





data["Year_of_birth"]=data["Year_of_birth"].astype(np.int64)

data["Standard_Rating"]=data["Standard_Rating"].astype(np.int64)

data["Rapid_rating"]=data["Rapid_rating"].astype(np.int64)

data["Blitz_rating"]=data["Blitz_rating"].astype(np.int64)



#-----------Finding out the country for Alpha3 Code-----------------#



url="https://www.iban.com/country-codes"

dfl=pd.read_html(url,header=0)[0]

data=pd.merge(data,dfl, left_on="Federation", right_on="Alpha-3 code")

data.drop(["Alpha-2 code", "Alpha-3 code", "Numeric"], axis=1)





#-----------Reorganizing columns-----------------#





columns_name=["Fide id", "Name", "Federation", "Country", "Gender", "Year_of_birth", "Title", "Standard_Rating", "Rapid_rating", "Blitz_rating", "Inactive_flag"]

data=data.reindex(columns=columns_name)



data
data.dtypes
#pd.options.display.float_format = '{:.2f}%'.format



def missing_values(n):

    df=pd.DataFrame()

    df["missing, %"]=data.isnull().sum()*100/len(data.isnull())                           

    df["missing, num"]=data.isnull().sum()

    return df.sort_values(by="missing, %", ascending=False)

missing_values(data)



data.corr().style.background_gradient()
plt.rcParams["figure.figsize"]=[16,9]

sns.heatmap(data.corr(), annot=True, cmap="BuPu")
mask=(data["Rapid_rating"]!=0) & (data["Blitz_rating"]!=0)

sns.regplot(data=data[mask], x="Rapid_rating", y="Blitz_rating")

(data

 .groupby("Name")

 [["Standard_Rating", "Rapid_rating", "Blitz_rating"]]

 .max()

 .sort_values(by=["Standard_Rating", "Rapid_rating", "Blitz_rating"], ascending=False)

)
data["Country"].value_counts()
data["Name"].value_counts()
data[data["Year_of_birth"]!=0][["Name", "Year_of_birth", "Title", "Country"]].sort_values(by="Year_of_birth", ascending=True)

data[data["Year_of_birth"]!=0][["Name", "Year_of_birth", "Title", "Country"]].sort_values(by="Year_of_birth", ascending=False)
(data[data["Title"]!=0]

 .groupby("Title")

 [["Standard_Rating", "Rapid_rating", "Blitz_rating"]]

 .mean()

 .sort_values(by="Standard_Rating", ascending=False)

)
(data[data["Year_of_birth"]!=0]

 .groupby("Year_of_birth")

 [["Standard_Rating"]]

 .mean()

 .sort_values(by="Standard_Rating", ascending=True)

 .head(10)

)

(data[data["Year_of_birth"]!=0]

 .groupby("Year_of_birth")

 [["Standard_Rating"]]

 .mean()

 .sort_values(by="Standard_Rating", ascending=False)

 .head(10)

)
