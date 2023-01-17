import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



imdb = pd.read_csv("../input/imdb-data/IMDB-Movie-Data.csv") #reading dataset and showing info about it

imdb.info() 
imdb.columns
print("HEAD HEAD HEAD HEAD HEAD HEAD \n")

print(imdb.head(5))  #first 5 data from dataset

print("\n TAIL TAIL TAIL TAIL TAIL TAIL \n")

print(imdb.tail(5))  #last 5 data from dataset
#creating line plot



imdb.Rating.plot(kind = "line" , color = "green" , label = "Rating of the movie",  

                linewidth = 1,linestyle = ":")

imdb.Metascore.plot(kind = "line",color = "grey",alpha = 0.5,label = "Metascore of the movie",

                  linewidth = 1,linestyle = "--")

plt.xlabel("amount of the movies")

plt.ylabel("rates of the movies")

plt.legend()

plt.show()

#creating scatter plot

imdb.plot(kind = "scatter",x= "Votes",y ="Metascore",

          label = "metascore / votes",linewidth = 1,alpha = 0.5) 

plt.legend()

plt.show()
#creating histogram plot

imdb.Rating.plot(kind = "hist", label = "Rating",linewidth = 1,

                 bins = 50,alpha = 0.6,color = "orange",figsize = (6,6)) 

plt.xlabel("Rating")

plt.legend()

plt.show()
dic= {'nick' : 'zeno','city' : 'yalova'}

dic["age"] = 24 #adding new value

print(dic)

dic["city"] = "kastamonu" #updating data

print(dic)

del dic["city"]

print(dic) #removing value

print("zeno" in dic) #checking data
get_director_as_serie = imdb["Director"] == "James Gunn" # filtering data

imdb[get_director_as_serie]
imdb[(imdb["Director"] == "James Gunn") & (imdb["Rating"] > 6.8)] # multi filtering data 