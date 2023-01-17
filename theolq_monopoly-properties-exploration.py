import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns ; sns.set()



import os

os.chdir("../input/french-monopoly")



df=pd.read_csv('French_Monopoly.csv')
from random import randint

from collections import Counter



jail_case_number = 30

def new_movement(initial_position):

    new_position = initial_position + randint(1,6)

    new_position = new_position % 40

    if new_position == jail_case_number : 

        new_position = 10

    return new_position

    

def game(throw_number_of_dice):

    historic_position = [0]

    position = 0         #Start

    for i in range(throw_number_of_dice+1):

        position = new_movement(position)

        historic_position.append(position)

    dict_number_per_case = Counter(historic_position)

    case = range(len(dict_number_per_case))

    number_per_case = []

    for index in range(len(dict_number_per_case)):

        number_per_case.append(dict_number_per_case[index])

    return case, number_per_case

    

def study(number_of_games, throw_number_of_dice):

    for games in range(number_of_games):

        case, number_per_case = game(throw_number_of_dice)

        plt.plot(case, number_per_case,'o')

    plt.title("On which case do we fall in Monopoly ?")

    plt.show()

    

number_of_games = 5

throw_number_of_dice = 10000

study(number_of_games, throw_number_of_dice)
study(number_of_games,50000)
df.head(10)
b=df.loc[df.Property.isin(["Lecourbe","Republique","Paradis","Pigalle","Henri_Martin","La_Fayette","Capucines","Paix"])]

b.head()
order=["House_0","House_1","House_2","House_3","House_4","Hotel"]

plt.xticks(range(len(order)),order)



for i in range(b.shape[0]):

    plt.plot(list(b.iloc[i,1:7]),c=b.Color.iloc[i])

    plt.plot(list(b.iloc[i,1:7]),'o',label=b.iloc[i,0],c=b.Color.iloc[i])

plt.title("Money earned in 1 passage in function of the number of houses")

plt.legend()

plt.show()
plt.xticks(range(len(order)),order)



for i in range(b.shape[0]):

    z=list(b.iloc[i,1:7])

    Y=[z[0]-b.Price.iloc[i],z[1]-b.House_Price.iloc[i]-b.Price.iloc[i],z[2]-2*b.House_Price.iloc[i]-b.Price.iloc[i], z[3]-3*b.House_Price.iloc[i]-b.Price.iloc[i],z[4]-4*b.House_Price.iloc[i]-b.Price.iloc[i],z[5]-5*b.House_Price.iloc[i]-b.Price.iloc[i]]

    plt.plot(Y,c=b.Color.iloc[i])

    plt.plot(Y,'o',label=b.iloc[i,0],c=b.Color.iloc[i])



plt.title("Money earned/lost in 1 passage in function of the number of houses")

plt.legend()

plt.show()