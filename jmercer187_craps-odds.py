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

# Any results you write to the current directory are saved as output.
np.random.seed(187)

# create function that rolls dice
def dice_roll():
    """Returns the cumulative result of two d6 dice"""
    die1 = np.random.randint(1,7)
    die2 = np.random.randint(1,7)
    rollresult = die1 + die2
    
    return rollresult

# create function that runs game
def simple_craps_game(bettype, betamount, chipcount):
    """plays a simple game of craps, takes type of bet (pass / don't pass), bet amount, and current chip count as args, returns updated chip count based on game results"""
    # get come out dice roll
    comeoutroll = dice_roll()
    
    # determine results from come out roll
    if (comeoutroll in [7, 11] and bettype == "Pass") :
         return betamount + chipcount    
    elif (comeoutroll in [2, 3, 12] and bettype == "Pass") :
        return chipcount - betamount    
    elif (comeoutroll in [7, 11] and bettype == "Don't Pass") :
        return chipcount - betamount   
    elif (comeoutroll == 12 and bettype == "Don't Pass") :
        return chipcount
    
    point = comeoutroll
    
    # roll for point    
    pointroll = dice_roll()
    
    # determine results if rolling for the point    
    if bettype == "Pass" :
        while (pointroll != point or pointroll != 7) :
            pointroll = dice_roll()           
            if pointroll == point :
                return betamount + chipcount            
            elif pointroll == 7 :
                return chipcount - betamount
        
    if bettype == "Don't Pass" :
        while (pointroll != point or pointroll != 7) :
            pointroll = dice_roll()            
            if pointroll == point :
                return chipcount - betamount            
            elif pointroll == 7 :
                return chipcount + betamount
# set up random walk through X number of games for Pass and Don't Pass bet types
all_pass = []
all_dontpass = []

# play with the random walks
number_of_throws = 100
number_of_iterations = 10000

# PASS BET
for y in range(number_of_iterations) :
    chip_count_result = 100
    random_walk_pass = []
    for x in range(number_of_throws) :
        chip_count_result = simple_craps_game("Pass",5,chip_count_result)
        random_walk_pass.append(chip_count_result)     
    all_pass.append(random_walk_pass)

all_pass_t = np.transpose(np.array(all_pass))   
all_pass_final_chip_count = all_pass_t[-1,:]

print("Average Chip Count =", np.average(all_pass_final_chip_count),"\nMedian Chip Count =", np.median(all_pass_final_chip_count))
plt.hist(all_pass_final_chip_count,20,edgecolor='black')
plt.title("Pass Bet Histogram")
plt.xlabel("Final Chip Count")
plt.show()


# DON'T PASS BET
for y in range(number_of_iterations) :
    chip_count_result = 100
    random_walk_dontpass = []
    for x in range(number_of_throws) :
        chip_count_result = simple_craps_game("Don't Pass",5,chip_count_result)
        random_walk_dontpass.append(chip_count_result)
    all_dontpass.append(random_walk_dontpass)
    
all_dontpass_t = np.transpose(np.array(all_dontpass))
all_dontpass_final_chip_count = all_dontpass_t[-1,:]

print("Average Chip Count =", np.average(all_dontpass_final_chip_count),"\nMedian Chip Count =", np.median(all_dontpass_final_chip_count))
plt.hist(all_dontpass_final_chip_count,20,edgecolor='black')
plt.title("Don't Pass Bet Histogram")
plt.xlabel("Final Chip Count")
plt.show()
all_field = []

number_of_throws = 100
number_of_iterations = 10000

for y in range(number_of_iterations):

    field_walk = []
    chip_count = 100
    bet_amount = 5
    
    for x in range(number_of_throws) :
        roll = dice_roll()
        if roll in [3, 4, 9, 10, 11] :
            chip_count = chip_count + bet_amount
        elif roll in [5, 6, 7, 8] :
            chip_count = chip_count - bet_amount
        elif roll in [2, 12] :
            chip_count = chip_count + (bet_amount * 2)

        field_walk.append(chip_count)
    all_field.append(field_walk)
    
all_field_t = np.transpose(np.array(all_field))
field_final_chip_count = all_field_t[-1,:]

print("Average Chip Count =", np.average(field_final_chip_count),"\nMedian Chip Count =", np.median(field_final_chip_count))
plt.hist(field_final_chip_count,20,edgecolor='black')
plt.title("Field Bet Histogram")
plt.xlabel("Final Chip Count")
plt.show()

    
    
    
    
    
    