

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



ACC = pd.read_csv("../input/ACC_AUX.CSV") #reads the file

ACC1 = ACC[["A_REGION", "A_HR"]] #isolate the relevant columns from the original data

ACC2 = ACC1.sort_values(by=["A_REGION"]) #sort numerically by region

ACC2["A_HR"].replace(2, 0, inplace=True) #replace 2s with 0s

# ACC2["###CCC###"].replace(3, 0, inplace=True) #include if there are three options



def data_frame_split(number): #split data frame by region

    return ACC2[ACC2["A_REGION"] == number]

    

def just_empty_list(): #put each separated region into a list

    empty_list = [data_frame_split(x) for x in range(1,11)]

    return empty_list



def list_sums_part_1(number): #find the sum of 1's in a given region's matched column 

    list_1 = just_empty_list()

    return sum(list_1[number]["A_HR"])



def list_of_sums(): #create a list of said sums in each region

    final = [list_sums_part_1(x) for x in range(0,10)]

    return final



def length_of_list(number): #find the length of each list (needed to calculate ratios)

    list_1 = just_empty_list()

    lol = len(list_1[number])

    return lol



def list_of_lol(): #create a list of the lengths of each dataset

    list_of_lol = [length_of_list(x) for x in range(0,10)]

    return list_of_lol



def weighted_list(): #find the weighted ratios of each element by region

    lista = list_of_sums()

    print(lista)

    listb = list_of_lol()

    print(listb)

    final = [a/b for a,b in zip(lista,listb)]

    print(final)

    return final



list_x_varible = list(range(1,11)) #creates list 1 through 10

weighted_ratios_y_varible = weighted_list() #stores a varible to the list of weighted ratios



plt.xlabel("Region 1 through 10") #x-label for bar graph

plt.ylabel("Ratio of FCCs with a Hit and Run to Total FCC ") #y-label for bar graph

plt.title("Hit and Run FCC Ratios by Region") #title of the bar graph

plt.xticks(np.arange(min(list_x_varible), max(list_x_varible)+1, 2.0)) #lists x-ticks by 1's

ax = plt.subplot(111) #setting 'ax' varible for later command

barWidth = .75 #setting bar width to be used in next command

ax.bar(list_x_varible, weighted_ratios_y_varible, width=barWidth, align='center', color = "blue", alpha = .2) 

#^the function that actual graphs the bar chart^

plt.show()