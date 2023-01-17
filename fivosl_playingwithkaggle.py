print('first time writing code on Kaggle')
# Entering values into list 
my_list = [20, 30, 15, 40] 

# Iterating through list 
for i in my_list:
    print (i)
# Display whole array
print(my_list)
# Looping and referring to each item in list 
for counter in range(len(my_list)):
    print(counter, my_list[counter])
# Adding a nested conditional inside loop 
for counter in range(len(my_list)):
    if my_list[counter]>20:
        print(counter,'Higher than twenty')
    elif my_list[counter]==20:
        print(counter,'It is twenty')
    else:

        print(counter,'It is lower than twenty')
import pandas as pd
import os

# Current directory 
print(os.getcwd())

# Reading excel file
dataset = pd.read_excel('/kaggle/input/Kaggle_Dataset.xlsx')

# Displaying dataset 
print(dataset)