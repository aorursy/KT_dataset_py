#import necessary libraries

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt



#read in cereal.csv and isolate the manufacturer column

cereal = pd.read_csv("../input/cereal.csv")

manufacturer_data = cereal["mfr"]



#count number of cereals for each manufacturer 

manufacturers = ['American Home \n Food Products', 'General Mills', 'Kelloggs', 'Nabisco', 'Post', 'Quaker Oats', 'Ralston Purina']

y_pos = np.arange(len(manufacturers))

count = [0,0,0,0,0,0,0]

for x in manufacturer_data:

    for y in manufacturers:

        if x == y[0]:

            count[manufacturers.index(y)] += 1



#draw bar chart of manufacturer data

plt.bar(y_pos, count, align='center') #draw bar chart

plt.xticks(y_pos, manufacturers, rotation = 45, fontsize = 8) #set labels for each column

plt.ylabel("Numbers of products") #set y axis label

plt.xlabel("Cereal manufacturers") #set x axis label

plt.title("Number of Cereals by Manufacturer") #set chart title

plt.show() #remove text output
#draw pie chart of manufacturer data

plt.pie(count, labels=manufacturers, autopct='%1.1f%%', startangle = 150) #draw pie chart & add percentages on each slice

plt.axis('equal')

plt.show()