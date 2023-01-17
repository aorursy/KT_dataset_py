import numpy as np 

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Imports each CSV file as a dataframe



Disney_df = pd.read_csv("../input/disney-channel-broadcast-history/Disney Data (CSV).csv")

Nick_df = pd.read_csv("../input/nickelodeon-broadcast-history/Nick Data (CSV).csv")

CN_df = pd.read_csv("../input/cartoon-network-broadcast-history/Cartoon Network Data (CSV).csv")



plt.style.use("seaborn-poster")
Disney_df.head() # Shows the first 5 rows of the dataframe
Nick_df.head()
CN_df.head()
All_df = pd.concat([Disney_df, Nick_df, CN_df])   #Using pandas to reshape data here. I used the concatenate function to create a new dataframe that has all three datasets appended together.



All_df
yearLineGraph = pd.DataFrame({"Disney": Disney_df["First Aired"].value_counts(), "Nickelodeon": Nick_df["First Aired"].value_counts()})  #Value_counts creates a series containing the counts of the unique values within my "first aired" column.



# yearLineGraph is a new dataframe that holds all these counts



display(yearLineGraph)



yearLineGraph.to_csv("First Aired Counts.csv")

lineGraph = yearLineGraph.plot(kind = "line") # Plots the new dataframe as a line graph, and sets it equal to a new variable I will use below



lineGraph.set_xlabel("Year")

lineGraph.set_ylabel("Number of Series")

lineGraph.set_title("Series by Year")      # Adds title, x and y axis labels to graph
yearBarGraph1 = pd.DataFrame({"Disney": Disney_df["First Aired"].value_counts(), "Nickelodeon": Nick_df["First Aired"].value_counts()})  # yearBarGraph1 is a new dataframe that holds all the value counts







barGraph1 = yearLineGraph.plot(kind = "bar") # Plots the new dataframe as a bar graph, and sets it equal to a new variable I will use below



barGraph1.set_xlabel("Year")

barGraph1.set_ylabel("Number of Series")

barGraph1.set_title("Series by Year")  # Adds title, x and y axis labels to graph





# Made new dataframes to complete calculations for comparison

disneyCalc = pd.DataFrame({"Disney": Disney_df["First Aired"].value_counts()})

nickCalc = pd.DataFrame({"Nick": Nick_df["First Aired"].value_counts()})



print("Average:\n", disneyCalc.mean(), nickCalc.mean())  # Applied these summary functions to find the mean and max of the columns. Is there a better way to do this? I feel like it looks disorganized when it prints.



print("Max:\n", disneyCalc.max(), nickCalc.max())
#Same method as above, except adding a third bar for Cartoon Network



yearBarGraph2 = pd.DataFrame({"Disney": Disney_df["First Aired"].value_counts(), "Nickelodeon": Nick_df["First Aired"].value_counts(),"Cartoon Network": CN_df["First Aired"].value_counts()})





barGraph2 = yearBarGraph2.plot(kind = "bar")

barGraph2 = yearBarGraph2.plot(kind = "line")  #Adds a line graph with the same information

barGraph2.set_xlabel("Year")

barGraph2.set_ylabel("Number of Series")

barGraph2.set_title("Series by Year")



yearBarGraph2.to_csv("First Aired Counts 2.csv")



# Made new dataframes to complete calculations for comparison

CNCalc = pd.DataFrame({"Cartoon Network": CN_df["First Aired"].value_counts()})





print("Average:\n", CNCalc.mean())  # Applied these summary functions to find the mean and max of the columns.



print("Max:\n", CNCalc.max())
All_df.groupby(['First Aired'])['First Aired'].count().plot(kind='bar')  # This takes all of the values from "first aired", counts them, and plots them into a bar graph. Used the groupby method because it allows the years to be ordered chronologically

All_df["First Aired"].sort_values().value_counts().plot(kind = "bar") #This counts the values within "First Aired", sorts them in descending order, and plots the number of instances in a bar graph
seriesBarGraph = pd.DataFrame({"Disney": Disney_df["Series Type"].value_counts(), "Nickelodeon": Nick_df["Series Type"].value_counts()}) 

display(seriesBarGraph) #Creates new data frame called seriesBarGraph with 'series type' value counts in two separate columns



barGraph0 = seriesBarGraph.plot(kind= "bar")    #Plots bar graphs and creates new variable that I'll need to make labels

barGraph0.set_xlabel("Series Type")             

barGraph0.set_ylabel("Number of Series")     #Creates labels

#Used same method from above, just added Cartoon network to the new dataframe



seriesBarGraph0 = pd.DataFrame({"Disney": Disney_df["Series Type"].value_counts(), "Nickelodeon": Nick_df["Series Type"].value_counts(), "Cartoon Network": CN_df["Series Type"].value_counts()})



display(seriesBarGraph0)

barGraph = seriesBarGraph0.plot(kind= "bar")



barGraph.set_xlabel("Series Type")

barGraph.set_ylabel("Number of Series")
pieDis_df = pd.DataFrame({"Disney": Disney_df["Series Type"].value_counts()})      #Made a new dataframe with the "series type" value counts for disney



colors = ['#264653','#2A9D8F','#E9C46A','#F4A261', '#E76F51', '#e63946', '#f1faee']  #Created color variable with color hexes that will be used in the pie chart function 



plot = pieDis_df.plot.pie(y = "Disney", colors = colors, autopct='%1.1f%%')   #Plots pie chart plot, autopct adds percent

plt.legend(loc="upper left")  #changes location of legend to upper left



plot.set_title("Disney")  #Sets title

plt.tight_layout()       #Auto adjusts chart
#I used the same method as above here, I just made sure the colors were the same for each series type throughout the three pie charts. 

#Is there any way to get the three charts to be side by side?



pieNick_df = pd.DataFrame({"Nickelodeon": Nick_df["Series Type"].value_counts()})



colors = ['#264653','#2A9D8F','#F4A261','#e63946', '#E9C46A', '#E76F51']



plot1 = pieNick_df.plot.pie(y = "Nickelodeon", colors = colors, autopct='%1.1f%%')

plt.legend(loc="upper left")



plot1.set_title("Nickelodeon")

plt.tight_layout()
pieCN_df = pd.DataFrame({"Cartoon Network": CN_df["Series Type"].value_counts()})



colors = ['#264653','#E9C46A','#2A9D8F','#F4A261']



plot2 = pieCN_df.plot.pie(y = "Cartoon Network", colors = colors, autopct='%1.1f%%')

plt.legend(loc="upper left")



plot2.set_title("Cartoon Network")

plt.tight_layout()
All_df["Series Type"].value_counts().plot(kind="bar")   #Used the same bar chart method I had used earlier
disNic_df = pd.concat([Disney_df, Nick_df])    #Merged two dataframes to make new bar chart



disNic_df["Series Type"].value_counts().plot(kind="bar")