#imports

import matplotlib.pyplot as l

import pandas as pd



#reading 



data = pd.read_csv("../input/cereal.csv")



data.describe() #print the read file



sugars = data["sugars"]  #reading only the sugar column



l.hist(sugars)      #plot the graph

l.title("Sugar Content")
# Plot a histogram of sodium content with nine bins, a black edge 

# around the columns & at a larger size

l.hist(sugars, bins=10, edgecolor = "black")

l.title("Sugars ") # add a title

l.xlabel("Sugars in milligrams") # label the x axes 

l.ylabel("Count") # label the y axes