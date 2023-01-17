# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#First I assign the dataframe variable to the Pokedex cvs filed that I uploaded.

df = pd.read_csv("../input/pokedex.csv")



#To make sure that the file imported correctly, I use the .head function to see the first few rows of data

df.head(10)

#that seems to work pretty well, so I can move on to the next step.
#For this assignment I'm interested in looking at ghost pokemon and fairy pokemon, because they are some of my favorite aesthetically. To look at only the name, type, and attack, I'll grab those 3 columns that I'm interested in looking at.

#I'll use the 3 column dataframe variable to call only those 3 from the dataframe. 

three_col_df = df[["name__english", "type__001", "base__Attack"]]



#then I call the 3 column df variable just to make sure the action worked. 

three_col_df



#after making sure it works, I assign this view the name "type attack" so that I know this function will only show the 3 columns I'm working with. 

type_attack = three_col_df

type_attack
#In order to only look at a specific entry in the columns (in this case fairy type), I have to use a boolean function where the only Pokemon type that will display is the fairy type. If it is not a fairy type, it will not display. 

fairy = type_attack[df['type__001'] == 'Fairy']

#I print fairy just to make sure the above function worked.

fairy
#Now I want to get the same information from the ghost type Pokemon as I did with the fairy type. To do that, I will make a new variable "ghost". 

#In order to only look at a specific entry in the columns (in this case ghost type), I have to use a boolean function where the only Pokemon type that will display is the ghost type. If it is not a ghost type, it will not display. 

ghost = type_attack[df['type__001'] == 'Ghost']



#then I test it to make sure the function is working properly. 

ghost

#Now that I have the narrowed data I want from each type of pokemon, I will try to compare the two using visualization methods. 

#I will attempt a bar graph. 

#Here I will compare the mean attack stat of both Fairy and Ghost type pokemon. 



#Here I define a new variable to hold the mean of the Fairy type. 

fairy_attack_mean = fairy["base__Attack"].mean()



#Here I define a new variable to hold the mean of the Ghost type. 

ghost_attack_mean = ghost["base__Attack"].mean()



print("The Fairy type mean is ", fairy_attack_mean, "and the Ghost type mean is ", ghost_attack_mean)
#Now I want to visualize the various fairy pokemons' attack values in a bar graph. I will define 3 different strength categories: weak, pretty good, and awesome. 

#In order to do that I use the define and rank performance function and create some booleans to categories the stats.

fairy["base__Attack"]



def rank_performance(base__Attack):

    if base__Attack <= 30:

        return "Weak"

    elif base__Attack >41 and base__Attack <90:

        return "Pretty Good"

    else:

        return "Awesome"



#Then I apply the rank function to see how the stats are ranked. 

fairy["base__Attack"].apply(rank_performance)   

    
#To print the bar graph, I use the same rank function as before, but I also include the plot function to put in on a graph. 

#Since horizontal graphs are better for visualizing ranking, I use the barh style.

fairy["base__Attack"].apply(rank_performance).value_counts().plot(kind="barh")
#Now I want to see how the ghost types compare to the fairy types, so I basically to the same thing as before. 

#I will define 3 different strength categories: weak, pretty good, and awesome. 

#In order to do that I use the define and rank performance function and create some booleans to categories the stats.

ghost["base__Attack"]



def rank_performance(base__Attack):

    if base__Attack <= 30:

        return "Weak"

    elif base__Attack >41 and base__Attack <90:

        return "Pretty Good"

    else:

        return "Awesome"

    

#I apply the rank function to see how the stats are ranked. 

ghost["base__Attack"].apply(rank_performance)  
#To print the bar graph, I use the same rank function as before, and I also include the plot function to put it on a graph. 

#Since horizontal graphs are better for visualizing ranking, I use the barh style.

ghost["base__Attack"].apply(rank_performance).value_counts().plot(kind="barh")
#Next, I want to combine both outputs and out them on a bar graph together, to see how they stack up against each other. 

#First I give both datasets new variables, which include the base attack column for each. 

fairy["base__Attack"] = "fairy_BA"

ghost["base__Attack"] = "ghost_BA"



#Here I'm trying to set up the side by side bar graph by calling both keys. 

DF = pd.concat([fairy,ghost],keys=['fairy','ghost'])

DFGroup = DF.groupby(["type__001","base__Attack"])

DFGroup.sum().unstack('base__Attack').plot(kind='barh')



#I'm not entirely sure what's going on here. The error says there's no numeric data to enter but "base__Attack" is numeric data. Not sure which input is incorrect. 
#Now i wanted to see how HP and attack stats are related by using a scatter plot. 

#I'm using the entire data frame, so I use the df variable, i'm plotting the attack on the x axis and HP on y axis. 

df[["base__Attack", "base__HP"]].plot(kind="scatter", x="base__Attack", y="base__HP")



#It seems like there is a slight positive correlation but I want to try to make the scatter plot larger.

#I wasn't able to figure out how to make the size of the graph bigger, but I figured out how to size down the area of the single dots so that it's clearrer to see each plotted point.

#I also changed the color of the points to black to make them easier to see. 



df[["base__Attack", "base__HP"]].plot(kind="scatter", x="base__Attack", y="base__HP", color ="black", s=1)

# There seems to be a positive correlation between attack and HP, minus a few outliers as seen on the scatter plot. 