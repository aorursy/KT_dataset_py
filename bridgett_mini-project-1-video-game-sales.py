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
#Importing this first for plotting data later. 

import matplotlib.pyplot as plt

import datetime as dt

import matplotlib.dates as mdates

import altair as alt





#I'm going to start dissecting this data. But before doing that, I have to give the dataset/csv file I added a dedicated variable name. 

#Let's keep it simple and call it df.



df = pd.read_csv("../input/VideoGameSales1.csv")



#To make sure the file imported correctly, I'm using the .head(10) function to see the first few (ten) rows of data.



df.head(10)

#It seems to be showing up pretty well. Now we can move on. 
#For this assignment, I'm interested in looking at name, genre, North America Sales, year of release, and user score.

#To look at only those criteria, I'll use a five column function to call only those five items that I'm interested in looking at.



five_col_df = df[["Name", "Year_of_Release", "Genre", "NA_Sales", "Critic_Score"]]



#Then I call the 5 column df variable just to make sure the action worked. 



five_col_df



#after making sure it works, I assign this view the name "df5" so that I know this function will only show the 5 columns I'm working with. Then I test it again, you can never be too sure.



df5 = five_col_df

df5

#Now that I narrowed down the data frame, I want to try to look specifically at how a specific genre has performed over time. I'm personally interested in seeing how role-playing game sales have changed.

#To do that, I'm going to narrow it even further by singling out "role playing" in the genre column.



rpg_data = df5[df['Genre'] == 'Role-Playing']



#Now I print the new rpg variable to make sure it worked. Looks good so far. 

rpg_data
#Now I want to just plot the data points on a scatter plot, and see what's going on with the sales per year for role playing games.

#Here, I'm defining what the x and y axis plots should be, in this case the sales are the y axis and year of release is the x axis. 

x = rpg_data[["Year_of_Release"]]

y = rpg_data[["NA_Sales"]]



#I also want the plots to be somewhat small and visible, so I'm changing the color to black and size to 2 points. 

plt.scatter(x,y, alpha=0.5, c="black", s=2)



#From the scatter plot, I can see that RPG sales have pretty much been consistent, with a few outliers towards the top. I am assuming that the plots at the top are the best selling RPG games of that year. I can attribute those sales to very popular games coming out those years, such as Final Fantasy or Pokemon. 

#To better visualize this data, I'm going to try to see this in a line graph. 
#Now I'm going to try to plot this data in a line graph to see if that visualization would better suit this data. 



rpg_data.plot(y="NA_Sales", x="Year_of_Release")



#It seems like the line graph is all over the place in terms of sales over the years, from 2000-2016. I can probably attribute this trend to best selling RPGs being released consistently over the years.

#From the graph, I can see that there is a bit of a slump between the years 2006-2009. This is cool to see because there is something known in the game community as the "japanese slump" where Japanese game developers weren't producing high selling game during those years. Japan is also known for being a huge producer of role playing games. During the time of this slump, this style of game (JRPGs) stagnated with the arrival of the Xbox 360 and PS3. Sorry for the rambling!



#I'm going to look at a different video game genre and see if it follows the same trend, hopefully I'll get a different result.

#I'm curious to see how it will compare to RPG, I'll be looking at fighting games, which are one of my favorite genres. 
#To look specfically at fighting games, I'm going to pretty much follow the same steps I did for the RPGs. 

#First I'm defining a new variable to assign the fighting game sales to. 

#THen I'm using a function to indicate if the genre is not "fighting", don't show it. 



fighting_data = df5[df['Genre'] == 'Fighting']





#Now I print the variable only to make sure it worked. 

fighting_data
#Now I want to just plot the data points on a scatter plot, and see what's going on with the sales per year for fighting games.

#Here, I'm defining what the x and y axis plots should be, in this case the sales are the y axis and year of release is the x axis. 

x = fighting_data[["Year_of_Release"]]

y = fighting_data[["NA_Sales"]]



#I also want the plots to be somewhat small and visible, so again I'm changing the color to black and size to 2 points. 

plt.scatter(x,y, alpha=0.5, c="black", s=2)



#It seems like this scatter plot and the scatter plot for RPGs are following a similar trend. Let's see what it looks like in a line graph. 
#Let's try out the line graph for the fighting game sales. I'm hoping to see something different than the previous crazy graph.

fighting_data.plot(y="NA_Sales", x="Year_of_Release", color="orange")





#SO cool! So it seems like there is more of a pronounced change in this graph. It seems like there was a really popular game that sold well in 2001 (Super Smash Bros. Melee), 2008 (Street Fighter 4, one of my favs), and 2014 (Ultra Street Fighter 4), as indicated by the sharp spikes. 

#Unfortunately though, the spike seems more pronounced because only the year is being reported so there is a drastic drop between every year, which is skewing the visuallization.
import numpy as np

import matplotlib.pyplot as plt



#Now I want to try to see what this would look like in a stacked plot. It's supposed to look like different line graphs stacked on one another. This would help me compare how the two genres have performed in sales against each other. 

#I'm borrowing this code below from matplotlib (https://matplotlib.org/gallery/lines_bars_and_markers/stackplot_demo.html).

#First I'm identifying the dataframe source, which is df5 from the beginning.



source = df5



#Here I'm identifying the x axis and the 3 y-axes I want to look at, which are fighting, action, and role playing games.

x = ["NA_Sales"]



#The original code had numbers listed in the brackets, but I want to use the specific values for those genres. I think this is where the code is going wrong, I'm not sure how to pull this data effectively.

y1 = [df5['Genre'] == 'Fighting'] 

y2 = [df5['Genre'] == 'Action'] 

y3 = [df5['Genre'] == 'Role-Playing'] 



y = np.vstack([y1, y2, y3])



labels = ["Fighting ", "Action", "Role-Playing"]



fig, ax = plt.subplots()

ax.stackplot(x, y1, y2, y3, labels=labels)

ax.legend(loc='upper left')

plt.show()



#It seems like the graph is set up but the data is missing. This is kinda frustrating because I've tried it several different ways but can't figure out how to get the right on there. I think I'm identifying the wrong y axes. 
import numpy as np

import matplotlib.pyplot as plt



#Unfortunately, the first attempt didn't go as I wanted but I'm ready to build on that and try again.

#This is my second attempt at the stacked plot. This time I'm using different y axes and narrowing it to 2 only.

source = df5



#Instead of using the NA sales column, I'm trying the year of release column since that may be a better x axis. 

x = ["Year_of_Release"]



#In this attempt, I'm also using different/separate dataframes to pull out the data. 

y1 = [fighting_data["NA_Sales"]] 

y2 = [rpg_data["NA_Sales"]] 



y = np.vstack([y1, y2])



labels = ["Fighting ", "Role-Playing"]



fig, ax = plt.subplots()

ax.stackplot(x, y1, y2, labels=labels)

ax.legend(loc='upper left')

plt.show()



#Didn't work :=(

#In the error code, it tells me that the "axis must match exactly", but one df is 640 rows while the other is 1300.

#This is frustrating also, I think this version may be closer to getting it to work, but for some reason the xes have to be the same value. 
import numpy as np

import matplotlib.pyplot as plt



#Okay, last attempt. I copy and pasted the first iteration of the code. I'm going to try one more time before moving on to the next thing. 



source = df5



#In the first try, I didn't put df5 infront of the brackets, but I'm putting in here to see if maybe that was a mising component of the first try.

x = df5["NA_Sales"]



y1 = [df5['Genre'] == 'Fighting'] 

y2 = [df5['Genre'] == 'Action'] 

y3 = [df5['Genre'] == 'Role-Playing'] 



y = np.vstack([y1, y2, y3])



labels = ["Fighting ", "Action", "Role-Playing"]



fig, ax = plt.subplots()

ax.stackplot(x, y1, y2, y3, labels=labels)

ax.legend(loc='upper right')

plt.show()



#I got something! It looks like a hot mess but at least something is showing up on the graph. Pretty much the only thing I did this time vs the first time was add df5 in front of the brackets in the x= .

#In terms of how the data is showing, I'm not sure if it looks that way because the data is that way or if it's because I formatted the y= stacks incorrectly. 
#It didn't go too well with visualizing sales data.

#But moving on, I'm curious to see how the ratings data is going to look.



#First I'm bringing back the original dataframe with all the columns from the cvs files. 

df



#Looks like that works.
#I'm going to start with a line graph. I'm going to be looking at ratings for the action game genre. 

#First, I'm creating a new variable from the df5 five column df from the beginning of this project, since I haven't looked at action game data yet.

action_data = df5[df['Genre'] == 'Action']



#Now I'm testing that it works. 

action_data.head(10)

#Here I'm plotting the critic score of action games over time. 

action_data.plot(y="Critic_Score", x="Year_of_Release", color="green")





#Seems like this line graph isn't working too well. 

#I;m going to try a different kind of graph. 
#To better visualize the ratings for action games, I'm going to use a bar graph visualization. 

#Here I'm defining high, average, and low ratings for the games. 

#In order to do that I use the define and rank performance function and create some booleans to categories the stats.



action_data["Critic_Score"]



def rank_performance(Critic_Score):

   

    if Critic_Score <= 40:

        return "Low Rating"

    elif Critic_Score >40 and Critic_Score <=79:

        return "Average Rating"

    elif Critic_Score >79:

        return "High Rating"

    

#Then I apply the rank function to see how the stats are ranked. 

action_data["Critic_Score"].apply(rank_performance)   
#To print the bar graph, I use the same rank function as before, but I also include the plot function to put in on a graph. 

#Since horizontal graphs are better for visualizing ranking, I use the barh style.

action_data["Critic_Score"].apply(rank_performance).value_counts().plot(kind="barh", color="red")



#From the data it seems like most games are rated between 50 and 89 out of 100, an average rating. 

#I'm curious to see how that compared to other genres. 
#To better visualize the ratings for fighting games, I'm going to again use a bar graph visualization. 

#Here I'm defining high, average, and low ratings for the games, with the same values as before. 

#In order to do that I use the define and rank performance function and create some booleans to categories the stats.



fighting_data["Critic_Score"]



def rank_performance(Critic_Score):

   

    if Critic_Score <= 40:

        return "Low Rating"

    elif Critic_Score >40 and Critic_Score <=79:

        return "Average Rating"

    elif Critic_Score >79:

        return "High Rating"

    

#Then I apply the rank function to see how the stats are ranked. 

fighting_data["Critic_Score"].apply(rank_performance)   
#Here I'm printing the bar graph again.

fighting_data["Critic_Score"].apply(rank_performance).value_counts().plot(kind="barh", color="purple")





#From the visualization, it looks liket the ratings for fighting games are pretty similar to the action game ratings. I guess this shows that critic ratings don't really mean too much as they are just arbitrary values. 