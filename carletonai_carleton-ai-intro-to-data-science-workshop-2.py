# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# Get the Data Dictonary here

# https://open.canada.ca/data/en/dataset/1eb9eba7-71d1-4b30-9fb1-30cbdab7e63a

import pandas as pd #Data processing and data structures

import matplotlib.pyplot as plt #Plotting

import numpy as np #Linear Algerbra library



df = pd.read_csv('/kaggle/input/ncdb-2014/NCDB_2014.csv', na_values=["UUUU", "UU", "U", "XXXX", "XX", "X", "QQQQ", "QQ", "Q"])#Piece by Piece, after showing

#reads the csv file into a Pandas DataFrame

#na_values changes these values to "not available"
df = df.dropna() #discards the values we defined as not available

df = df.reset_index(drop=True) #Drop avoids creating new column for the index

df = df.astype({"C_HOUR": int, "C_RALN": int, "C_RSUR": int}) #retype float-valued columns to int. Convert it to categorical variables
plt.boxplot(df["C_VEHS"], vert=False) #Horizontal Boxplot

plt.title("Number of Vehicles Involved in Collision") #Adds a title

#Collisions of only 2 cars are so common, the entire box of our boxplot is hidden within this yellow line!
slices = [0, 0, 0] #define an array to hold the proportions of the pie



isev_counts = df["P_ISEV"].value_counts() #creates an array of the tallies of each value in the column "P_ISEV"



#define slice sizes

slices[0] = isev_counts[1] + isev_counts[2] #Non-Fatal

slices[1] = isev_counts[0] #Injury

slices[2] = isev_counts[3] #Fatality



plt.pie(slices, #slice proportions

        explode=[0,0,.25], #Bump out the "Fatality" category

        labels=["No Injury", "Injury", "Fatality"], #Labels

        autopct="%1.1f%%" #Percent formatting

        )

plt.title("Consequences for Persons Involved") #Adds a title


plt.scatter(df["C_HOUR"], df["C_VEHS"]) # Scatter plot

plt.title("Number of Vehicles Involved in Collision Vs. Time of day of Collision") #Adds a title

plt.xlabel("Time of Collision (Hrs. since 12am)") #X-axis Label

plt.ylabel("Number of Vehicles Involved") #Y-axis Label

plt.plot(np.unique(df["C_HOUR"]), np.poly1d(np.polyfit(df["C_HOUR"], df["C_VEHS"], 1))(np.unique(df["C_HOUR"])), color="red")

#Line of best fit^^
collisionsPerHour = df["C_HOUR"].value_counts() #total for every hour

heights = df[df["C_SEV"] == 1]["C_HOUR"].value_counts() #total fatalities for every hour



heights /= collisionsPerHour #totals for every hour each divided by the total fatalities occuring at that hour.



plt.bar(heights.index, heights) #Plots the x of hour, with the y of our custom metric total fatalities per hour.



plt.title("Percent of Collisions Resulting in Death Vs. Time of day of Collision") #Adds a title

plt.xlabel("Time of Collision (Hrs. since 12am)") #X-Axis Label

plt.ylabel("Percent of Collisions Resulting in Death") #Y-Axis Label 
weather_counts = df["C_WTHR"].value_counts() #Tallies of each value in C_WTHR



plt.pie(weather_counts, #Slice Proportions

        explode=[0,0,0,0, .75, 0, 0.75], #Label separation

        labels=["Clear and Sunny", "Overcast", "Rain", "Snow", "Freezing Rain/Sleet/Hail", "Visibility Limitation", "Strong Wind"], #Label Text

        autopct="%1.1f%%" #Formatted percentages

        )

plt.title("Weather Conditions of Collision") #Adds title
heights = df[df["C_SEV"] == 2]["C_RSUR"].value_counts() #Tallies of each value of C_RSUR where C_SEV is 2



plt.bar(heights.index, # value

        heights, # occurences of value

        tick_label=["Dry", "Wet", "Snow", "Slush", 

                     "Icy", "Sand\nGravel\nDirt", "Muddy", "Oil", "Flooded"]) # Label text



plt.title("Collisions Resulting in Injury Vs. Road Surface of Collision") # Add a title

plt.xlabel("Raod Surface") # X-Axis label

plt.ylabel("Injuries") # Y-Axis label
heights = df[df["C_SEV"] == 2]["C_RALN"].value_counts() #Tallies of each value of C_RALN where C_SEV is 2





plt.bar(heights.index, # value

        heights,# occurences of value

        tick_label=["Striaght\nNo\nGradient", "Straight\nw/\nGradient",

                    "Curved\nNo\nGradient", "Curved\nw/\nGradient", 

                     "Hilltop", "Bottom\nof\nHill"])# Label text



plt.title("Collisions Resulting in Injury Vs. Road Alignment") #Adds title

plt.xlabel("Raod Alignment") #X-Axis Label

plt.ylabel("Injuries") #Y-Axis label

safety = df[df["C_SEV"] == 1]["P_SAFE"].value_counts() # Tallies of each value of P_SAFE where C_SEV is 1



proper = safety[0] # the first value of the safety array

unproper = safety.sum() - proper #the sum of the rest of the safety options, minus the proper option 



plt.pie([unproper, proper], #Array of slice sizes

        labels=["No Safety Device Used", "Safety Device Used"], #Label text

        autopct="%1.1f%%" #Percent Formatting

        )

plt.title("Safety Devices Used Vs. Not Used\n(Comparing Number of Fatalities)") # Add title
