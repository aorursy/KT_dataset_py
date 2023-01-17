# This Kernel includes my studies on World Happiness Report 2019



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/world-happiness/2019.csv")        # Reads file and equalize it to data variable

data.info()              # Getting summary on data 
# Columns of data



data.columns
# Renaming columns of data



data.rename(columns={'Overall rank':'Overall_rank', 'Country or region':'Country_or_region', 'GDP per capita':'GDP_per_capita',

                     'Social support':'Social_support','Healthy life expectancy':'Healthy_life_expectancy',

                    'Freedom to make life choices':'Freedom_to_make_life_choices','Perceptions of corruption':'Perceptions_of_corruption'},

           inplace=True)



data.columns
data.corr()          #Shows the correlation in data columns, values around 1.0 means positive correlation where as around -1.0 negative correlation.
#Correlation map : Creating heatmap using correlation output from data.corr()



fig, ax = plt.subplots(figsize=(8,8))       # subplots() function creates a single figure object as "fig" and a single axis object as "ax"

                                            # figsize() argument defines the size of the figure object

    

sns.heatmap(data.corr(), annot=True,linewidth=0.5, linecolor="blue", fmt=".2f", ax=ax)

plt.show()
data.head(10)       #Shows first 10 rows of data

                    #data.head() shows first 5 rows of data
data.tail(10)         #Shows last 10 rows of data
# Multiple Line plot using <figure> and <ax> object



fig, ax = plt.subplots(figsize=(10,10))    #Creating figure and axis object in order to plot multiple lines in one plot

x=data["Score"]                            #Setting data for x axis



#Setting data for y axis



y0=data["Overall_rank"]

y1=data["Country_or_region"]

y2=data["GDP_per_capita"]

y3=data["Social_support"]

y4=data["Healthy_life_expectancy"]

y5=data["Freedom_to_make_life_choices"]

y6=data["Generosity"]

y7=data["Perceptions_of_corruption"]



#Plotting lines of selected y datas vs x data in one graph



ax.plot(x,y2,label="GDP_per_capita", color="red", linewidth=2, linestyle="-.", alpha=0.9 )   # Line plotting using plot() attribute of ax object

ax.plot(x,y3,label="Social_support", color="black", linewidth=1)

ax.plot(x,y4,label="Healthy_life_expectancy", color="purple", linewidth=1.5)

ax.plot(x,y5,label="Freedom_to_make_life_choices", color="blue")

ax.plot(x,y7,label="Perceptions_of_corruption", color="green", linewidth=2, linestyle="-.")



plt.title("Effects on Happiness of 2019")       # Shows title of the plot

plt.xlabel("Happiness Score")                   # Shows X-Axis definition

plt.ylabel("Values")                            # Shows Y-Axis definition

plt.legend(loc="upper left")                    # Shows legend at location of upper left on plot



plt.show()
# Multiple line plot specifying target of <ax> in <DataFrame.plot>



ax1= data.plot(kind="line", x="Score", y="GDP_per_capita", color="red", linewidth=2, linestyle="-.", alpha=0.9 )

ax2= data.plot(kind="line", x="Score", y="Social_support", color="black", linewidth=1, ax=ax1)

data.plot(kind="line", x="Score", y="Healthy_life_expectancy", color="purple", linewidth=1.5, ax=ax2)



plt.show()
# Scatter plot (Plotting scatters seperately in different plots)



data.plot(kind="scatter", x="Healthy_life_expectancy", y="Freedom_to_make_life_choices", color="blue")  

data.plot(kind="scatter", x="Healthy_life_expectancy", y="Social_support", color="red")

data.plot(kind="scatter", x="Healthy_life_expectancy", y="GDP_per_capita", color="green")

plt.show()
# Multiple scatter plotting --> by specifying target of <ax> in <DataFrame.plot>



# Defining <ax> object as scatter plot of "Healthy life expectancy" vs "Freedom to make life choices".

ax = data.plot(kind="scatter", x="Healthy_life_expectancy", y="Freedom_to_make_life_choices", color="blue", label="Freedom_to_make_life_choices")



# Defining <ax1> object as scatter plot of "Healthy life expectancy" vs "Social support"

# Specifying target of ax to ax

ax1= data.plot(kind="scatter", x="Healthy_life_expectancy", y="Social_support", color="red", ax=ax, label="Social_support")



#Specifying target of ax to ax1

data.plot(kind="scatter", x="Healthy_life_expectancy", y="GDP_per_capita", color="green", ax=ax1,grid=True, label="GDP_per_capita")



plt.legend(loc="upper left")    # Showing and locating of legend

plt.title("Scatter Plot")       # Showing title

ax.set_ylabel("Values")         # Defining and showing y-axis by using <ax> object.

plt.show()

# Histogram of all columns in <data>



data.plot(kind="hist", bins=50, figsize=(15,10), edgecolor="black", grid=True)

plt.xlabel("Values")

plt.ylabel("Frequency of countries")

plt.show()
# Overall_rank column is dropped and defined new dataframe as <new_data>

new_data=data.drop(columns=["Overall_rank"])    # Creating <new_data> by dropping two columns from <data>

new_data.plot(kind="hist", bins=250, figsize=(20,5), edgecolor="black", grid=True)    # Plotting histogram

plt.xlabel("Values")

plt.ylabel("Frequency of Countries")

plt.legend(loc="upper center")

plt.show()

           
# Histogram



data.GDP_per_capita.plot(kind="hist", bins=20, figsize=(5,5), color="green", edgecolor="black")

plt.title("Histogram of GDP")

plt.legend(loc="upper left")

plt.show()

# Histogram Happiness Score

data.Score.plot(kind="hist", bins=20, figsize=(5,5), color="purple", edgecolor="black")               #Plotting histogram of Hapiness Score.



plt.title("Histogram of Score")

plt.legend(loc="upper left")

plt.show()
