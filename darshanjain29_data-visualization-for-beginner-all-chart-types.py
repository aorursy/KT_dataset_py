

import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Set up code checking

import os

if not os.path.exists("../input/fifa.csv"):

    os.symlink("../input/data-for-datavis/fifa.csv", "../input/fifa.csv")  

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex1 import *

print("Setup Complete")
# Path of the file to read

fifa_filepath = "../input/fifa.csv"



# Read the file into a variable fifa_data

fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)

print (fifa_data.info())
# Set the width and height of the figure

plt.figure(figsize=(16,10))

plt.xlabel("Years", fontsize=18)

plt.ylabel("Rank in the Tournament")



#Either do below steps or

#plt.plot(fifa_data)

#plt.show()



#Or use lineplot

# Line chart showing how FIFA rankings evolved over time

sns.lineplot(data=fifa_data)
# Path of the file to read

spotify_filepath = "../input/spotify.csv"



# Read the file into a variable spotify_data

spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)



spotify_data.head(10)
spotify_data.tail(10)
#Basic view

sns.lineplot(data = spotify_data)
#Now let's see the map in an enlarged view

plt.figure(figsize=(14,6))

plt.title("Daily Global Streams of Popular Songs in 2017-2018")

plt.xlabel("Dates")

plt.ylabel("No. of streams")

sns.lineplot(data = spotify_data)
#How to view subset of a dataset on line chart

plt.title("Subset view of the entire dataset")

sns.lineplot(data = spotify_data["Shape of You"], label = "Shape of You")

sns.lineplot(data = spotify_data["Despacito"], label="Despacito")
flight_filepath = "../input/flight_delays.csv"



flight_data = pd.read_csv(flight_filepath, index_col = "Month")
#Viewind the data of the file

flight_data
plt.figure(figsize = (15, 6))

plt.title("Monthwise average arrival delay of the flight")



sns.barplot(x = flight_data.index, y = flight_data["NK"])



# Add label for vertical axis

plt.ylabel("Arrival delay (in minutes)")
plt.figure(figsize = (14, 7))

plt.title("Average arrival delay fo each airline, monthwise")



sns.heatmap(data = flight_data, annot = True)



plt.xlabel("Airlines")
insurance_filepath = "../input/insurance.csv"



insurance_data = pd.read_csv(insurance_filepath)
insurance_data.head(10)
sns.scatterplot(x = insurance_data["bmi"], y= insurance_data["charges"])
#Adding a regression line on the above scatter plot

sns.regplot(x = insurance_data["bmi"], y= insurance_data["charges"])
sns.scatterplot(x=insurance_data["bmi"], y = insurance_data["charges"], hue = insurance_data["smoker"])
#Adding a regression line on the above scatter plot

sns.lmplot(x="bmi", y = "charges", hue = "smoker", data = insurance_data)
#use this to extend the scatter plots for the categorial data like gender= male/female, smoker = yes/no

sns.swarmplot(x = insurance_data["smoker"], y = insurance_data["charges"])
iris_filepath = "../input/iris.csv"



# Read the file into a variable iris_data

iris_data = pd.read_csv(iris_filepath, index_col="Id")

iris_data.head()
sns.distplot(a = iris_data["Petal Length (cm)"], kde =  False) 
sns.kdeplot(data = iris_data["Petal Length (cm)"], shade = True)
sns.jointplot(x = iris_data["Petal Length (cm)"], y = iris_data["Petal Width (cm)"], kind = "kde")
# Paths of the files to read

iris_set_filepath = "../input/iris_setosa.csv"

iris_ver_filepath = "../input/iris_versicolor.csv"

iris_vir_filepath = "../input/iris_virginica.csv"



# Read the files into variables 

iris_set_data = pd.read_csv(iris_set_filepath, index_col="Id")

iris_ver_data = pd.read_csv(iris_ver_filepath, index_col="Id")

iris_vir_data = pd.read_csv(iris_vir_filepath, index_col="Id")



# Print the first 5 rows of the Iris versicolor data

iris_ver_data.head()
sns.distplot(a = iris_set_data["Petal Width (cm)"], label = "Iris-setsova", kde= False)

sns.distplot(a = iris_ver_data["Petal Width (cm)"], label = "Iris-versicolor", kde= False)

sns.distplot(a = iris_vir_data["Petal Width (cm)"], label = "Iris-virginica", kde = False)



plt.legend()
#KDE plots for the above species

sns.set_style("whitegrid")

#sns.kdeplot(data = iris_data["Petal Length (cm)"], shade = True)

sns.kdeplot(data = iris_set_data["Petal Width (cm)"], label = 'Iris-setsova', shade = True)

sns.kdeplot(data = iris_ver_data["Petal Width (cm)"], label = 'Iris-versicolor', shade = True)

sns.kdeplot(data = iris_vir_data["Petal Width (cm)"], label = 'Iris-virginica', shade = True)
# sns.lineplot



sns.set_style("whitegrid")



# Line chart 

plt.figure(figsize=(12,6))

sns.lineplot(data=spotify_data)