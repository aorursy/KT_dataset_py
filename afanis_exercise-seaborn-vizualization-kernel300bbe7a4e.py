# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Path of the file to read



fifa_filepath = "../input/data-for-datavis/fifa.csv"





# Read the file into a variable fifa_data

fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)

# Set the width and height of the figure

plt.figure(figsize=(16,6))



# Line chart showing how FIFA rankings evolved over time

sns.lineplot(data=fifa_data)
# Path of the file to read

spotify_filepath = "../input/data-for-datavis/spotify.csv"



# Read the file into a variable spotify_data

spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)

# Print the first 5 rows of the data

spotify_data.head()
# Print the last five rows of the data

spotify_data.tail()
# Line chart showing daily global streams of each song 

sns.lineplot(data=spotify_data)
# Set the width and height of the figure

plt.figure(figsize=(14,6))



# Add title

plt.title("Daily Global Streams of Popular Songs in 2017-2018")



# Line chart showing daily global streams of each song 

sns.lineplot(data=spotify_data)
# Set the width and height of the figure

plt.figure(figsize=(14,6))



# Add title

plt.title("Daily Global Streams of Popular Songs in 2017-2018")



# Line chart showing daily global streams of 'Shape of You'

sns.lineplot(data=spotify_data['Shape of You'], label="Shape of You")



# Line chart showing daily global streams of 'Despacito'

sns.lineplot(data=spotify_data['Despacito'], label="Despacito")



# Add label for horizontal axis

plt.xlabel("Date")
# Path of the file to read

museum_filepath = "../input/data-for-datavis/museum_visitors.csv"



# Fill in the line below to read the file into a variable museum_data

museum_data = pd.read_csv(museum_filepath, index_col="Date", parse_dates=True)
# Print the last five rows of the data 

museum_data.tail()
# Line chart showing the number of visitors to each museum over time

# Set the width and height of the figure

plt.figure(figsize=(12,6))

# Line chart showing the number of visitors to each museum over time

sns.lineplot(data=museum_data)

# Add title

plt.title("Monthly Visitors to Los Angeles City Museums")
# Set the width and height of the figure

plt.figure(figsize=(12,6))

# Add title

plt.title("Monthly Visitors to Avila Adobe")

# Line chart showing the number of visitors to Avila Adobe over time

sns.lineplot(data=museum_data['Avila Adobe'])

# Add label for horizontal axis

plt.xlabel("Date")
# Path of the file to read

flight_filepath = "../input/data-for-datavis/flight_delays.csv"



# Read the file into a variable flight_data

flight_data = pd.read_csv(flight_filepath, index_col="Month")
# Print the data

flight_data



# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=flight_data.index, y=flight_data['NK'])



# Add label for vertical axis

plt.ylabel("Arrival delay (in minutes)")
# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=flight_data.index, y=flight_data['NK'])
# Set the width and height of the figure

plt.figure(figsize=(14,7))



# Add title

plt.title("Average Arrival Delay for Each Airline, by Month")



# Heatmap showing average arrival delay for each airline by month

sns.heatmap(data=flight_data, annot=True)



# Add label for horizontal axis

plt.xlabel("Airline")
#The relevant code to create the heatmap is as follows:



# Heatmap showing average arrival delay for each airline by month

sns.heatmap(data=flight_data, annot=True)
# Path of the file to read

ign_filepath = "../input/data-for-datavis/ign_scores.csv"



# Fill in the line below to read the file into a variable ign_data

ign_data = pd.read_csv(ign_filepath, index_col="Platform")

# Print the data

ign_data
# Fill in the line below: What is the highest average score received by PC games,

# for any platform?

high_score = 7.759930



# On the Playstation Vita platform, which genre has the 

# lowest average score? Please provide the name of the column, and put your answer 

# in single quotes (e.g., 'Action', 'Adventure', 'Fighting', etc.)

worst_genre = 'Simulation'
# Bar chart showing average score for racing games by platform



# Set the width and height of the figure

plt.figure(figsize=(8, 6))

# Bar chart showing average score for racing games by platform

sns.barplot(x=ign_data['Racing'], y=ign_data.index)

# Add label for horizontal axis

plt.xlabel("")

# Add label for vertical axis

plt.title("Average Score for Racing Games, by Platform")

# Set the width and height of the figure

plt.figure(figsize=(10,10))

# Heatmap showing average game score by platform and genre

sns.heatmap(ign_data, annot=True)

# Add label for horizontal axis

plt.xlabel("Genre")

# Add label for vertical axis

plt.title("Average Game Score, by Platform and Genre")

# Path of the file to read

insurance_filepath = "../input/data-for-datavis/insurance.csv"



# Read the file into a variable insurance_data

insurance_data = pd.read_csv(insurance_filepath)
insurance_data.head()
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])
sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])
sns.swarmplot(x=insurance_data['smoker'],

              y=insurance_data['charges'])

# Path of the file to read

candy_filepath = "../input/data-for-datavis/candy.csv"



# Fill in the line below to read the file into a variable candy_data

candy_data = pd.read_csv(candy_filepath, index_col="id")
# Print the first five rows of the data

candy_data.head() 
# Fill in the line below: Which candy was more popular with survey respondents:

# '3 Musketeers' or 'Almond Joy'?  (Please enclose your answer in single quotes.)

more_popular = '3 Musketeers'

# Fill in the line below: Which candy has higher sugar content: 'Air Heads'

# or 'Baby Ruth'? (Please enclose your answer in single quotes.)

more_sugar = 'Air Heads'
# Scatter plot showing the relationship between 'sugarpercent' and 'winpercent'

sns.scatterplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])
# Scatter plot w/ regression line showing the relationship between 'sugarpercent' and 'winpercent'

sns.regplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])

# Color-coded scatter plot w/ regression lines

sns.lmplot(x="pricepercent", y="winpercent", hue="chocolate", data=candy_data)

# Scatter plot showing the relationship between 'chocolate' and 'winpercent'

sns.swarmplot(x=candy_data['chocolate'], y=candy_data['winpercent'])

# Path of the file to read

iris_filepath = "../input/data-for-datavis/iris.csv"



# Read the file into a variable iris_data

iris_data = pd.read_csv(iris_filepath, index_col="Id")



# Print the first 5 rows of the data

iris_data.head()
# Histogram 

sns.distplot(a=iris_data['Petal Length (cm)'], kde=False)
# KDE plot 

sns.kdeplot(data=iris_data['Petal Length (cm)'], shade=True)
# 2D KDE plot

sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind="kde")
# Paths of the files to read

iris_set_filepath = "../input/data-for-datavis/iris_setosa.csv"

iris_ver_filepath = "../input/data-for-datavis/iris_versicolor.csv"

iris_vir_filepath = "../input/data-for-datavis/iris_virginica.csv"



# Read the files into variables 

iris_set_data = pd.read_csv(iris_set_filepath, index_col="Id")

iris_ver_data = pd.read_csv(iris_ver_filepath, index_col="Id")

iris_vir_data = pd.read_csv(iris_vir_filepath, index_col="Id")



# Print the first 5 rows of the Iris versicolor data

iris_ver_data.head()
# Histograms for each species

sns.distplot(a=iris_set_data['Petal Length (cm)'], label="Iris-setosa", kde=False)

sns.distplot(a=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", kde=False)

sns.distplot(a=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", kde=False)



# Add title

plt.title("Histogram of Petal Lengths, by Species")



# Force legend to appear

plt.legend()
# KDE plots for each species

sns.kdeplot(data=iris_set_data['Petal Length (cm)'], label="Iris-setosa", shade=True)

sns.kdeplot(data=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", shade=True)

sns.kdeplot(data=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", shade=True)



# Add title

plt.title("Distribution of Petal Lengths, by Species")
# Paths of the files to read

cancer_b_filepath = "../input/data-for-datavis/cancer_b.csv"

cancer_m_filepath = "../input/data-for-datavis/cancer_m.csv"



# Fill in the line below to read the (benign) file into a variable cancer_b_data

cancer_b_data = pd.read_csv(cancer_b_filepath, index_col="Id")



# Fill in the line below to read the (malignant) file into a variable cancer_m_data

cancer_m_data = pd.read_csv(cancer_m_filepath, index_col="Id")
# Print the first five rows of the (benign) data

cancer_b_data.head()
# Print the first five rows of the (malignant) data

cancer_m_data.head()
# Fill in the line below: In the first five rows of the data for benign tumors, what is the

# largest value for 'Perimeter (mean)'?

max_perim = 87.46



# Fill in the line below: What is the value for 'Radius (mean)' for the tumor with Id 842517?

mean_radius = 20.57
# Histograms for benign and maligant tumors

sns.distplot(a=cancer_b_data['Area (mean)'], label="Benign", kde=False)

sns.distplot(a=cancer_m_data['Area (mean)'], label="Malignant", kde=False)

plt.legend()

# KDE plots for benign and malignant tumors

sns.kdeplot(data=cancer_b_data['Radius (worst)'], shade=True, label="Benign")

sns.kdeplot(data=cancer_m_data['Radius (worst)'], shade=True, label="Malignant")

# Path of the file to read

spotify_filepath = "../input/data-for-datavis/spotify.csv"



# Read the file into a variable spotify_data

spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)



# Line chart 

plt.figure(figsize=(12,6))

sns.lineplot(data=spotify_data)
# Change the style of the figure to the "dark" theme

sns.set_style("dark")



# Line chart 

plt.figure(figsize=(12,6))

sns.lineplot(data=spotify_data)
# Path of the file to read

spotify_filepath = "../input/data-for-datavis/spotify.csv"



# Read the file into a variable spotify_data

spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)
# Change the style of the figure

sns.set_style("dark")



# Line chart 

plt.figure(figsize=(12,6))

sns.lineplot(data=spotify_data)

# Change the style of the figure

sns.set_style("darkgrid")



# Line chart 

plt.figure(figsize=(12,6))

sns.lineplot(data=spotify_data)
# Change the style of the figure

sns.set_style("whitegrid")



# Line chart 

plt.figure(figsize=(12,6))

sns.lineplot(data=spotify_data)
# Change the style of the figure

sns.set_style("white")



# Line chart 

plt.figure(figsize=(12,6))

sns.lineplot(data=spotify_data)
# Change the style of the figure

sns.set_style("ticks")



# Line chart 

plt.figure(figsize=(12,6))

sns.lineplot(data=spotify_data)