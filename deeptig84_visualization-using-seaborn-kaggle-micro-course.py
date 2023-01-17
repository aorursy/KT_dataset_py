# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

print("Setup Complete")
fifa = pd.read_csv("../input/fifa.csv",index_col = "Date", parse_dates=True)
fifa.head()
# Set the width and height of the figure

figure = plt.figure(figsize=(16,10))

# Line chart showing how FIFA rankings evolved over time 

sns.lineplot(data=fifa)
spotify  = pd.read_csv("../input/spotify.csv",index_col = "Date", parse_dates=True)
# Set Plot Figure size inches

plt.figure(figsize=(20,10))

# Add Title

plt.title("Daily Global Streams Trend of popular songs in 2017-18")

sns.lineplot(data=spotify)

# Plot data only related to any 2 songs

list(spotify.columns)
#  Create a figure

plt.figure(figsize=(20,10))



# Set Title of the plot

plt.title("Daily Trend of global popular songs 2017-2018")



# Line chart showing streaming trend of Despacito

sns.lineplot(data=spotify['Despacito'],label = 'Despacito')



# Line chart showing streaming trend of Shape of You

sns.lineplot(data=spotify['Shape of You'],label='Shape of You')



# Add label to horizontal axis



plt.xlabel("Dates")

# Path of the file to read

flight_filepath = "../input/flight_delays.csv"

# Read the file into a variable flight_data

flight_data = pd.read_csv(flight_filepath, index_col="Month")
flight_data
# Set the Plot figure dimensions



plt.figure(figsize=(15,8))



# Add Title

plt.title("Average Arrival Delay for Spirit Airlines Flights by month")



# Create barplot

sns.barplot(x=flight_data.index,y=flight_data['NK'])



#Create Label for vertical axis

plt.ylabel("Arrival Delay(in minutes)")

# Set width and height of the figure

plt.figure(figsize=(20,15))



# Add Title

plt.title("Average delay for each flight by month")



# Draw heatmap showing average arrival delay of each airline by month

sns.heatmap(data=flight_data,annot=True)



# Add label for horizontal axis

plt.xlabel("Airline")
# Path of the file to read

ign_filepath = "../input/ign_scores.csv"



# Load the data into a dataframe

ign_data = pd.read_csv(ign_filepath,index_col = 'Platform')
# view the data

ign_data
# Set the plot figure dimensions

plt.figure(figsize=(10,6))



# Add Title

plt.title("Average score for Racing Games, by platform")



# Create barplot

sns.barplot(x=ign_data['Racing'],y=ign_data.index)
# Heatmap showing average game score by platform and genre

# Create figure size for plot

plt.figure(figsize=(16,10))



# Add Title 

plt.title("Average score for each video game genre , by platform")



# Create HEatmap

sns.heatmap(data=ign_data,annot=True)





# Path of the file to read

insurance_filepath = "../input/insurance.csv"



# Read the file into a variable insurance_data

insurance_data = pd.read_csv(insurance_filepath)
insurance_data.head()
plt.figure(figsize=(10,5))

sns.scatterplot(x= insurance_data['bmi'],y=insurance_data['charges'])

plt.title("BMI vs Insurance Charges")
plt.figure(figsize=(10,5))

sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])

plt.title("BMI vs Insurance Charges with regression line")
plt.figure(figsize=(10,5))

sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])

plt.title("BMI vs Insurance Charges for Smoker/Non smoker")

plt.figure(figsize=(10,6))

sns.lmplot(x='bmi', y='charges', hue='smoker',data=insurance_data)

plt.title("BMI vs Insurance Charges for Smoker/non-Smoker Regression Line")
plt.figure(figsize=(10,6))

sns.swarmplot( y='charges', x='smoker',data=insurance_data)

plt.title("Smoker vs Insurance Charges")
# Path of the file to read

candy_filepath = "../input/candy.csv"



# load the csv file

candy_data = pd.read_csv(candy_filepath,index_col ='id')
candy_data.head()
# Which candy was more popular with survey respondents: '3 Musketeers' or 'Almond Joy'

Musketeers = candy_data[candy_data['competitorname'] == '3 Musketeers']['winpercent']

AlmondJoy = candy_data[candy_data['competitorname'] == 'Almond Joy']['winpercent']

print(Musketeers,AlmondJoy)



# Which candy has higher sugar content: 'Air Heads' or 'Baby Ruth'?



AirHeads = candy_data[candy_data['competitorname'] == 'Air Heads']['sugarpercent']

BabyRuth = candy_data[candy_data['competitorname'] == 'Baby Ruth']['sugarpercent']



print(AirHeads,BabyRuth)
# Create a scatter plot that shows the relationship between `'sugarpercent'` (on the horizontal x-axis) and `'winpercent'` (on the vertical y-axis)



plt.figure(figsize=(10,6))

sns.scatterplot(x='sugarpercent',y='winpercent', data=candy_data)

plt.title("Suger v/s Popularity Trends in Candies")
# Create the same scatter plot you created above but now with a regression line!

plt.figure(figsize=(10,6))

sns.regplot(x='sugarpercent',y='winpercent', data=candy_data)

plt.title("Suger v/s Popularity Trends in Candies Regression Line")
# create a scatter plot to show the relationship between `'pricepercent'` (on the horizontal x-axis) and `'winpercent'` 

#  (on the vertical y-axis). Use the `'chocolate'` column to color-code the points.



plt.figure(figsize=(10,6))

sns.scatterplot(x='pricepercent', y='winpercent',hue='chocolate',data=candy_data)

plt.title("Price v/s Popularity percentage trend for Choco/Non Choco candies")
# Create the same scatter plot you created in **Step 5**, but now with two regression lines, corresponding to (1) chocolate candies and (2) candies without chocolate.

plt.figure(figsize=(10,6))

sns.lmplot(x='pricepercent', y='winpercent',hue='chocolate',data=candy_data)

plt.title("Price v/s Popularity percentage trend for Choco/Non Choco candies with regression line")
# Create a categorical scatter plot to highlight the relationship between `'chocolate'` and `'winpercent'`.  Put `'chocolate'` on the (horizontal) x-axis, and `'winpercent'` on the (vertical) y-axis.

plt.figure(figsize=(10,6))

sns.swarmplot(x='chocolate', y='winpercent', data=candy_data)

plt.title("Popularity comparison for Choco/Non Choco candies")

# Path of the file to read

iris_filepath = "../input/iris.csv"



# Read the file into a variable iris_data

iris_data = pd.read_csv(iris_filepath, index_col="Id")



# Print the first 5 rows of the data

iris_data.head()
# Petal length distribution

plt.figure(figsize=(10,6))

sns.distplot(a=iris_data['Petal Length (cm)'], kde=False)

plt.title("Petal Length Distribution")
plt.figure(figsize=(10,6))

sns.kdeplot(data=iris_data['Petal Length (cm)'], shade=True)

plt.title("Petal Length Smooth Density Distribution")
# 2D KDE of petal Length and Petal Width

plt.figure(figsize=(10,6))

sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind="kde")

plt.title("Petal Length and Sepal Width Smooth Density Distribution")
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
iris_vir_data.head()
iris_set_data.head()
plt.figure(figsize=(10,6))

sns.distplot(a=iris_set_data['Petal Length (cm)'],kde=False, label = "Setosa")

sns.distplot(a=iris_vir_data['Petal Length (cm)'],kde=False, label ="Versicolor")

sns.distplot(a=iris_ver_data['Petal Length (cm)'],kde=False, label ="Virginica")

plt.title("Petal Length Distribution of various species")

plt.legend()
plt.figure(figsize=(10,6))

sns.kdeplot(data=iris_set_data['Petal Length (cm)'],shade=True, label = "Setosa")

sns.kdeplot(data=iris_vir_data['Petal Length (cm)'],shade=True, label ="Versicolor")

sns.kdeplot(data=iris_ver_data['Petal Length (cm)'],shade=True, label ="Virginica")

plt.title("Petal Length Distribution of various species")

plt.legend()
# Paths of the files to read

cancer_b_filepath = "../input/cancer_b.csv"

cancer_m_filepath = "../input/cancer_m.csv"



# read the (benign) file into a variable cancer_b_data

cancer_b_data = pd.read_csv(cancer_b_filepath,index_col='Id')



# read the (malignant) file into a variable cancer_m_data

cancer_m_data = pd.read_csv(cancer_m_filepath,index_col='Id')
cancer_b_data.head()
cancer_m_data.head()
# create two histograms that show the distribution in values for `'Area (mean)'` for both benign and malignant tumors.

plt.figure(figsize=(10,6))

sns.distplot(a=cancer_b_data['Area (mean)'],kde=False, label="Benign")

sns.distplot(a=cancer_m_data['Area (mean)'],kde=False, label="Malignant")

plt.legend()
# Use the code cell below to create two KDE plots that show the distribution in values for `'Radius (worst)'` for both benign and malignant tumors.

plt.figure(figsize=(10,6))

sns.kdeplot(data=cancer_b_data['Radius (worst)'],shade=True,label='benign')

sns.kdeplot(data=cancer_m_data['Radius (worst)'],shade=True,label="Malignant")

plt.title("Benign v/s Malignant Tumor Radius Worst Distribution")

plt.legend()
# Change Theme

sns.set_style("dark")

plt.figure(figsize=(20,8))

sns.lineplot(data=spotify)
