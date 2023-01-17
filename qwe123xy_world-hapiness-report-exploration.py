# This study is my first data analyse. I will update it as i new things about data analyse.
#In this study, 2017 World Hapiness Report is examined using basic codes.
# DATA EXPLANATIONS
#Country: Name of the country.
#Region: Region the country belongs to.
#Happiness Rank: Rank of the country based on the Happiness Score.
#Happiness Score: A metric measured in 2015 by asking the sampled people the question: "How would you rate your happiness on a scale of 0 to 10 where 10 is the happiest."
#Standard Error: The standard error of the happiness score.
#Economy (GDP per Capita): The extent to which GDP (Gross Domestic Products) contributes to the calculation of the Happiness Score.
#Family: The extent to which Family contributes to the calculation of the Happiness Score
#Health (Life Expectancy): The extent to which Life expectancy contributed to the calculation of the Happiness Score
#Freedom: The extent to which Freedom contributed to the calculation of the Happiness Score.
#Trust (Government Corruption): The extent to which Perception of Corruption contributes to Happiness Score.
#Generosity: The extent to which Generosity contributed to the calculation of the Happiness Score.
#Dystopia Residual: The extent to which Dystopia Residual contributed to the calculation of the Happiness Score.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
data2017=pd.read_csv("../input/2017.csv")
data2017.columns
data2017.info()
data2017.describe()
data2017.head()
#let's change the column's names to handle data easily
data2017.columns=["country","hapiness_rank","hapiness_score","whisker_high","whisker_low",
                  "economy","family","health","freedom","generosity","trust","dystopia_residuel"]
data2017.columns
#let's look at the data correlation;
data2017.corr()
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(data2017.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
# Line Plot
#Line plot is better when x axis is time (but we do not have time depended variable)
# we put hapiness_score instead of time, and examine the changes economy and health via happiness score
plt.plot(data2017["hapiness_score"], data2017["economy"], color="red")
plt.plot(data2017["hapiness_score"], data2017["health"], color="blue")
plt.grid()
plt.xlabel("Hapiness Score")
plt.ylabel("Value")
plt.title("Economy & Health via Hapiness Score")
plt.legend()
plt.show()
# we can divide the graph
plt.subplot(2,1,1)
plt.plot(data2017["hapiness_score"], data2017["economy"], color="red")
plt.ylabel("Economy")
plt.subplot(2,1,2)
plt.plot(data2017["hapiness_score"], data2017["health"], color="blue")
plt.ylabel("Health")
plt.show()
# Scatter Plot
#Scatter is better when there is correlation between two variables
#let's examine hapiness score with economy and freedom

plt.scatter(data2017["hapiness_score"], data2017["economy"], color="red")
plt.scatter(data2017["hapiness_score"], data2017["freedom"], color="green")
plt.scatter(data2017["hapiness_score"], data2017["health"], color="blue")
plt.show()
# we can divide the graph
plt.subplot(3,1,1)
plt.scatter(data2017["hapiness_score"], data2017["economy"], color="red")
plt.ylabel("economy")
plt.title("Economy & Freedom & Health via Health Score")
plt.subplot(3,1,2)
plt.scatter(data2017["hapiness_score"], data2017["freedom"], color="green")
plt.ylabel("freedom")
plt.subplot(3,1,3)
plt.scatter(data2017["hapiness_score"], data2017["health"], color="blue")
plt.ylabel("health")
plt.xlabel("Health Score")
plt.show()
# Histogram
plt.hist(data2017["hapiness_score"], bins=40, color="brown", alpha=0.5)
plt.xlabel("Hapiness Score")
plt.ylabel("Frequency")
plt.title("Hapiness Histogram")
plt.show()
# Bar Plot
plt.bar(data2017["hapiness_score"], data2017["economy"], color="brown", alpha=0.5)
plt.xlabel("hapiness_score")
plt.ylabel("economy")
plt.title("Economy - Hapiness Score")
plt.grid()
plt.show()
# let's look at the average hapiness value
average_hapiness_score=data2017["hapiness_score"].mean()
print("Hapiness Score Average: ",average_hapiness_score)
# Add new column to decide which country over the avarage which not
data2017["hapiness_level"]=["high" if each>average_hapiness_score else "low" for each in data2017["hapiness_score"]]
data2017
#lets find the countries that economies higher than average but hapiness_level low
economy_average=data2017["economy"].mean()
print("economy average: ",economy_average)

#lets find the countries that health higher than average but hapiness_level high
health_average=data2017["health"].mean()
print("health average: ",health_average)
# Find the countries have higher economic value than average, but hapiness value is lower the average
data2017[(data2017["economy"]>economy_average) & (data2017["hapiness_level"]=="low")]
# Find the countries have lower health value than average, but hapiness value is higher the average
data2017[(data2017["health"]<health_average) & (data2017["hapiness_level"]=="high")]
#lets find which countries have economy and health values lower the average, but they are happy. 
data2017[(data2017["economy"]<economy_average) & (data2017["health"]<health_average) & (data2017["hapiness_level"]=="high")]
#what is the common value of these countries that have economy and health values LOWER the average but HAPPY.
family_average=data2017["family"].mean()
print("family average: ",family_average)
freedom_average=data2017["freedom"].mean()
print("freedom average: ",freedom_average)
generosity_average=data2017["generosity"].mean()
print("generosity average: ",generosity_average)
trust_average=data2017["trust"].mean()
print("trust average: ",trust_average)
dystopia_residuel_average=data2017["dystopia_residuel"].mean()
print("dystopia residuel average :",dystopia_residuel_average)
x=data2017[(data2017["economy"]<economy_average) & (data2017["health"]<health_average) & (data2017["hapiness_level"]=="high")]
x
family_list=["low" if each<family_average else "high" for each in x["family"]]
print("family_list: ",family_list)
freedom_list=["low" if each<freedom_average else "high" for each in x["freedom"]]
print("freedom_list: ",freedom_list)
generosity_list=["low" if each<generosity_average else "high" for each in x["generosity"]]
print("generosity_list: ",generosity_list )
trust_list=["low" if each<trust_average else "high" for each in x["trust"]]
print("trust_list: ",trust_list)
dystopia_residuel_list=["low" if each<dystopia_residuel_average else "high" for each in x["dystopia_residuel"]]
print("dystopia_residuel_list: ",dystopia_residuel_list)
# Only FREEDOM value is higher the average in all of the countries we examine so it shows us importance of FREEDOM upon HAPINESS.

