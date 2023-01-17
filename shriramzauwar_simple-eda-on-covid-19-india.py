#importing all necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_excel("../input/Covid_19_India_latest_update.xlsx").rename(columns = {"Name of State/UT": "States"})

df.head(10)
# Set the width and height of the figure

plt.figure(figsize=(12,7))



# Add title

plt.title("Total confirmed cases")



# Bar chart showing Total confirmed cases according to states

sns.barplot(x=df['States'], y=df['Total Cases'])



# Add label for vertical axis

plt.ylabel("confirmed cases")
# Set the width and height of the figure

plt.figure(figsize=(12,6))



# Add title

plt.title("Total number of cases recovered")



# Bar chart showing No of confirmed cases recovered according to states

sns.barplot(x=df['States'], y=df['Recovered'])



# Add label for vertical axis

plt.ylabel("No of cases recovered")
# Set the width and height of the figure

plt.figure(figsize=(12,6))



# Add title

plt.title("Total number of deaths")



# Bar chart showing No number of deaths

sns.barplot(x=df['States'], y=df['Deaths'])



# Add label for vertical axis

plt.ylabel("No of Deaths")
# Set the width and height of the figure

plt.figure(figsize=(12,6))



# Add title

plt.title("Total number of cases today")



# Bar chart showing No number of deaths

sns.barplot(x=df['States'], y=df['Cases Today'])



# Add label for vertical axis

plt.ylabel("No of cases today")