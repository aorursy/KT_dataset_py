import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# Importing all the required libraries
df = pd.read_csv("../input/dataset.csv")

df
df.columns
df.describe()
df = df[df.WHOIS_COUNTRY != 'None']

whois = df.WHOIS_COUNTRY.value_counts()

# filtering out malicious website that dont belong to any country

whois

df['WHOIS_COUNTRY'].unique()
df
df.isnull().sum()

#finding if we have null values in columns
df_mal = df[df.Type != 0]  

# 

df_mal
(whois/df.shape[0]).plot(kind="bar");

plt.title("WHOIS_COUNTRY");

# plotting bar graph for entire data
# function to make plot

def makePlot(city_count,column, title, ylabel, xlabel):

    """

    This function takes in common paramters and produces a plot 

    

    Parameter:

    column(str): name of column from the dataframe

    title(str): title of the chart

    xlabel(str): x-axis title

    ylabel(str): y-axis title



    """



    plt.figure(figsize=(15,5))

    sns.barplot(city_count.index, city_count.values, alpha=0.8)

    plt.title(title)

    plt.ylabel(ylabel, fontsize=15)

    plt.xlabel(xlabel, fontsize=15)

    plt.show()
    city_count  = df['WHOIS_COUNTRY'].value_counts()

    city_count 

    city_count = city_count[:10,]

#malicious data

makePlot(city_count,'WHOIS_COUNTRY', 'Top 10 countries in the World', 'Number of Occurrences','Country')



# Get country data.
city_count  = df_mal['WHOIS_COUNTRY'].value_counts()

city_count = city_count[:10,]

makePlot(city_count,'WHOIS_COUNTRY', 'Top 10 countries in the World', 'Number of Malcious Websites','Country')



#malicious data
df_mal_state = df_mal[df_mal.WHOIS_COUNTRY == 'ES']

city_count  = df_mal_state['WHOIS_STATEPRO'].value_counts()

makePlot(city_count,'WHOIS_STATEPRO', 'States in Spain', 'Number of Malcious Websites','State')



# data of barcelona - malicious
corrMatrix = df.corr()

plt.figure(figsize = (20,20))

sns.heatmap(corrMatrix,linewidths=2, annot=True)

plt.show()

#plotting the corellation matrix
sns.heatmap(df.isnull(), cbar=False)