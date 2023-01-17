import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # mat plot library



# Input data files are available in the "../input/" directory.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# read in the data and check the first 5 rows

df = pd.read_csv('../input/zoo-animals-median-life-expectancy/AZA_MLE_Jul2018_2.csv')

df.head()
# read in the data and check the last 5 rows

df = pd.read_csv('../input/zoo-animals-median-life-expectancy/AZA_MLE_Jul2018_2.csv')

df.tail()
# get the mean of the overall MLE

meanMLE = df['Overall MLE'].mean()

print("The mean overall life expectancy of the 330 species in this dataset is " + str(meanMLE) + " years.")



meanMaleMLE = df['Male MLE'].mean()

print("The mean male life expectancy of the 330 species in this dataset is " + str(meanMaleMLE) + " years.")



meanFemaleMLE = df['Female MLE'].mean()

print("The mean female life expectancy of the 330 species in this dataset is " + str(meanFemaleMLE) + " years.")
# plot a pie chart of the taxon classes to understand the makeup of the dataset

df['TaxonClass'].value_counts().plot(kind='pie')
# create a dataframe with the species with the longest median life expectancies

dftopmle = df.nlargest(10, ['Overall MLE'])



# create a horizonal bar chart with the top species and their overall MLE

# horizontal bar chart is so it's easier to see the species names

dftopmle.plot.barh(x='Species Common Name', y='Overall MLE', rot=0)
# create a dataframe with the species with the shortest median life expectancies

dfbottommle = df.nsmallest(10, ['Overall MLE'])



# create a horizonal bar chart with the bottom species and their overall MLE

# horizontal bar chart is so it's easier to see the species names

dfbottommle.plot.barh(x='Species Common Name', y='Overall MLE', rot=0)
# get the mean values by TaxonClass

dftaxons = pd.DataFrame(df.groupby('TaxonClass').mean())



# mean

dftaxons.plot.bar(y='Overall MLE', use_index=True)
# get average of overall MLE and store in a variable

overall_mle_mean = df['Overall MLE'].mean()



# get average of male MLE and store in a variable

male_mle_mean = df['Male MLE'].mean()



# get average of female MLE and store in a variable

female_mle_mean = df['Female MLE'].mean()



# create a dataframe with the new labels and values

meandf = pd.DataFrame({'Median Life Expectancy':['Overall', 'Male', 'Female'], 'Years':[overall_mle_mean, male_mle_mean, female_mle_mean]})



# plot the MLE means by gender in a bar chart

ax = meandf.plot.bar(x='Median Life Expectancy', y = "Years", rot=0)
# get the mean values by TaxonClass

dftaxons = pd.DataFrame(df.groupby('TaxonClass').mean())



# mean

dftaxons.plot.bar(y=['Overall MLE', 'Male MLE', 'Female MLE'], use_index=True)
# get the median values by TaxonClass

dftaxonsmed = pd.DataFrame(df.groupby('TaxonClass').median())



# mean

dftaxonsmed.plot.bar(y=['Overall MLE', 'Male MLE', 'Female MLE'], use_index=True)
# create a new column with the MLE difference by gender per species

df['MLE_Diff'] = df['Male MLE'] - df['Female MLE']



# find the species where the males live for longer

dfbiggestdif = df.nlargest(10, ['MLE_Diff'])



# create a horizonal bar chart with the top species and their overall MLE

# horizontal bar chart is so it's easier to see the species names

dfbiggestdif.plot.barh(x='Species Common Name', y='MLE_Diff', rot=0)
# find the species where the females live for longer

dfbiggestdif = df.nsmallest(10, ['MLE_Diff'])



# create a horizonal bar chart with the top species and their overall MLE

# horizontal bar chart is so it's easier to see the species names

dfbiggestdif.plot.barh(x='Species Common Name', y='MLE_Diff', rot=0)
# ask the user to enter an animal name and turn it to lowercase

print("Type an animal and then hit Enter:")

animal = input().lower()



# if the animal is in the data set, print out a graph with the male, female, and overall MLE

df.columns = [c.replace(' ', '_') for c in df.columns]



# turn all characters in species common name in df to lowercase

df['Species_Common_Name'] = df['Species_Common_Name'].str.lower()



# check to see if the user's input is in the data set

if df.Species_Common_Name.str.contains(animal).any():

    df_animal_only = df[df.Species_Common_Name.str.contains(animal)]

    animal_mle = df_animal_only["Overall_MLE"]

    animal_mle_output = animal_mle.values[0]

    print("The species '" + animal + "' is in the life expectancy data set and its overall MLE is", animal_mle_output, " years.")

    

    axes=plt.gca()



    # set the y-axis from 0-100

    axes.set_ylim([0,100])



    # plot a bar graph for the query

    plt.bar(animal,animal_mle_output, label = "Median Life Expectancy of Animal")

    plt.xlabel("Animal Species")

    plt.ylabel("Median Life Expectancy in Years")

    plt.title("Median Life Expectancy of Species")

    plt.legend()

    plt.show()

    

# if the animal is not in the data set, print out that the animal is not in the dataset

else: print(animal + " is NOT in the life expectancy data set.")



# connect to Google Trends query data via pytrends, the unofficial API for Google Trends

# learn more at https://github.com/GeneralMills/pytrends

from pytrends.request import TrendReq



# connect to Google

pytrends = TrendReq(hl='en-us', tz=360)



# pass in the user animal query

kw_list = [animal]



# build payload for interest over time

pytrends.build_payload(kw_list)



# create a dataframe of interest over time for the species queries week over week

# numbers represent search interest relative to the highest point on the chart for the given region and time. 

# a value of 100 is the peak popularity for the term. 

# a value of 50 means that the term is half as popular. 

# a score of 0 means there was not enough data for this term.

interest_over_time_df = pytrends.interest_over_time()

#print(interest_over_time_df)



# average the animal column to find the average of the query popularity over time

animal_popularity_mean = interest_over_time_df.mean()

animal_mean = animal_popularity_mean[0]

print("The mean popularity of the query '" + animal + "' on Google over time is", animal_mean)

axes=plt.gca()



# set the y-axis from 0-100

axes.set_ylim([0,100])



# plot a bar graph for the query

plt.bar(animal,animal_mean, label = "Mean Popularity of Query over Time")

plt.xlabel("Animal Query")

plt.ylabel("Mean Popularity")

plt.title("Mean Popularity of Query 2015-2020")

plt.legend()

plt.show()






