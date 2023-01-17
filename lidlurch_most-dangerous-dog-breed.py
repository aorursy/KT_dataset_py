import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization

import matplotlib.pyplot as plt #more data visualization

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore') # ignore warnings

from scipy.stats import ttest_ind # for the t-test we'll be doing

from subprocess import check_output 

print(check_output(["ls", "../input"]).decode("utf8"))
# Read in the Data

bites = pd.read_csv("../input/Health_AnimalBites.csv")

bites.head()
bites.shape #look at how many rows we have (rows, columns)
# Create a dataframe where there are only dogs included

dogs = bites.loc[bites['SpeciesIDDesc'] == 'DOG', :]

dogs.shape # prints out (rows, columns)
dogs_with_breed = dogs.dropna(subset = ['BreedIDDesc'])

dogs_with_breed.shape # prints out (rows, columns)
with sns.plotting_context('notebook', font_scale=2): #justs makes our breed names bigger

    sns.set_style("whitegrid") # makes our plot have a white background

    ax = plt.subplots(figsize=(20,25)) # makes our plot larger

    

    #Plot the number of dogs in each breed

    sns.countplot(y= 'BreedIDDesc' # the breeds go on our y axis

                  , data = dogs_with_breed # tells sns.countplot which dataset we're using

                  , order = dogs_with_breed['BreedIDDesc'].value_counts().index # Orders our results by size

                 )

    #Change aesthetic stuff

    plt.xticks(rotation=90) # rotates our x-axis labels so that they're readable

    plt.title('Count of Dog Bites by Breed', fontsize = 40) # Puts the title on with larger text size

    plt.xlabel('Count', fontsize = 35) # puts x axis label on with larger text size

    plt.ylabel('Breed', fontsize = 35) # puts y axis label on with larger text size

    plt.subplots_adjust(top=2, bottom=.8, left=0.10, right=0.95, hspace= 1

                        , wspace=0.5) # Changes the size of my bars and spacing
# Find out how many of the bites had a known outcome



rabies_data = dogs_with_breed.loc[dogs_with_breed['ResultsIDDesc'] != 'UNKNOWN', :] # Get rid of "UNKNOWN"

rabies_data = rabies_data.dropna(subset = ['ResultsIDDesc']) # Get rid of "NaN"

rabies_data = rabies_data.loc[dogs_with_breed['ResultsIDDesc'] == 'POSITIVE', :] #Only Display "POSITIVE" results

print('(rows, columns) = ', rabies_data.shape)

rabies_data.head()