#Libraries 

import numpy as np # linear algebra

import pandas as pd # data processing



#Imports files from local directory to kaggle. 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Import Apriori Libraries 

from mlxtend.frequent_patterns import apriori #Apriori Algorithm that operates on databases with transactions

from mlxtend.frequent_patterns import association_rules #Installs the Apriori Association Rules
#Loading data into Kaggle 

podcast_data = pd.read_csv("/kaggle/input/datapodcast/Podcast.csv")

print(podcast_data)
#Data Cleaning 

podcast = podcast_data.dropna() #Removes NaN values

podcast.drop(["Podcast_Name"], axis =1, inplace = True )

print(podcast)
#Checking the frequency of each variable

from collections import Counter



c = [Counter(j for j in i).items() for i in podcast.values.T]

pd.DataFrame.from_records(c, index=podcast.columns).T
#Get the dimensions of the dataset

print(podcast.shape)
#Training the model 

itemsets = apriori(podcast, min_support=.01, use_colnames= True )

rules = association_rules(itemsets, metric = "support", min_threshold = .1)

rules
#Viewing based on max lift

rules.sort_values(by = "lift", ascending = False)
#Filtering out based on conditions

rules [(rules['lift']>1)& (rules['confidence']>.1 ) & (rules['support']>.01)]