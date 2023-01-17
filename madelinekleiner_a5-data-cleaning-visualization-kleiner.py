import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt # mat plot library



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# read in the data and check the first 5 rows

df = pd.read_csv('../input/seattle-pet-licenses-2020/Seattle_Pet_Licenses.csv')

df.head()
# read in the data and check the last 5 rows

df = pd.read_csv('../input/seattle-pet-licenses-2020/Seattle_Pet_Licenses.csv')

df.tail()
# plot a pie chart of the species

df['Species'].value_counts().plot(kind='pie')
# get a dataframe for dogs only

dfdogs = df[df.Species == 'Dog']



# get the values for each dog breed

dfdogbreeds = dfdogs['Primary Breed'].value_counts()



# new table with top 10 dog breeds

dftopdogbreeds = dfdogbreeds.head(10)



# plot the top dog breeds in a bar chart

dftopdogbreeds.plot(kind='bar')
# get a dataframe for cats only

dfcats = df[df.Species == 'Cat']



# get the values for each cat breed

dfcatbreeds = dfcats['Primary Breed'].value_counts()



# new table with top 10 cat breeds

dftopcatbreeds = dfcatbreeds.head(10)



# plot the top cat breeds in a bar chart

dftopcatbreeds.plot(kind='bar')
# new data frame with number of dog names

dfdogsnames = dfdogs["Animal's Name"].value_counts()



# get the top 10 dog names

dftopdogsnames = dfdogsnames.head(10)



# plot the top 10 dog names in a bar chart

dftopdogsnames.plot(kind='bar')
# new data frame with number of cat names

dfcatsnames = dfcats["Animal's Name"].value_counts()



# get the top 10 cat names

dftopcatsnames = dfcatsnames.head(10)



# plot the top 10 cat names in a bar chart

dftopcatsnames.plot(kind='bar')
import squarify



# count the number of dogs by zip code

dfdogszip = dfdogs["ZIP Code"].value_counts()

dfdogszip = pd.DataFrame(dfdogszip)



# create a dataframe with zip codes and values for zips with more than 1 dog only

dfdogszipmorethanone = dfdogszip[dfdogszip.xs('ZIP Code', axis=1) > 1]



# plot a treemap for all zip codes with more than 1 dog

squarify.plot(sizes=dfdogszipmorethanone['ZIP Code'], label=dfdogszipmorethanone.index, alpha=.8)

plt.axis('off')

plt.show()
import squarify



# count the number of cats by zip code

dfcatszip = dfcats["ZIP Code"].value_counts()

dfcatszip = pd.DataFrame(dfcatszip)



# create a dataframe with zip codes and values for zips with more than 1 cat only

dfcatszipmorethanone = dfcatszip[dfcatszip.xs('ZIP Code', axis=1) > 1]



# plot a treemap for all zip codes with more than 1 cat

squarify.plot(sizes=dfcatszipmorethanone['ZIP Code'], label=dfcatszipmorethanone.index, alpha=.8)

plt.axis('off')

plt.show()