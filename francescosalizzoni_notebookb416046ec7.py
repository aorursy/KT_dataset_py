#importing basic libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

from collections import Counter
#let's read our files and store them into artists and artworks variables 

artists = pd.read_csv("../input/artists.csv")

artworks = pd.read_csv("../input/artworks.csv")
#artists.sort_index(axis=1, ascending=False)

#artists.sort_values(by = "Nationality")



a = artists['Gender'].to_string()

print(a)

#taking a look at the artists dataset

artists.head(3)
#running the .info() method to get the dataset first informations

artists.info()
print("Nationality has {}% of null values".format(1- artists["Nationality"].count()/artists["Artist ID"].count()))

print("Gender has {}% of null values".format(1 - artists["Gender"].count()/artists["Artist ID"].count()))

print("Birth Year has {}% of null values".format(1- artists["Birth Year"].count()/artists["Artist ID"].count()))

print("Death Year has {}% of null values".format(1- artists["Death Year"].count()/artists["Artist ID"].count()))
NatList = [x for x in artists["Nationality"] if str(x) != 'NaN']

NatList = [x for x in artists["Nationality"] if str(x) != 'nan']

z = pd.DataFrame(NatList)







nat_counts = Counter(NatList)



df = pd.DataFrame.from_dict(nat_counts, orient='index')

df1 = df.reset_index()

df1 = df1.rename(columns= {'index':'country'})

df1.sample(5)

#taking a look at the artworks dataset

artworks.head(2)
#how many artists per nationlity we have?

apn = artists["Nationality"].value_counts()



#how many males and females?

gender = artists["Gender"].value_counts()

#print(apn)

#print(gender)

#checking how many nationalities we have in our dataset

print(len(artists['Nationality'].unique()))



artists['Birth Year']