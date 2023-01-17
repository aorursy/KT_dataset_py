## Cleaning the datasets in the input folder using numpy and pandas

import numpy as np 

import pandas as pd 



import os

print(os.listdir("../input"))



## Our datasets:

## BL-Flickr-Images-Book.csv: A csv file containing information about books from the British Library

## University_towns.txt: A text file containing names of college towns in every US state

## Olympics.csv: A csv file summarizing the participation of all countries in the Summer and Winter Olympics

## BL-Flickr-Images-Book.csv

## Here we'll clean by dropping unnecessary columns, assinging the correct dtypes to

## important columns, defining a more useful indexing, getting rid of records with no dates, 

## and fixing some value inconsistencies



## let's begin by creating a dataframe out of the 'BL-Flickr-Images-Book' file and see some sample

df_book = pd.read_csv("../input/BL-Flickr-Images-Book.csv")

df_book.head()
## Dropping unnecessary columns



## We can see some columns provide ancillary information that would be useful for the library 

## but aren't very descriptive of the books

## I'll drop those columns to have just the information about the books



# list of columns to be dropped

to_drop = ['Edition Statement', 'Corporate Author', 'Corporate Contributors',

           'Former owner', 'Engraver', 'Contributors', 'Issuance type', 'Shelfmarks']



df_book.drop(columns=to_drop, inplace=True)

df_book.head()
## Setting the Identifier column = index



## We can assume that when someone wants to access the information about a particular book

## they'd look up using some unique identifier



df_book['Identifier'].is_unique ## True



## in order to make sure it would be a good idea to use this column as index, let's see at some example

df_book.loc[0] 



## it can be noticed that the 1st record in our dataframe has the identifier 206 which can be accesed 

## in a straightforward way with loc[] 
## loc[] allows us to to label-based indexing (labeling of a row regardless of its position),

## so why not use the 'Identifier' column as index so it's easier just to look up all the records

## using their index which -after update- will be the same as their identifier



df_book.set_index('Identifier', inplace=True)

df_book.loc[206]



## we notice that the index or our 1st record has been updated from being 0 to being 206 

## which is the same as its identifier
## the same record can also be accessed by position (as we know it's 1st) using iloc[]

## which does position-based indexing

df_book.iloc[0]



## and we see that it's the same as above
## Cleaning fields in the data (dtypes and data inconsistencies)



## it's important to get specific columns like date to a uniform format for better understanding

## and enforce consistency



## let's first check the different datatypes we have in our dataset

df_book.get_dtype_counts() # 6

df_book.dtypes
## all of the datatypes are currently 'object' ~ str in native Ptyhon

## 'Date of Publication' is one column where it makes sense to enforce a numeric value so calculations

## can be performed later on



# let's check a few values first

df_book['Date of Publication'].head(15)
## we can notice a few issues here:

## 1. extra dates in []: 1879 [1868]

## 2. date ranges: 1839, 38-54

## 3. NaNs



## we can fix these with regex as we notice that we only need the first 4 digits to get our year correct

regex = r'^(\d{4})' # to find any 4 digits at the beginning

date_num = df_book['Date of Publication'].str.extract(regex, expand=False)

date_num.head(15)
## now that we've got our desired year values, the dtype of this column is still 'object'

## let's make it numeric and check

df_book['Date of Publication'] = pd.to_numeric(date_num, downcast='integer')

df_book['Date of Publication'].dtype
## we still have NaNs, let's check how many

df_book['Date of Publication'].isnull().sum() # 971



## maybe a percentage over the whole set is better to decide is it's problematic

df_book['Date of Publication'].isnull().sum() / len(df_book)

## ~11.72% of our data is missing which is not a big problem for us right now



# but still we can just skip those records

df_book = df_book[df_book['Date of Publication'].isnull() != True]

df_book.head(15)
## Now let's check a str column 'Place of Publication' for inconsistencies

df_book['Place of Publication'].head(15)



## we see: for some rows the place of publications contains some unnecessary information

## further check leads to identifing 'London' and 'Oxford' to be the values concerned
## comparing 2 entries for another issue

book_idx = [4157862, 4159587]

df_book.loc[book_idx]



## we see: both values are same but one of them is separated using '-'
## let's fix all the records for the aforementioned cases with str methods combined with numpy

df_book['Place of Publication'] = np.where(df_book['Place of Publication'].str.contains('London'), 'London',

                                           df_book['Place of Publication'].str.replace('-', ' '))



## let's test specific examples that we know earlier were to be fixed                                                   

book_idx = [216, 4157862, 4159587]

df_book.loc[book_idx]



## And that's all for this dataset
## University_towns.txt

## Here not much has to be done as it's a small dataset, 

## what we'll do is update the state and region names to

## more sensible values, and we'll do that for the whole file at once 

## by using applymap function



## let's check the file

uni_towns = []

with open("../input/university_towns.txt") as file:

    for line in file:

        if '[edit]' in line:

            state = line

        else:

            uni_towns.append((state, line))

            

uni_towns[:5]
## now let's create a dataframe 

df_towns = pd.DataFrame(uni_towns, columns=['State', 'RegionName'])



df_towns.head()
## those []s in State and ()s in RegionName are just useless here, why not get rid of them

## defining a function to just get the city and region names 

def get_cityregion(item):

    if ' (' in item:

        return item[:item.find(' (')]

    elif '[' in item:

        return item[:item.find('[')]

    else:

        return item
## and now let's apply this function to our dataframe

df_towns = df_towns.applymap(get_cityregion)

df_towns.head()



## Tada! Now we have a nice looking and noise free dataframe to work with

## And that's all for this file
## Olympics.csv

## Here we'll do some renaming of columns 

## and skipping of not-so-descriptive records



## let's again start with a dataframe (it's easrier to look at data with df :D )

df_olym = pd.read_csv("../input/olympics.csv")

df_olym.head()
## ew! that's messy, just look at those column names

## we see: the row which should've been our header (the one that's used for column names)

## is at 

df_olym.iloc[0]



## so now we can say maybe

## ? Summer = Summer Games

## 01 ! = Gold...



## we need to fix these 2 things:

## skip one row and set the header as the first (0-index) row

## Rename the columns
## Skipping the 1st row can already be done while creating the dataframe

df_olym = pd.read_csv("../input/olympics.csv", header=1)

df_olym.head()



## Cool! We have our column names, but not so descriptive
## creating a dict to map the current col names to more useful ones

new_col_names =  {'Unnamed: 0': 'Country',

                  '? Summer': 'Summer Olympics',

                  '01 !': 'Gold',

                  '02 !': 'Silver',

                  '03 !': 'Bronze',

                  '? Winter': 'Winter Olympics',

                  '01 !.1': 'Gold.1',

                  '02 !.1': 'Silver.1',

                  '03 !.1': 'Bronze.1',

                  '? Games': '# Games',

                  '01 !.2': 'Gold.2',

                  '02 !.2': 'Silver.2',

                  '03 !.2': 'Bronze.2'}



new_col_names
## rename the columns 

df_olym.rename(columns=new_col_names, inplace=True)
## let's check

df_olym.head()



## much better! 

## And that's all for this dataset
## That's all for this data cleaning task!