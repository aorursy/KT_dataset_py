# install fuzzy package

!pip install Fuzzy
import fuzzy

print(fuzzy.nysiis('colour'))

print(fuzzy.nysiis('color'))
import pandas as pd

# Reading in datasets/nytkids_yearly.csv, which is semicolon delimited.

author_df = pd.read_csv("../input/nytkids_yearly.csv", delimiter=';')



# Looping through author_df['Author'] to extract the authors first names

first_name = []

for name in author_df['Author']:

  first_name.append(name.split()[0])



# Adding first_name as a column 

author_df['first_name'] = first_name





# Checking out the first few rows of author_df

author_df.head()
# Importing numpy

import numpy as np



# Looping through author's first names to create the nysiis (fuzzy) equivalent

nysiis_name = []

for name in first_name:

    nysiis_name.append(fuzzy.nysiis(name))



# Adding nysiis_name as a column to author_df

author_df['nysiis_name'] = nysiis_name



print(author_df.head())



# Printing out the difference between unique firstnames and unique nysiis_names:

print("\n\n difference between unique firstnames and unique nysiis_names:", len(np.unique(first_name)) - len(np.unique(nysiis_name)))
# Reading in datasets/babynames_nysiis.csv, which is semicolon delimited.

babies_df = pd.read_csv("../input/babynames_nysiis.csv", delimiter=';')



# # Looping through babies_df to and filling up gender

gender = []

for i in babies_df.index:

    if(babies_df['perc_female'][i] == babies_df['perc_male'][i]):

        gender.append('N')

    elif (babies_df['perc_female'][i] > babies_df['perc_male'][i]):

        gender.append('F')

    else:

        gender.append('M')



# # Adding a gender column to babies_df

babies_df['gender'] = gender



# Printing out the first few rows of babies_df

print(babies_df.head())
# This function returns the location of an element in a_list.

# Where an item does not exist, it returns -1.

def locate_in_list(a_list, element):

    loc_of_name = a_list.index(element) if element in a_list else -1

    return(loc_of_name)



# Looping through author_df['nysiis_name'] and appending the gender of each

# author to author_gender.

author_gender = []

for name in author_df['nysiis_name']:

  index = locate_in_list(list(babies_df['babynysiis']), name)

  if index >-1:

    author_gender.append(babies_df.loc[index]['gender'])

  else:

    author_gender.append('Unknown')



# Adding author_gender to the author_df

author_df['author_gender'] = author_gender



# Counting the author's genders

author_df['author_gender'].count()
# Creating a list of unique years, sorted in ascending order.

years = author_df['Year'].sort_values().unique()



# Initializing lists

males_by_yr = []

females_by_yr = []

unknown_by_yr = []



# Looping through years to find the number of male, female and unknown authors per year

for year in years:

    females_by_yr.append(len(author_df[

    (author_df["author_gender"] == 'F') &

    (author_df['Year']==year) ]))

    males_by_yr.append(len(author_df[

    (author_df["author_gender"] == 'M') &

    (author_df['Year']==year) ]))

    unknown_by_yr.append(len(author_df[

    (author_df["author_gender"] != 'M') &

    (author_df["author_gender"] != 'F') &

    (author_df['Year']==year) ]))



print(females_by_yr)

print(males_by_yr)

print(unknown_by_yr)

data = np.array([males_by_yr, females_by_yr, unknown_by_yr])

headers=['males', 'females', 'unknowns']

print(pd.DataFrame(data, headers, years))
# Importing matplotlib

import matplotlib.pyplot as plt



# This makes plots appear in the notebook

%matplotlib inline



# Plotting the bar chart

plt.bar(years, unknown_by_yr)



# [OPTIONAL] - Setting a title, and axes labels

plt.title("Unknown by Year")

plt.xlabel("Years")

plt.ylabel("Count")
# Creating a new list, where 0.25 is added to each year

years_shifted = years + 0.25

print(years_shifted)

# Plotting males_by_yr by year

plt.bar(years, males_by_yr, color='lightblue', width=0.25)



# Plotting females_by_yr by years_shifted

plt.bar(years_shifted, females_by_yr, color='pink', width=0.25)