# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#I made one of my files into a dataframe and examined the first 5 lines to see how it was organized

stranger = pd.read_csv('/kaggle/input/searchofnetflixtop10/strangerthings_bycountry.csv')

stranger.head()
#I merged two of the files together into a single dataframe

reasons = pd.read_csv('/kaggle/input/searchofnetflixtop10/13reasons_bycountry.csv')

allshows = pd.merge(stranger, reasons, how='outer', on='Country')

#creating a list of the rest of the files

all_files = ['/kaggle/input/searchofnetflixtop10/deadtome_bycountry.csv',

 '/kaggle/input/searchofnetflixtop10/raisingdion_bycountry.csv',

 '/kaggle/input/searchofnetflixtop10/sexeducation_bycountry.csv',

 '/kaggle/input/searchofnetflixtop10/umbrellaacademy_bycountry.csv',

 '/kaggle/input/searchofnetflixtop10/unbelievable_bycountry.csv',

 '/kaggle/input/searchofnetflixtop10/whentheyseeus_bycountry.csv',

 '/kaggle/input/searchofnetflixtop10/witcher_bycountry.csv',

 '/kaggle/input/searchofnetflixtop10/youseason2_bycountry.csv']
#iterated through each of the files in the list to add them to the joint dataframe and storing it as a csv

for f in all_files:

    newshow = pd.read_csv(f) 

    allshows = pd.merge(allshows, newshow, how='outer', on='Country')

    allshows.to_csv("combined_csv.csv", index='Country', encoding='utf-8-sig')

#checking my work

fullshowdf = pd.read_csv('/kaggle/working/combined_csv.csv')

fullshowdf.head()
fullshowdf.columns #looking at current column names
#changing the column names and replacing null with 0 to make it easier to work with and checking my work

fullshowdf.columns =['Index', 'Country', 'Stranger Things', '13 Reasons', 'Dead2Me', 'Raising Dion', 'Sex Education', 'Umbrella Academy', 'Unbelievable', 'When they see us', 'The Witcher', 'You']

fullshowdf.drop(['Index'], axis = 1, inplace = True)  #removed the extra column that had been added

#fullshowdf.fillna(0, inplace=True) 

fullshowdf.head()

#importing matplotlib

import matplotlib.pyplot as plt





fullshowdf['Dead2Me'].plot(kind = 'bar') #I am able to plot the values but there are too many countries, most of which are NaN

fullshowdf = fullshowdf.dropna() #I'll get rid of the NaN values which will also make it easier to compare the shows since we'll be looking across all the same countries

fullshowdf.tail(15)
results_sorted = fullshowdf.sort_values('Dead2Me',ascending=False) #sorting the results of one column

results_sorted.plot(x='Country', y='Dead2Me', kind = 'barh') #creating a bar chart of that column
def sorted_bar(showname):  #function that will create a sorted bar chart for any of the shows

    results_sorted = fullshowdf.sort_values(showname,ascending=True)

    results_sorted.plot(x='Country', y=showname, kind = 'barh', figsize=(12,8)) #I made the figure size bigger so it's easier to read the country name

    

sorted_bar('13 Reasons')



#fullshowdf.groupby("Country")['Dead2Me'].plot(kind='bar')
fullshowdf.get_dtype_counts()
fullshowdf.replace('<1', 1, inplace = True) #replaced the problem in the Raising Dion Row

fullshowdf['Raising Dion'] = fullshowdf['Raising Dion'].astype(float)

fullshowdf['When they see us'] = fullshowdf['When they see us'].astype(float) #changed the two problem columns into floats

fullshowdf.head()
showcolumns = fullshowdf.select_dtypes(include = ['float']).columns

print(showcolumns)
for show in showcolumns:

    sorted_bar(show)
fullshowdf.plot(x='Country', y=showcolumns, figsize=(20,10)) #seeing how it looks with all of the shows as bars 