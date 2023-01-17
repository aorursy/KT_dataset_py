# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sbor # for visualization 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# 'NBA player of the week' data been chosen as the trial data (Player of the week data from 1984-1985 to 2017-2018)
data = pd.read_csv('../input/NBA_player_of_the_week.csv')
# get the data info
data.info()
# data correlation in tabular format?
data.corr()

# as expected, correlation within numbered items only (4xint64 and 1xfloat64)
# some high correlations do occur:
# - Seasons in league <-> Age
# - Real_value <-> Draft Year (negative)
# - Draft Year <-> Season short
# - Seasons short <-> Real_value (negative)
# correlation map
f,ax = plt.subplots(figsize=(14, 14))
sbor.heatmap(data.corr(), annot=True, linewidths=.2, fmt= '.1f',ax=ax)
plt.show()

# dark ones are the negative correlations (mutual relations)
# light creme ones (apart from the white-ish diagonal relations) are the
#                               positive correlations (mutual relations)

print(data.head(10))   #first 10 rows
print(data.tail(10))   #last 10 rows

data.columns   #name of columns, or "features", namely
#feature/column name format improvments: (I will check a smarter way to write a method to clean the naming mess:
#                          split methods for 2 and for 3 words, made below in a very strange way, but it works!)
data.columns = [each.lower() for each in data.columns];  #all letters in lower
# string concat operations
data.columns = [each.split()[0]+"_"+each.split()[1]+"_"+each.split()[2] if(len(each.split())>2) else each for each in data.columns];
data.columns = [each.split()[0]+"_"+each.split()[1] if(len(each.split())>1) else each for each in data.columns];
print(data.columns);

# Another way would be to re-assign the column names of the data frame, after correcting them manually
# Not a nice way to do, but for a limited subset of column names, it may help to move quick
# alternative assignment is given below (commented out, not functional):

#data.columns = ['age', 'conference', 'date', 'draft_year', 'height', 'player',
#       'position', 'season', 'season_short', 'seasons_in_league', 'team',
#       'weight', 'real_value'


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.age.plot(kind = 'line', color = 'g',label = 'Age',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.real_value.plot(color = 'b',label = 'Real Value',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
data.seasons_in_league.plot(color = 'r',label = 'Seasons in League',linewidth=1, alpha = 0.5,grid = True)
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
# x = attack, y = defense
data.plot(kind='scatter', x='season_short', y='real_value',alpha = 0.5,color = 'red')
plt.xlabel('Season Short')              # label = name of label
plt.ylabel('Real Value')
plt.title('Real Value Distribution versus The Season Year')            # title = title of plot
plt.show()       # explanation for the real_value distribution given in the original data set: If two awards given at the same week [East & West]
#                   the player got 0.5, else 1 point.  Thus, below Plot shows that, although very naive & crude simple, after 
# Histogram    Choosing a proper feature to be shown by Histogram makes the visualization much clear, proper and readable.
# bins = number of bar in figure
data.age.plot(kind = 'hist',bins = 40,figsize = (10,10))
plt.show()
#Create multiple dictionaries from the same csv file and look its keys and values

#Index(['age', 'conference', 'date', 'draft_year', 'height', 'player',
#       'position', 'season', 'season_short', 'seasons_in_league', 'team',
#       'weight', 'real_value'],
#      dtype='object')

# Some examples of dictionaries created from the DataFrame of the file 'NBA_player_of_the_week.csv'

# first, a query to check the uniqueness in keys (player as an example):
data["player"].unique()
len(data["player"].unique())

dict_Of_Age = {} #dictionary of 'player' (index:5) and 'age'(index:0) relationship (the call len(data["player"].unique()) gives 274 unique player's name, 
                    # but person-age relationship -although a little weird- depends on the year in which that specific player won the point(real_value)
                    # so dimension reduction is not meaningfull cause we loose the player's age info at that year 
                    # and to make it more complex: it can all be calculated just after the first entry, but too heavy I guess for that homework :)
dict_Of_Age = {row[5] : row[0] for _, row in data.iterrows()}
len(dict_Of_Age)    # unfortunately, stuck with unique 274 records! But which ones? Presumably first entries by default...

dict_Of_Height = {} #dictionary of 'player'(index:5) and 'height'(index4) relationship
dict_Of_Height = {row[5] : row[4] for _, row in data.iterrows()}
len(dict_Of_Height)    # unfortunately, stuck with unique 274 records! But which ones? Presumably first entries by default...

dict_Of_Players_Real_Value = {} # the only "rationaly full" list-dictionary by 'player'(index:5) and 'real_value'(index12). 
dict_Of_Players_Real_Value = {row[5] : row[12] for _, row in data.iterrows()}
len(dict_Of_Players_Real_Value)    # unfortunately, stuck with unique 274 records! But which ones? Presumably first entries by default...

# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique

dict_Of_Age.clear()                   # remove all entries in dict
dict_Of_Height.clear()                   # remove all entries in dict
#dict_Of_Players_Real_Value.clear()                   # remove all entries in dict BUT not executed to be used in below loop/while work




# data = pd.read_csv('../input/NBA_player_of_the_week.csv') 
# I didn't want to reload the dataframe cause, I've already made category naming modifications (lower case and concat operations)

series = data['age']        # data['age'] = series
print(type(series))
data_frame = data[['age']]  # data[['Defense']] = data frame
print(type(data_frame))
# 1 - Filtering Pandas data frame
x = data['age']>30     # There are 166 entries with over 30 years of age, but repetitions do occur due to the nature of the data
data[x]
# 2 - Filtering pandas with logical_and
# There are 26 entries over 30's and plays in G (Guard) position ('and' operation of '>int64'(age) with 'equal?'string/object(G as Guard)  )
data[np.logical_and(data['age']>30, data['position'] == "G" )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['age']>30) & (data['position'] == "G")]
# Stay in loop if condition( i is not equal 5) is true
i = 0
while i != 5 :
    print('i is: ',i)
    i +=1 
print(i,'is:  5')
# Stay in loop if condition( i is not equal 5) is true
lis = [11,12,13,14,15]
for i in lis:
    print('i is: ',i)
#print('')   # I didn't get the use of it?

# Enumerate index and value of list
# index : value = 0:11, 1:12, 2:13, 3:14, 4:15
for index,value in enumerate(lis):
    print(index," : ",value)
print('')   

# For dictionaries
# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

#dict_Of_Players_Real_Value
for key,value in dict_Of_Players_Real_Value.items():
    print(key," : ",value)
print('')

# For pandas we can achieve index and value
for index,value in data[['age']][0:20].iterrows():   #first 20 entries
    print(index," : ",value)