import numpy as np

import pandas as pd

import re

import csv

import sys

from matplotlib import pyplot as plt

%matplotlib inline

from wordcloud import WordCloud

import seaborn as sns
#df = pd.read_csv("data_cafe.csv") <-- reading directly into a dataframe gives errors, bad formatting?

#open('data_cafe.csv') # Opening this gives codec errors. needs utf-8 encoding 

#we will use csv.reader:



data = []

with open('../input/data_cafe.csv', encoding="utf8") as File:

    reader = csv.reader(File, delimiter=';', quotechar=',', quoting=csv.QUOTE_MINIMAL)

    for row in reader:

        data.append(row)



#Example of our data::

print(data[0]) # column headers

print(data[1]) # review 1
#First row is a list of columns lets copy this and remove it from the data



cols = data[0] # store column headers 

del data[0] #remove column headers from datalist

#Check how many rows are abnormal:

incorrect_rows = 0

for row in data:

    if len(row) != 6:

        incorrect_rows +=1

        

print(f"Number of abnormal rows: {incorrect_rows}")

#Lets catch these rows and merge the comments columns to form a single element

for row in data:

    if len(row) != 6:

        rows = len(row) # number of rows it has.

        row[5:rows+1] = [','.join(row[5:rows+1])]

        

#Check how many rows are abnormal after fix:

incorrect_rows = 0

for row in data:

    if len(row) != 6:

        incorrect_rows +=1

print(f"Number of abnormal rows after fix: {incorrect_rows}")
#cols = data[0] # store column headers             #for reference - did this already above

#del data[0] #remove column headers from datalist  #for reference - did this already above



df = pd.DataFrame(data,columns= cols) # create new dataframe

df.head()
#An overview of the dataframe:

df.info()

df.describe()


#noticed that some "type" columns contain empty "  " cells. lets find these and strip them of whitespace to get "".

#Go through each cafe type and strip all trailing/leading white space. This will help us later.

for i in range(len(df["Type"])):

    df.loc[i,"Type"] = str(df.loc[i,"Type"]).strip()
#Pull a list of unique values

ListOfLocations = df.Location.unique()

count = [] #list of corresponding count of cafes



for city in ListOfLocations:

    dffilt = df[df.Location == city]

    count.append(int(len(dffilt.Location)))









    

#Plot a bar graph to show number of cafes per location using [ListOfLocations] and [count]

#sns styles - darkgrid/whitegrid/dark/white

sns.set_style("whitegrid")

fig = plt.figure(figsize=(9,7))

ax = fig.add_axes([1,1,1,1])

#ax.bar(ListOfLocations,count,width=1,align='center')

ax.set_xlabel("Cities", weight="bold", fontsize=14)

ax.set_ylabel("Number of cafes", weight="bold", fontsize=14)

ax.set_title("Number of cafes in each city", weight="bold", fontsize=16)



sns.barplot(ListOfLocations,count,ax=ax)



def locFilteredDf(location): # output a list of cafe types in the city of "location" - and ensure datapoints are compatible for wordcloud vis

    row_droplist = [] # a list of rows to drop as they have blank "Type" elements

    df_loc_cleaned = df # this will be a new df but with empty " " elements removed and words formatted correctly for the plot

    for i in range(0,len(df["Type"])):

 

        element = df.loc[i,"Type"]#store datapoint in "element" so we can clean it up and put it back into original df (1 means look at "type" col)

        element = element.strip() #remove leading/tailing whitespace

        element = element.replace(" ","")#remove whitespace inbtween words

        

        if element == "": # checking if its a blank element

            row_droplist.append(i) # collect index values for data points with empty " " values, to be dropped later 

        else:

            df_loc_cleaned.loc[i,"Type"] = element # store cleaned datapoints into the new dataframe. lets not touch the original

    

    df_loc_cleaned = df_loc_cleaned.drop(row_droplist)



    #Now we have all datapoints ready that are associated with given -> location. lets filter 

    df_loc_cleaned = df_loc_cleaned[df_loc_cleaned.Location == location]

    return df_loc_cleaned #returns a filtered dataframe specific to a location







fig = plt.figure(figsize=(15,15))

i = 1

for location in ListOfLocations:

    #print(i) # a counter to show the plots being drawn. at first i thought it was an infinate loop but turns out its a super long loop...

    Vistext = locFilteredDf(location)['Type'].values.tolist() # This is a list to drop. convert it into a long string with space seperation

    Vistext_str = ' '.join(Vistext)



    # Create the wordcloud object

    wordcloudobj = WordCloud(width=600, height=600, margin=0, max_font_size=100, min_font_size=12,

                       background_color="lightyellow", collocations = False).generate(Vistext_str)  

         

    ax = fig.add_subplot(3,4,i)

    ax.set_title(f"Types of cafes in {location}\n ({len(Vistext)} cafes)")



    ax.axis("off")

    ax.imshow(wordcloudobj, interpolation='bilinear') # Display the generated image:

    i += 1

    



        
#A function that:

#  - takes a list of cities, and for each city:

#     - sorts the cafe types to pull the top 5 popular ones

#     - makes a count of how many of each type of cafe

#     - plots a graph of top cafe type and score





def city2avgScore(locations):

    num_of_cities = len(locations)

    plotcountincrement = 1

    # new df filtering only data relating to city of choice

    for city in locations:

        dffilt_city = df[df["Location"] == city] 





        # pivot table to summarise data and get a count of number of

        #occurances of cafe type - store as a new dataframe

        df_toptypes = dffilt_city.pivot_table(index=['Type'], aggfunc='count') 

        # we have number of occurances, now sort. this is a pivot

        #table to all columns show count 

        df_toptypes = df_toptypes.sort_values(by="Score",ascending=False)

        #note that "df_toptypes" has an index = "type". 

        #lets reset the index to have integer index for ease



        # reset index to have ingeger index and 

        #keep the "type" column (which used to be the index)

        df_toptypes = df_toptypes.reset_index(drop=False) 

        # drop all empty types

        df_toptypes = df_toptypes.drop(df_toptypes[df_toptypes.Type == ""].index) 

        # reset index again as we just dropped a row

        df_toptypes = df_toptypes.reset_index(drop=False) 



        # grab the top 5 cafe types and store in a list.

        #note first row is a blank

        top5list = df_toptypes.loc[0:4,"Type"].tolist() 



        # dataframe containing values in column "Types" 

        #that match those in the list:"top5list" from our chosen city

        dffilt_city = dffilt_city[dffilt_city["Type"].isin(top5list)]



        #now dffilt_city["Type"] and dffilt_city["Score"] can be used to plot a boxplot. 

        #note the score is normalized to 1. so multiply all score values

        #by 5 and reset index so we can work with it. 

        

        #reset the index to ensure incremental index as we will loop through it now

        dffilt_city = dffilt_city.reset_index(drop=True) 

        for i in range(0,len(dffilt_city["Score"])): # go through each row in the the dataframe 

            dffilt_city.loc[i,"Score"] = float(dffilt_city.loc[i,"Score"])*5 # replace score with (score*5)





        dffilt_city['Score'] = dffilt_city['Score'].astype(float)

        dffilt_city['Type']  = dffilt_city['Type'].astype(str)





        sns.set_style("whitegrid")

        ax = fig.add_subplot(4,1,plotcountincrement)

        #adjust subplot spacing

        fig.subplots_adjust(hspace=0.5)

        

        sns.stripplot(x=dffilt_city["Type"], y=dffilt_city["Score"], jitter=0.2, size=4)

        ax.set_xlabel("Cafe Type", weight="bold", fontsize=10)

        ax.set_ylabel("Score", weight="bold", fontsize=10)

        ax.set_title(f"Score distribution in {city}", weight="bold", fontsize=16)

        ax.set_ylim(0, 6)



        plotcountincrement += 1



# from the previous graph we find these cities with the most cafes         

x =["Koln","Nurnberg"]  

plt.close(fig)#close previous plot

fig = plt.figure(figsize=(9,17))

city2avgScore(x)
