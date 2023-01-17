#!pip install wordcloud  # install wordcloud - run this once

import pandas as pd 

from urllib.request import Request, urlopen # for opening a connection to a website

from bs4 import BeautifulSoup #web scraping 

import folium #map visualization

import html

import sys # mostly used for debugging purposes

#matplotlib for graphs

import matplotlib.pyplot as plt

from wordcloud import WordCloud

%matplotlib inline 

import seaborn as sns

import re

    
df = pd.read_csv('../input/bournemouth_venues.csv')



print("Infomation abut the dataset:\n")

df.info()



print("\nFirst few rows of the dataset:\n")

print(df.head())



df.tail()
df = df.rename(columns={

'Venue Name': 'Name',

'Venue Category': 'Category',

'Venue Latitude': 'Latitude',

'Venue Longitude': 'Longitude'})
# In excel you could use a pivot table to summarise the "Venue Name" column and find the number of occurances of each venue name. 

#Lets use something similar in pandas:



#create a pivot table to count the number of occurances in the "Name" column

df_pivot = pd.pivot_table(df, index=['Name'], aggfunc='count')



#Filter our pivot table to only show venues that occur more than once

df_pivot_filter = df_pivot[df_pivot.Category > 1]





Total_venues = len(df.Name) #total number of venues in our data

chain_venues = len(df_pivot_filter.Category) #of those venues how many of them are part of a chain?



print(f"In total we have {Total_venues} venues. Of these venues, {chain_venues} of them are part of a chain")
#First generate a wordcloud of all the Categories



#Lets grab the Category column and put it into a list.

cata_list = df["Category"].values.tolist()



#Go through the list and remove all leading and trailing white space. 

cata_list = [ele.strip(" ") for ele in cata_list]





#Go through the list and unify any double barral words

cata_list = [ele.replace(" ","") for ele in cata_list]



#finally convert the list to a long string, seperated by a space

cata_list = ' '.join(cata_list)





#******The plot*******



#create a new figure instance

fig = plt.figure(figsize=(7,7))



#create a new axes for this plot. its a single graph so subplots are not needed

ax = fig.add_axes([1,1,1,1])



#Create the wordcloud object

wordcloudobj = WordCloud(width=500, height=500,

               margin=0, max_font_size=100, min_font_size=12,

               background_color="lightyellow", collocations = False).generate(cata_list)



#plot details

ax.set_title("WordCloud of different category types")

ax.axis("off") #turn off axis



# show plot

ax.imshow(wordcloudobj, interpolation='bilinear') # Display the generated image:

#plotting a bar graph for number of occurances of category type



#create a pivot table to count the number of occurances in the "category" column

df_catapivot = pd.pivot_table(df, index=['Category'], aggfunc='count')

print(df_catapivot.head())
#we have df_catapivot as seen above we will clean this to prepare for bar plot:



#drop lat long columns:

del df_catapivot["Latitude"]

del df_catapivot["Longitude"]



#rename "Name" colum to "count" as, that is what its showing

df_catapivot = df_catapivot.rename(columns={'Name': 'Count'})



#set an integer index and keep the former index as a new column

df_catapivot = df_catapivot.reset_index(drop=False)



print(df_catapivot.head())
# And now for the bar plot



# Close previous figure:

plt.close(fig)



#create a new figure instance

fig = plt.figure(figsize=(7,7))



#create a new axes for this plot. 

ax = fig.add_axes([1,1,1,1])



#plot the bar plot using seaborn

sns.barplot(df_catapivot["Category"],df_catapivot["Count"])
#filter the dataframe to remove all X values with count = 1

df_catapivot_filt = df_catapivot[df_catapivot.Count > 1]





# And finally plotting the bar plot with filtered data



# Close previous figure:

plt.close(fig)



#create a new figure instance

fig = plt.figure(figsize=(7,7))



#create a new axes for this plot. 

ax = fig.add_axes([1,1,1,1])



#details



#set title

ax.set_title("Count of the different venue types in Bournemouth")

#rotate x-axis labels

for label in ax.xaxis.get_ticklabels():

    label.set_rotation(90)





#plot the bar plot using seaborn

sns.barplot(df_catapivot_filt["Category"],df_catapivot_filt["Count"])


    

#First we need to center out map on Bournemouth. A simple google search reveals:

#"The latitude of Bournemouth, UK is 50.720806, and the longitude is -1.904755."



Bournemouth_coord = (50.720806, -1.904755)



# create empty map zoomed in on Bournemouth

bournemap = folium.Map(location=Bournemouth_coord, zoom_start=12)



rownum = len(df["Name"])

i = 0



for i in range(0,rownum):    



    name = df.loc[i,"Name"]

    cate = df.loc[i,"Category"]

    text = str(name + " (" + cate + ")")

    folium.CircleMarker(

        [df.loc[i,"Latitude"], df.loc[i,"Longitude"]],

        radius=8,

        popup = html.escape(text),

        color='blue',

        fill_color='green',

        fill=True,

        fill_opacity=0.7,

        clustered_marker = True

        ).add_to(bournemap)

    

    i += 1





#display our map

print("Map of Bournemouth with venue locations")

display(bournemap)











print("The pivot table - showing venues that are part of a chain of venues")

print(df_pivot_filter)
print("Resetting the Index whilst keeping the original index but as a new column")

df_pivot_filter = df_pivot_filter.reset_index(drop=False)

print(df_pivot_filter)
#Now Make a copy of our df:

dfbs = df



#Add a new column "rating"

dfbs["Review"] = ""



#Now loop through "df_pivot_filter" and drop all rows containing the same venues 

row = 0

for chain in df_pivot_filter["Name"]:

    dfbs = dfbs[dfbs.Name != chain] #remove the rows contianing chain (which are the chain venues)

    row +=1

    

#we have removed rows. we need to reset the index

dfbs = dfbs.reset_index(drop=True)



#upon testing the code it was noticed that certain strings (Lola’s) threw errors.

#Need to loop through and remove all instances of "’"









i = 0



for i in range (len(dfbs["Name"])):

    

    venue = dfbs.loc[i,"Name"]

    

    #check if the string venue has "’"  and if so replace with"'"

    venue = venue.replace("’","'")



    #we wil effectivly do a google search. so we will constuct this here:

    #we search for: "UK Bournemouth venueName" - this should give us a side box with venue details in google

    location = "UK Bournemouth " + venue

    

    #constructing the weblink

    web = "https://www.google.com/search?safe=active&source=hp&ei=bo4vXfiVIY-yUtPGomg&q="

    web_full = web + location.replace(" ","+")

    

    #sending a request using urllib with added headers to ensure we get a connection

    #the header just tells the wepage that we are accessing it using mozilla verion 5.0

    req = Request(web_full, headers={'User-Agent': 'Mozilla/5.0'})

    

    

    

    

    

    #open the webpage and read

    page = urlopen(req).read()

    

    #using beautiful soup turn the html of the page into a soup format 

    soup = BeautifulSoup(page, 'html.parser')

    

    

    

    #Once we grab the review data, we check it its a value or "none"

    #if it is "none", we get an attribute error which we will except and go on to store "nan"

    try:

        data1 = soup.find('span', attrs={'class': 'oqSTJd'})

        score = data1.text.strip() # strip() is used to remove starting and trailing

        

        dfbs.loc[i,"Review"] = score  #store the score into dfbs

        

    except AttributeError:

        dfbs.loc[i,"Review"] = 0.0 # since no score was available, set score to 0

        

    i +=1



    

#show the first fiew rows of our dataframe

print(dfbs.head())















#we know we have some points that have this sort of shape: 8.3/10

#we need to take the 8.3 and divde by 10 then * by 5 to get a out of 5 score. then store this as a float:

#We can use regex to extract everything behind the /10

#some regex to ensure we grab the data in the correct format:

pullrule = '([0-9]+.[0-9]+)' # REGEX RULE: example: 4.8

checkrule = '([0-9]+.[0-9]+/)' # REGEX RULE: example: 4.8/

i = 0



#loop through the Review data

for i in range(len(dfbs["Review"])):

    

    data = str(dfbs.loc[i,"Review"]) #convert data into string so we can use REGEX on it

    is_check = re.match(checkrule, data) #check if "/" exists in data - returns boolean

    

    #check if "/" exists in string:

    if is_check:

        # use pullrule to extract everything behind "/" eg 4.8/10 -> 4.8

        score_fmt = re.findall(pullrule, data)#find the REGEX match, store into score_fmt

        x  = float(score_fmt[0]) / 2

        

        dfbs.loc[i,"Review"] = x #take first element of list + convert to float' and store back into df

        i += 1

    else:     

        dfbs.loc[i,"Review"] = float(data) #convert back to float

        i += 1

    



dfbs.head()