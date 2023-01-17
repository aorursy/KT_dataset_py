import numpy as np #linear algebra

import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)



import os



#read and list all files in input

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#read and list all files in output

for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#import another library (we don't need all of matplotlib) and give it a shorthand name

import matplotlib.pyplot as plt

from os import path

#import image to be able to export wordclouds

from PIL import Image

#import wordcloud library for special visualizations

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



#state the plot style to be used throughout

plt.style.use('seaborn')



#create a variable to read the dataframes we're interested in exploring

df_critic = pd.read_csv('../input/animal-crossing/critic.csv')

df_user = pd.read_csv('../input/animal-crossing/user_reviews.csv')

#this is a revised version of the new villager dataset, not the one I used in A5

#the revision was done by opening Excel and changing formatting of the birthday column

#the revision is just a change in the formatting of the villagers birthdates so I can use them to check against my zodiac sign lists

df_villagers = pd.read_csv('../input/animalcrossingrevised/villagers.csv')
#for a quick overview of data in the critic review dataset

df_critic.head(5)
#i learned about wordclouds from here: https://www.datacamp.com/community/tutorials/wordcloud-python

#start with one review

text = df_critic.text[0]



#create variable to generate a wordcloud image for that one review, with some new optional arguments

wordcloud = WordCloud(max_font_size=50, background_color="white").generate(text)



#show wordcloud image

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
#create new variable to gather data from critic reviews dataset

text = " ".join(review for review in df_critic.text)

#print number to make sure there's a bunch of words, which should mean they were all joined

print ("There are {} words total in all critic reviews.".format(len(text)))
#create stopword list

#stopwords are words we want to filter out/not use

stopwords = set(STOPWORDS)

stopwords.update(["animal", "crossing", "new", "horizons", "game", "games", "gameplay", "series", "franchise", "player", "Nintendo", "Switch"])



#generate wordcloud image

wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)



#show wordcloud image

#the matplotlib way:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
##for a quick overview of data in the user review dataset

df_user.head(5)
#create new variable to gather every review's data from user reviews dataset

text_users = " ".join(review for review in df_user.text)

#print number to make sure there's a bunch of words, which should mean they were all joined

print ("There are {} words total in all user reviews.".format(len(text_users)))
#generate wordcloud image

wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text_users)



#show wordcloud image

#the matplotlib way:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
#for a quick overview of data in the critic review dataset

df_villagers.head(5)
#this function came with the Kaggle notebook, and I thought it'd be fun to see what it does

#it basically does what I did in A5 for getting familiar with data, but in a much quicker way, albeit with a more complex function

#distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
#run the function to plot the graphs

plotPerColumnDistribution(df_villagers, 3, 3)
#define lists of dates for each zodiac sign

aries = ['3/21', '3/22', '3/23', '3/24', '3/25', '3/26', '3/27', '3/28', '3/29', '3/30', '3/31', '4/1', '4/2', '4/3', '4/4', '4/5', '4/6', '4/7', '4/8', '4/9', '4/10', '4/11', '4/12', '4/13', '4/14', '4/15', '4/16', '4/17', '4/18', '4/19']

taurus = ['4/20', '4/21', '4/22', '4/23', '4/24', '4/25', '4/26', '4/27', '4/28', '4/29', '4/30', '5/1', '5/2', '5/3', '5/4', '5/5', '5/6', '5/7', '5/8', '5/9', '5/10', '5/11', '5/12', '5/13', '5/14', '5/15', '5/16', '5/17', '5/18', '5/19', '5/20']

gemini = ['5/21', '5/22', '5/23', '5/24', '5/25', '5/26', '5/27', '5/28', '5/29', '5/30', '5/31', '6/1', '6/2', '6/3', '6/4', '6/5', '6/6', '6/7', '6/8', '6/9', '6/10', '6/11', '6/12', '6/13', '6/14', '6/15', '6/16', '6/17', '6/18', '6/19', '6/20']

cancer = ['6/21', '6/22', '6/23', '6/24', '6/25', '6/26', '6/27', '6/28', '6/29', '6/30', '7/1', '7/2', '7/3', '7/4', '7/5', '7/6', '7/7', '7/8', '7/9', '7/10', '7/11', '7/12', '7/13', '7/14', '7/15', '7/16', '7/17', '7/18', '7/19', '7/20', '7/21', '7/22']

leo = ['7/23', '7/24', '7/25', '7/26', '7/27', '7/28', '7/29', '7/30', '7/31', '8/1', '8/2', '8/3', '8/4', '8/5', '8/6', '8/7', '8/8', '8/9', '8/10', '8/11', '8/12', '8/13', '8/14', '8/15', '8/16', '8/17', '8/18', '8/19', '8/20', '8/21', '8/22']

virgo = ['8/23', '8/24', '8/25', '8/26', '8/27', '8/28', '8/29', '8/30', '8/31', '9/1', '9/2', '9/3', '9/4', '9/5', '9/6', '9/7', '9/8', '9/9', '9/10', '9/11', '9/12', '9/13', '9/14', '9/15', '9/16', '9/17', '9/18', '9/19', '9/20', '9/21', '9/22']

libra = ['9/23', '9/24', '9/25', '9/26', '9/27', '9/28', '9/29', '9/30', '10/1', '10/2', '10/3', '10/4', '10/5', '10/6', '10/7', '10/8', '10/9', '10/10', '10/11', '10/12', '10/13', '10/14', '10/15', '10/16', '10/17', '10/18', '10/19', '10/20', '10/21', '10/22']

scorpio = ['10/23', '10/24', '10/25', '10/26', '10/27', '10/28', '10/29', '10/30', '10/31', '11/1', '11/2', '11/3', '11/4', '11/5', '11/6', '11/7', '11/8', '11/9', '11/10', '11/11', '11/12', '11/13', '11/14', '11/15', '11/16', '11/17', '11/18', '11/19', '11/20', '11/21']

sagittarius = ['11/22', '11/23', '11/24', '11/25', '11/26', '11/27', '11/28', '11/29', '11/30', '12/1', '12/2', '12/3', '12/4', '12/5', '12/6', '12/7', '12/8', '12/9', '12/10', '12/11', '12/12', '12/13', '12/14', '12/15', '12/16', '12/17', '12/18', '12/19', '12/20', '12/21']

capricorn = ['12/22', '12/23', '12/24', '12/25', '12/26', '12/27', '12/28', '12/29', '12/30', '12/31', '1/1', '1/2', '1/3', '1/4', '1/5', '1/6', '1/7', '1/8', '1/9', '1/10', '1/11', '1/12', '1/13', '1/14', '1/15', '1/16', '1/17', '1/18', '1/19']

aquarius = ['1/20', '1/21', '1/22', '1/23', '1/24', '1/25', '1/26', '1/27', '1/28', '1/29', '1/30', '1/31', '2/1', '2/2', '2/3', '2/4', '2/5', '2/6', '2/7', '2/8', '2/9', '2/10', '2/11', '2/12', '2/13', '2/14', '2/15', '2/16', '2/17', '2/18']

pisces = ['2/19', '2/20', '2/21', '2/22', '2/23', '2/24', '2/25', '2/26', '2/27', '2/28', '2/29', '3/1', '3/2', '3/3', '3/4', '3/5', '3/6', '3/7', '3/8', '3/9', '3/10', '3/11', '3/12', '3/13', '3/14', '3/15', '3/16', '3/17', '3/18', '3/19', '3/20']



#create a function to sort by items that fall into the lists above

def zodiac_sign(birthday):

    #if the birthday falls into the zodiac sign's category

    if birthday in aries:

        #i'm going to categorize these as that zodiac sign

        return "Aries"

    elif birthday in taurus:

        return "Taurus"

    elif birthday in gemini:

        return "Gemini"

    elif birthday in cancer:

        return "Cancer"

    elif birthday in leo:

        return "Leo"

    elif birthday in virgo:

        return "Virgo"

    elif birthday in libra:

        return "Libra"

    elif birthday in scorpio:

        return "Scorpio"

    elif birthday in sagittarius:

        return "Sagittarius"

    elif birthday in capricorn:

        return "Capricorn"

    elif birthday in aquarius:

        return "Aquarius"

    elif birthday in pisces:

        return "Pisces"

    #otherwise, return an error message

    else:

        "Something went wrong"



#apply the zodiac sign function to the acnh villager dataset, count total occurrences, and plot as a bar graph

df_villagers['birthday'].apply(zodiac_sign).value_counts().plot(kind='bar')
#read the existing file

write_acnh = pd.read_csv("../input/animalcrossingrevised/villagers.csv")  

#run the function in a new column called "zodiac"

write_acnh["zodiac"] = df_villagers['birthday'].apply(zodiac_sign)    

#send this revised data to a csv file

write_acnh.to_csv("/kaggle/working/villagers_zodiac.csv")
#introducing... villager data for animal crossing new leaf!

df_acnl = pd.read_csv('../input/acnlvillagers/acnlvillagers.csv')
#creating a way to reference the new file I created that includes the villagers' zodiac signs post-application of zodiac sign function

df_zodiac = pd.read_csv('/kaggle/working/villagers_zodiac.csv', header=0)

#for a quick overview of data in the animal crossing: new leaf villagers dataset

df_acnl.head()
#used this (courtesy of Rafal) to figure out why I kept getting a key error

#this showed me that all my columns had an extra space at the end of the word

#so I revised the file and reuploaded it

print(df_acnl.columns)
#run the same function as done before for zodiac analysis and plot it for the new leaf villagers

df_acnl['Birthday'].apply(zodiac_sign).value_counts().plot(kind='bar')
#new dataframe for both villager datasets

#I revised these in Excel to show only the name and birthday columns, and made sure the column titles matched

df1 = pd.read_csv('../input/datamerge/acnl_merge.csv')

df2 = pd.read_csv('../input/datamerge/acnh_merge.csv')



#borrowed this function from: https://hackersandslackers.com/compare-rows-pandas-dataframes/

def dataframe_difference(df1, df2, which=None):

    #find rows which are different between two DataFrames

    comparison_df = df1.merge(df2,

                              indicator=True,

                              how='outer')

    if which is None:

        diff_df = comparison_df[comparison_df['_merge'] != 'both']

    else:

        diff_df = comparison_df[comparison_df['_merge'] == which]

    diff_df.to_csv('/kaggle/working/datadiff.csv')

    return diff_df



#run the function to see which villagers only occur in one dataset or the other

dataframe_difference(df1, df2, which=None)
#create variable to call the csv file we just created that shows the villagers that only occur in one file or the other

diff_df = pd.read_csv('/kaggle/working/datadiff.csv')

#apply the zodiac function to see what zodiac signs occur for the villagers that are unique to either game

diff_df['birthday'].apply(zodiac_sign).value_counts().plot(kind='bar')
#learned about double bar graph from here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.bar.html

#assign a variable to the different value counts I'm looking for

newleaf = df_acnl['Birthday'].apply(zodiac_sign).value_counts()

newhorizons = df_villagers['birthday'].apply(zodiac_sign).value_counts()

diffdata = diff_df['birthday'].apply(zodiac_sign).value_counts()



#give the variables some labels for the legend

df = pd.DataFrame({'Animal Crossing: New Leaf': newleaf, 'Animal Crossing: New Horizons': newhorizons, 'Removed/New': diffdata})

#show value count labels and turn 90 degrees

ax = df.plot.bar(rot=90, stacked=True)
#review counts of zodiac signs from new horizons

df_villagers['birthday'].apply(zodiac_sign).value_counts()
#review counts of zodiac signs from new leaf

df_acnl['Birthday'].apply(zodiac_sign).value_counts()
#learned about double bar graph from here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.bar.html

#assign a variable to the different value counts I'm looking for

newleaf = df_acnl['Birthday'].apply(zodiac_sign).value_counts()

newhorizons = df_villagers['birthday'].apply(zodiac_sign).value_counts()



#give the variables some labels for the legend

df = pd.DataFrame({'Animal Crossing: New Leaf': newleaf, 'Animal Crossing: New Horizons': newhorizons})

#show value count labels and turn 90 degrees

ax = df.plot.bar(rot=90)