# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing the matplotlib library

import matplotlib.pyplot as plt



#Read in the JSON file of Amazon reviews for Tools and Home Improvement products

#Setting the file equal to 'tools'

tools = pd.read_json('../input/mp2-amazon/Tools_and_Home_Improvement_5.json', lines=True)
#Previewing the tools data

tools
#Show the first 10 rows of tools.  This is really just a warm-up activity for me.

tools.head(10)
#Determining the number of rows in the tools dataset

print("There are " + str(len(tools)) + " reviews in the tools dataset.")
#Printing the list of column headers

print(tools.columns.values)
#Show the values in the reviewText column

reviews = tools[tools.columns[4]]

reviews
#Calculating the mean of all overall ratings

overall_mean = tools.overall.mean()

print("The mean overall score is " + str(overall_mean) + " out of 5")
#Create a histogram for the 5-point ratings for all tools and home improvement products

tools.hist(['overall'])



#Defining the x-axis range

plt.xticks(np.arange(1, 6, 1))



#Configuring chart title and axis labels

plt.title("Overall Ratings for Tools and Home Improvement Products")

plt.xlabel("Rating")

plt.ylabel("Frequency")
#Used value_counts() to pull the distribution of rating scale responses and counts

ratingCounts = tools['overall'].value_counts()



print(ratingCounts)
#Sorting all of the data by the ratingCounts values with highest being on top, lowest on bottom

ratingCounts = ratingCounts.sort_values() 



#Creating a bar chart of the distribution of ratings

ratingCounts.plot.barh()



#Adding chart title and axis labels

plt.title('Distribution of ratings')

plt.xlabel('Frequency')

plt.ylabel('Rating')
#Pulling the review for one specific individual

brennan = tools.loc[tools['reviewerName'] == 'D. Brennan', 'reviewText'].iloc[0]

brennan
#Determining the rating scale response for each row in the list with 1048000001X ASIN.

asin_example = tools.loc[tools['asin'] == '104800001X', 'overall']

asin_example
#Used value_counts() to pull the distribution of rating scale responses and counts for a specific ASIN

asinRating = tools.loc[tools['asin'] == '104800001X', 'overall'].value_counts()



print(asinRating)
#Create a histogram for the 5-point ratings for all tools and home improvement products

asin_example.hist()



#Defining the x-axis range and number of ticks

plt.xticks(np.arange(1, 6, 1))



#Configuring chart title and labels

plt.title("Overall Ratings for ASIN 104800001X")

plt.xlabel("Rating")

plt.ylabel("Frequency")
#Importing the TextBlob library for processing textual data

from textblob import TextBlob



#Creating a text blob

blob = TextBlob(brennan)

#Print the brennan review

blob_brennan = TextBlob(brennan)

print(TextBlob(brennan))



#Applying a conditional to determine if polarity rating is positive, negative, or neutral

#Values range from -1.0 (negative) => +1.0 (positive)

if blob_brennan.sentiment.polarity > 0:

    pol = 'positive'

elif pblob_brennan.sentiment.polarity < 0:

    pol = 'negative'

elif blob_brennan.sentiment.polarity == 0:

    pol = 'neutral'



#Calculate and print polarity

print("Polarity: " + str(blob_brennan.sentiment.polarity) + " (" + pol + ")")



    

#Applying a conditional to determine if subjectivity rating is subjective or objective

#Values range from +0.0 (objective) => +1.0 (subjective)

if blob_brennan.sentiment.subjectivity > 0.5:

    sub = 'subjective'

elif blob_brennan.sentiment.subjectivity < 0.5:

    sub = 'objective'

elif blob_brennan.sentiment.subjectivity == 0.5:

    sub = 'between subjective and objective'



#Calculate and print subjectivity

print("Subjectivity: " + str(blob_brennan.sentiment.subjectivity) + " (" + sub + ")")
#Breaking the review into individual sentences

brennan_sentences = blob.sentences
#Iterate through each sentence in brennan review

for sentence in brennan_sentences:

    

    #Printing each sentence

    print(sentence)
#Iterating through each sentence of Brennan's review

for sentence in brennan_sentences:

    #Printing each sentence

    print(sentence)

    

    #Determining polarity for each sentence: negative vs. positive 

    #Values range from -1.0 (negative) => +1.0 (positive)

    polarity = sentence.sentiment.polarity

    

    #Applying a conditional to determine if polarity rating is positive, negative, or neutral

    if polarity > 0:

        pol = 'positive'

    elif polarity < 0:

        pol = 'negative'

    elif polarity == 0:

        pol = 'neutral'

    

    #Print polarity rating

    print('Polarity: ' + str(polarity) + ' (' + pol + ')')

    

    #Determining subjectivity for each sentence: objective vs. subjective

    #Values range from +0.0 (objective) => +1.0 (subjective)

    subjectivity = sentence.sentiment.subjectivity

    

    #Applying a conditional to determine if subjectivity rating is subjective or objective

    if subjectivity > 0.5:

        sub = 'subjective'

    elif subjectivity < 0.5:

        sub = 'objective'

    elif subjectivity == 0.5:

        sub = 'between subjective and objective'

    

    #Print subjectivity rating

    print('Subjectivity: ' + str(subjectivity) + ' (' + sub + ')')

    

    #Print empty line between sentences

    print()
#Pull 100 rows from the data.  Specifically pulling the reviewText and overall columns.

reviews = tools[['overall', 'reviewText']].sample(100)

reviews
#Creating a function to calculate polarity.

def calc_polarity(text):

    return TextBlob(text).sentiment.polarity



#Creating a function to calculate subjectivity.

def calc_subjectivity(text):

    return TextBlob(text).sentiment.subjectivity





#Applying the polarity calculation to the reviewText column in the reviews data

reviews['polarity'] = reviews.reviewText.apply(calc_polarity)



#Applying the subjectivity  calculation to the reviewText column in the reviews data

reviews['subjectivity'] = reviews.reviewText.apply(calc_subjectivity)



#Previewing the data

reviews.head()
#Applying the polarity calculation to the reviewText column in the entire tools data.  Adding a new polarity column.

tools['polarity'] = tools.reviewText.apply(calc_polarity)



#Applying the subjectivity calculation to the reviewText column in the entire tools data.  adding a new subjectivity column.

tools['subjectivity'] = tools.reviewText.apply(calc_subjectivity)



tools
#Calculating the highest value of all polarity scores in the Tools dataset

highest_polarity = tools.polarity.max()

print("Highest Tools and Home Improvement Products Polarity Score: " + str(highest_polarity))



#Calculating the lowest value of all polarity scores in the Tools dataset

lowest_polarity = tools.polarity.min()

print("Lowest Tools and Home Improvement Products Polarity Score: " + str(lowest_polarity))
#Pulling the list of reviews with the highest possible polarity score of 1.0

highest_polarity_reviews = tools.loc[tools['polarity'] == 1, 'reviewText']

highest_polarity_reviews
#Calculating the total number of reviews with the highest possible 1.0 polarity score

print("There are " + str(len(highest_polarity_reviews)) + " reviews with a maximum +1.0 polarity score")

print()

print("A few examples:")



#Pulling the review in the 1st three rows of highest polarity reviews

highest_polarity_example = tools.loc[tools['polarity'] == 1, 'reviewText'].iloc[0]

highest_polarity_example2 = tools.loc[tools['polarity'] == 1, 'reviewText'].iloc[1]

highest_polarity_example3 = tools.loc[tools['polarity'] == 1, 'reviewText'].iloc[2]

print('"' + highest_polarity_example + '"')

print()

print('"' + highest_polarity_example2 + '"')

print()

print('"' + highest_polarity_example3 + '"')
#Pulling the list of reviews with the lowest possible polarity score of -1.0

lowest_polarity_reviews = tools.loc[tools['polarity'] == -1, 'reviewText']

lowest_polarity_reviews
#Calculating the total number of reviews with the lowest possible -1.0 polarity score

print("There are " + str(len(lowest_polarity_reviews)) + " reviews with a -1.0 polarity score")

print()

print("A few examples:")



#Pulling the review in the 1st three rows of lowest polarity reviews

lowest_polarity_example = tools.loc[tools['polarity'] == -1, 'reviewText'].iloc[0]

lowest_polarity_example2 = tools.loc[tools['polarity'] == -1, 'reviewText'].iloc[1]

lowest_polarity_example3 = tools.loc[tools['polarity'] == -1, 'reviewText'].iloc[2]

print('"' + lowest_polarity_example + '""')

print()

print('"' + lowest_polarity_example2 + '""')

print()

print('"' + lowest_polarity_example3 + '""')

#Calculating the highest value of all subjectivity scores in the Tools dataset

highest_subjectivity = tools.subjectivity.max()

print("Highest Tools and Home Improvement Products Subjectivity Score: " + str(highest_subjectivity))



#Calculating the lowest value of all subjectivity scores in the Tools dataset

lowest_subjectivity = tools.subjectivity.min()

print("Lowest Tools and Home Improvement Products Subjectivity Score: " + str(lowest_subjectivity))
#Pulling the list of reviews with the highest possible subjectivity score of 1.0

highest_subjectivity_reviews = tools.loc[tools['subjectivity'] == 1, 'reviewText']

highest_subjectivity_reviews
#Calculating the total number of reviews with the highest possible subjectivity score of 1.0

print("There are " + str(len(highest_subjectivity_reviews)) + " reviews with a maximum +1.0 subjectivity score")

print()

print("A few examples:")



#Pulling the review in the 1st three rows of highest subjectivity reviews

highest_subjectivity_example = tools.loc[tools['subjectivity'] == 1, 'reviewText'].iloc[0]

highest_subjectivity_example2 = tools.loc[tools['subjectivity'] == 1, 'reviewText'].iloc[1]

highest_subjectivity_example3 = tools.loc[tools['subjectivity'] == 1, 'reviewText'].iloc[2]

print('"' + highest_subjectivity_example + '"')

print()

print('"' + highest_subjectivity_example2 + '"')

print()

print('"' + highest_subjectivity_example3 + '"')
#Pulling the list of reviews with the lowest possible subjectivity score of 0

lowest_subjectivity_reviews = tools.loc[tools['subjectivity'] == 0, 'reviewText']

lowest_subjectivity_reviews
#Calculating the total number of reviews with the lowest possible subjectivity score of 0

print("There are " + str(len(lowest_subjectivity_reviews)) + " reviews with a 0 subjectivity score")

print()

print("An example:")



#Pulling the review in the 1st three rows of lowest subjectivity reviews

lowest_subjectivity_example = tools.loc[tools['subjectivity'] == 0, 'reviewText'].iloc[0]

lowest_subjectivity_example2 = tools.loc[tools['subjectivity'] == 0, 'reviewText'].iloc[1]

lowest_subjectivity_example3 = tools.loc[tools['subjectivity'] == 0, 'reviewText'].iloc[2]

print('"' + lowest_subjectivity_example + '"')

print()

print('"' + lowest_subjectivity_example2 + '"')

print()

print('"' + lowest_subjectivity_example3 + '"')
mean_polarity = tools.reviewText.apply(calc_polarity).mean()



#Applying a conditional to determine if polarity rating is positive, negative, or neutral

if mean_polarity > 0:

    pol = 'positive'

elif mean_polarity < 0:

    pol = 'negative'

elif mean_polarity == 0:

    pol = 'neutral'



#Print polarity rating

print('Mean Tools and Home Improvement Products Polarity: ' + str(mean_polarity) + ' (' + pol + ')')



#Values range from +0.0 (objective) => +1.0 (subjective)

mean_subjectivity = tools.reviewText.apply(calc_subjectivity).mean()



#Applying a conditional to determine if subjectivity rating is subjective or objective

if mean_subjectivity > 0.5:

    sub = 'subjective'

elif mean_subjectivity < 0.5:

    sub = 'objective'

elif mean_subjectivity == 0.5:

    sub = 'between subjective and objective'



#Print subjectivity rating

print('Mean Tools and Home Improvement Products Subjectivity: ' + str(mean_subjectivity) + ' (' + sub + ')')
#Create a histogram for the sentiment polarity for all tools and home improvement products

tools_pol = tools.reviewText.apply(calc_polarity)

tools_pol.hist()



#Defining the x-axis range and number of ticks

plt.xticks(np.arange(-1, 1.25, .25))



#Configuring chart title and labels

plt.title("Distribution of Polarity for All Tools and Home Improvement Products")

plt.xlabel("Polarity")

plt.ylabel("Frequency")
#Create a histogram for the sentiment subjectivity for all tools and home improvement products

tools_sub = tools.reviewText.apply(calc_subjectivity)

tools_sub.hist()



#Defining the x-axis range and number of ticks

plt.xticks(np.arange(0, 1.25, .25))



#Configuring chart title and labels

plt.title("Distribution of Subjectivity for All Tools and Home Improvement Products")

plt.xlabel("Subjectivity")

plt.ylabel("Frequency")
#Part-of-speech tags for the brennan review

for sentence in brennan_sentences:

    #Printing each sentence

    print(sentence)

    

    #Provides the part-of-speech for each word within that sentence

    s_tags = sentence.tags

    print(s_tags)

    print()
s = sentence.tags



s = str(s).replace('CC', 'conjunction, coordinating')

s = str(s).replace('CD', 'cardinal number')

s = str(s).replace('DT', 'determiner')

s = str(s).replace('EX', 'existential there')

s = str(s).replace('FW', 'foreign word')

s = str(s).replace('IN', 'conjunction, subordinating or preposition')

s = str(s).replace('JJR', 'adjective, comparative')

s = str(s).replace('JJS', 'adjective, superlative')

s = str(s).replace('JJ', 'adjective')

s = str(s).replace('LS', 'list item marker')

s = str(s).replace('MD', 'verb, modal auxillary')

s = str(s).replace('NNS', 'noun, plural')

s = str(s).replace('NNP', 'noun, proper singular')

s = str(s).replace('NNPS', 'noun, proper plural')

s = str(s).replace('NN', 'noun, singular or mass')

s = str(s).replace('PDT', 'predeterminer')

s = str(s).replace('POS', 'possessive ending')

s = str(s).replace('PRP$', 'pronoun, possessive')

s = str(s).replace('PRP', 'pronoun, personal')

s = str(s).replace('RBR', 'adverb, comparative')

s = str(s).replace('RBS', 'adverb, superlative')

s = str(s).replace('RB', 'adverb')

s = str(s).replace('RP', 'adverb, particle')

s = str(s).replace('SYM', 'symbol')

s = str(s).replace('TO', 'infinitival to')

s = str(s).replace('UH', 'interjection')

s = str(s).replace('VBZ', 'verb, 3rd person singular present')

s = str(s).replace('VBP', 'verb, non-3rd person singular present')

s = str(s).replace('VBD', 'verb, past tense')

s = str(s).replace('VBN', 'verb, past participle')

s = str(s).replace('VBG', 'verb, gerund or present participle')

s = str(s).replace('VB', 'verb, base form')

s = str(s).replace('WDT', 'wh-determiner')

s = str(s).replace('WP$', 'wh-pronoun, possessive')

s = str(s).replace('WP', 'wh-pronoun, personal')

s = str(s).replace('WRB', 'wh-adverb')

s

        





        

        