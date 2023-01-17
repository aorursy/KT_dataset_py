# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Let's import the data into this Kaggle workspace first.
InstrumentData = "/kaggle/input/Musical_instruments_reviews.csv"
InstrumentDF = pd.read_csv("../input/amazon-music-reviews/Musical_instruments_reviews.csv")
InstrumentDF.head(20)
InstrumentDF1Ratings = pd.DataFrame(columns = ['reviewerID', 'asin', 'reviewerName', 'helpful', 'reviewText', 'overall', 'summary', 'unixReviewTime', 'reviewTime']) 
InstrumentDF2Ratings = pd.DataFrame(columns = ['reviewerID', 'asin', 'reviewerName', 'helpful', 'reviewText', 'overall', 'summary', 'unixReviewTime', 'reviewTime']) 
InstrumentDF3Ratings = pd.DataFrame(columns = ['reviewerID', 'asin', 'reviewerName', 'helpful', 'reviewText', 'overall', 'summary', 'unixReviewTime', 'reviewTime']) 
InstrumentDF4Ratings = pd.DataFrame(columns = ['reviewerID', 'asin', 'reviewerName', 'helpful', 'reviewText', 'overall', 'summary', 'unixReviewTime', 'reviewTime']) 
InstrumentDF5Ratings = pd.DataFrame(columns = ['reviewerID', 'asin', 'reviewerName', 'helpful', 'reviewText', 'overall', 'summary', 'unixReviewTime', 'reviewTime']) 

def instrumentRatingSplitter(InstrumentDF):

    for i in range(0,1428):

    #If the cell value matches the specified rating, then append the 
    #entire row to the correct rating-associated Dataframe: 
    
            if InstrumentDF.at[i, 'overall'] == 1.0:
                InstrumentDF1Ratings.loc[i] = InstrumentDF.loc[i]

            elif InstrumentDF.at[i, 'overall'] == 2.0:
                InstrumentDF2Ratings.loc[i] = InstrumentDF.loc[i]
    
            elif InstrumentDF.at[i, 'overall'] == 3.0:  
                InstrumentDF3Ratings.loc[i] = InstrumentDF.loc[i]
    
            elif InstrumentDF.at[i, 'overall'] == 4.0:
                InstrumentDF4Ratings.loc[i] = InstrumentDF.loc[i]
    
            elif InstrumentDF.at[i, 'overall'] == 5.0:
                InstrumentDF5Ratings.loc[i] = InstrumentDF.loc[i]
    
    return (InstrumentDF1Ratings, InstrumentDF2Ratings, InstrumentDF3Ratings, InstrumentDF4Ratings, InstrumentDF5Ratings)
    
#Let's call that function and take a look at one of the outputs: 
instrumentRatingSplitter(InstrumentDF)
InstrumentDF2Ratings.head(10)
InstrumentDF1RatingsWordFreqs = pd.DataFrame(columns = ['Words', 'FrequencyUtilized'])
InstrumentDF2RatingsWordFreqs = pd.DataFrame(columns = ['Words', 'FrequencyUtilized'])
InstrumentDF3RatingsWordFreqs = pd.DataFrame(columns = ['Words', 'FrequencyUtilized'])
InstrumentDF4RatingsWordFreqs = pd.DataFrame(columns = ['Words', 'FrequencyUtilized'])
InstrumentDF5RatingsWordFreqs = pd.DataFrame(columns = ['Words', 'FrequencyUtilized'])

#I'm using the list of the 100 most common words in the English language cotained in this Wikipedia article: https://en.wikipedia.org/wiki/Most_common_words_in_English to determine which words to exclude in the frequency analysis.
#The aim of this approach is to try and retain only meaningful words in the analysis. 
notAcceptedWordList = ['the', 'be', 'to', 'of', 'a', 'in', 'that', 'have', 'I', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just','him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'want', 'because', 'any','these', 'give', 'day', 'most', 'us', 'and', 'this', 'is', 'It', 'it', 'are', 'was', 'more', 'few', 'guitar','The','very', 'at', 'This', 'sound']
                       
def wordFrequencyAnalyzer(reviewsDataFrame, notAcceptedWordList, wordFrequencyDataFrame): 
    
    for reviewRow in reviewsDataFrame.itertuples():
        
        reviewText = reviewRow.reviewText.split()
        
        for word in reviewText: 
            
                if word not in notAcceptedWordList and word not in wordFrequencyDataFrame['Words']: 
                    wordFrequencyDataFrame.loc[word] = [word, 1]
                    
                elif word not in notAcceptedWordList and word in wordFrequencyDataFrame['Words']: 
                    wordFrequencyDataFrame['FrequencyUtilized'].loc[wordFrequencyDataFrame['Words'] == word] = wordFrequencyDataFrame['FrequencyUtilized'] + 1
                    
    return wordFrequencyDataFrame
#First, let's make a list containing the five output DF's from the 
#"instrumentRatingSplitter" function. Let's also create a List of Dataframes with ALL of the approved words
#and their frequencies that have been separated into rating-specific categories by the first function.
#Let's do this using the "wordFrequencyAnalyzer" function:

ratingSpecificDFList = [InstrumentDF1Ratings, InstrumentDF2Ratings, InstrumentDF3Ratings, InstrumentDF4Ratings, InstrumentDF5Ratings]
ratingSpecificWordFreqsDFList = [InstrumentDF1RatingsWordFreqs, InstrumentDF2RatingsWordFreqs, InstrumentDF3RatingsWordFreqs, InstrumentDF4RatingsWordFreqs, InstrumentDF5RatingsWordFreqs]

for int in range(5):
    ratingSpecificWordFreqsDFList[int] = wordFrequencyAnalyzer(ratingSpecificDFList[int], notAcceptedWordList, ratingSpecificWordFreqsDFList[int])
    
def topFiveReviewWordsAnalyzer (originalRatingSpecWordFreqsDF, topFiveRatingSpecWordFreqsDF):
    
    for int in range(5):
        
        currMaxWordFreq = 0
            
        currMaxWordFreqRowIndex = 0
        
        #In the inner loop, iterate over all of the tuple representations of the Original rating-specific Dataframe rows. Determine
        #which word frequency value is the highest over each series of iterations over the Dataframe. At the end of each inner loop iteration,
        #set the maximum frequency value to "currMaxWordFreq", store the index of the word with the current maximum frequency, 
        #and store the word with the current maximum frequency with the "currMaxWord" reference. Then,
        #store the most-used word and its frequency in the "topFiveDF" Dataframe. Finally, remove the current Maximally-used word
        #from the original rating-specific instrument reviews Dataframe. 
        #Repeat this entire process five times with the outer loop.
       
        for rows in originalRatingSpecWordFreqsDF.itertuples():
                
            if rows.FrequencyUtilized > currMaxWordFreq:
                    
                    currMaxWordFreq = rows.FrequencyUtilized
            
                    currMaxFreqWord = rows.Words       
    
        topFiveRatingSpecWordFreqsDF.loc[int] = [currMaxFreqWord, currMaxWordFreq]
    
        originalRatingSpecWordFreqsDF = originalRatingSpecWordFreqsDF[originalRatingSpecWordFreqsDF.Words != currMaxFreqWord]
    
    return topFiveRatingSpecWordFreqsDF
    

topFiveWordsByRatingsDFList = ['empty'] * 5

for int in range(5):
    
    topFiveWordsByRatingsDFList[int] = pd.DataFrame(columns = ['Words','FrequencyUtilized'])
    
for int in range(5):
    
    topFiveWordsByRatingsDFList[int] = topFiveReviewWordsAnalyzer(ratingSpecificWordFreqsDFList[int], topFiveWordsByRatingsDFList[int])
    
print(topFiveWordsByRatingsDFList[0].to_string())
#Let's visualize the five-most common words in one-star reviews first. Remember, this data is contained at the index of "0" in the "topFiveWordsByRatingsDFList" list: 
import matplotlib.pyplot as plt

plt.bar(topFiveWordsByRatingsDFList[0]['Words'], topFiveWordsByRatingsDFList[0]['FrequencyUtilized'])

plt.xticks(topFiveWordsByRatingsDFList[0]['Words'])

plt.title('Top 5 Most Frequently Used Words in 1-Star Musical Instrument Amazon Reviews')

plt.xlabel('Top 5 Words')

plt.ylabel('Word Frequencies')
#Let's visualize the five-most common words in two-star reviews next:
import matplotlib.pyplot as plt

plt.bar(topFiveWordsByRatingsDFList[1]['Words'], topFiveWordsByRatingsDFList[1]['FrequencyUtilized'])

plt.xticks(topFiveWordsByRatingsDFList[1]['Words'])

plt.title('Top 5 Most Frequently Used Words in 2-Star Musical Instrument Amazon Reviews')

plt.xlabel('Top 5 Words')

plt.ylabel('Word Frequencies')
#Let's visualize the five-most common words in three-star reviews first:
import matplotlib.pyplot as plt

plt.bar(topFiveWordsByRatingsDFList[2]['Words'], topFiveWordsByRatingsDFList[2]['FrequencyUtilized'])

plt.xticks(topFiveWordsByRatingsDFList[2]['Words'])

plt.title('Top 5 Most Frequently Used Words in 1-Star Musical Instrument Amazon Reviews')

plt.xlabel('Top 5 Words')

plt.ylabel('Word Frequencies')
#Let's visualize the five-most common words in four-star reviews next: 
import matplotlib.pyplot as plt

plt.bar(topFiveWordsByRatingsDFList[3]['Words'], topFiveWordsByRatingsDFList[3]['FrequencyUtilized'])

plt.xticks(topFiveWordsByRatingsDFList[3]['Words'])

plt.title('Top 5 Most Frequently Used Words in 1-Star Musical Instrument Amazon Reviews')

plt.xlabel('Top 5 Words')

plt.ylabel('Word Frequencies')
#Let's visualize the five-most common words in five-star reviews last: 
import matplotlib.pyplot as plt

plt.bar(topFiveWordsByRatingsDFList[4]['Words'], topFiveWordsByRatingsDFList[4]['FrequencyUtilized'])

plt.xticks(topFiveWordsByRatingsDFList[4]['Words'])

plt.title('Top 5 Most Frequently Used Words in 1-Star Musical Instrument Amazon Reviews')

plt.xlabel('Top 5 Words')

plt.ylabel('Word Frequencies')

#Next, I'll remove "major" words shared by the review categories. 
