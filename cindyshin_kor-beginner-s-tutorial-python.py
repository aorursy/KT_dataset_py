# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the "../input/" directory.

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('../kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# # Any results you write to the current directory are saved as output.
# read in some helpful libraries

import nltk # the natural language toolkit, open-source NLP

import pandas as pd # dataframes

import zipfile



### Read in the data



# read our data into a dataframe

df = pd.DataFrame()

#pd.read_csv() - df = pd.read_csv('../input/spooky-author-identification/train.zip', compression='zip', header=0, sep=',', quotechar='"')

Dataset = 'train'



# Will unzip the files so that you can see them..

with zipfile.ZipFile("../input/spooky-author-identification/"+Dataset+".zip","r") as z:

    z.extractall(".")



texts = pd.read_csv(Dataset + '.csv')



# look at the first few rows

texts.head()
#print(os.path.join(dirname, filename))
#import os

#print(os.listdir('../input/spooky-author-identification/train.zip'))
### Split data



# split the data by author

byAuthor = texts.groupby('author')



### Tokenize (split into individual words) our text



# word frequency by author

wordFreqByAuthor = nltk.probability.ConditionalFreqDist()



# for each author...

for name, group in byAuthor:

    # get all of the sentences they wrote and collapse them into a

    # single long string

    sentences = group['text'].str.cat(sep = ' ')

    

    # convert everything to lower case (so 'The' and 'the' get counted as

    # the same word rather than two different words)

    sentences = sentences.lower()

    

    # split the text into individual tokens

    tokens = nltk.tokenize.word_tokenize(sentences)

    

    # calculate the frequency of each token

    frequency = nltk.FreqDist(tokens)

    

    # add the frequencies for each author to our dictionary

    wordFreqByAuthor[name] = (frequency)

    

# now we have an dictionary where each entry is the frequency distribution

# of words for a specific author
# see how often each author says 'blood'

for i in wordFreqByAuthor.keys():

    print('blood: ' + i)

    print(wordFreqByAuthor[i].freq('blood'))



# print a blank line

print()



# see how often each author says 'scream'

for i in wordFreqByAuthor.keys():

    print('scream: ' + i)

    print(wordFreqByAuthor[i].freq('scream'))

    

# print a blank line

print()



# see how often each author says 'fear'

for i in wordFreqByAuthor.keys():

    print('fear: ' + i)

    print(wordFreqByAuthor[i].freq('fear'))
# One way to guess authorship is to use the joint probability that each

# author used each word in a given sentence



# first, let's start with a test sentence

testSentence = "It was a dark and stormy night."



# and then lowercase & tokenize our test sentence

preProcessedTestSentence = nltk.tokenize.word_tokenize(testSentence.lower())



# create an empty dataframe to put our output in

testProbabilities = pd.DataFrame(columns = ['author', 'word','probability'])



# For each author...

for i in wordFreqByAuthor.keys():

    # for each word in our test sentence..

    for j in preProcessedTestSentence:

        # find out how frequentyly the author used that word

        wordFreq = wordFreqByAuthor[i].freq(j)

        # and add a very small amount to every prob. so none of them are 0

        smoothedWordFreq = wordFreq + 0.000001

        # add the author, word and smoothed freq. to our dataframe

        output = pd.DataFrame([[i,j,smoothedWordFreq]], columns = ['author', 'word','probability'])

        testProbabilities = testProbabilities.append(output, ignore_index = True)



# empty dataframe for the probability that each author wrote the snetence

testProbabilitiesByAuthor = pd.DataFrame(columns = ['author','jointProbability'])



# now let's group the dataframe with our frequency by author

for i in wordFreqByAuthor.keys():

    # get the joint probability that each author wrote each word

    oneAuthor = testProbabilities.query('author == "' + i + '"')

    jointProbability = oneAuthor.product(numeric_only = True)[0]

    

    # and add that to our dataframe

    output = pd.DataFrame([[i, jointProbability]], columns = ['author','jointProbability'])

    testProbabilitiesByAuthor = testProbabilitiesByAuthor.append(output, ignore_index=True)

    

# and our winner is...

testProbabilitiesByAuthor.loc[testProbabilitiesByAuthor['jointProbability'].idxmax(),'author']
testProbabilitiesByAuthor.to_csv("testProbabilityByAuthor.csv", index=False)