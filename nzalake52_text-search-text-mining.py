#We will be implementing a simple search engine using nltk in python
#With the help of TF-IDF ranking and cosine similarity we can rank the documents and get the desired output.
#Following is the code for the same, many machine learning algorithms can be applied and the use of this search engine 
#can be extended.
#This is the simplest search engine implementation


import re
from nltk import word_tokenize
import string
import  gensim.models as md
from gensim.models.phrases import Phrases, Phraser


#Creating the function for preprocessing the text file, the functions does the following:
#Removing the URLs in the file
#Removing blank lines
#Converting the letters that were not read properly due to encoding, to be viewed properly
#Removing stopwords from the text
#Removing the punctuations form the text

def preProcessor(textFile):
    print('Starting pre-processing of the corpus..')
    print('Start: Word Tokenizing')

    textFilev1 = []
    textFilev1 = [word_tokenize(sent) for sent in textFile]

    print('Stop: Word Tokenizing')
    print('Start: ASCII encoding for special characters')

    textFilev2 = []
    for sent in textFilev1:
        new_sent = []
        for word in sent:
            new_word = word.encode('ascii', 'ignore').decode('utf-8')
            if new_word != '':
                new_sent.append(new_word)
        textFilev2.append(new_sent)

    print('Stop: ASCII encoding for special characters')
    print('Start: Stopwords Removal')

    stopwordsFile = open('../input/stopwords/stopwords.txt')
    stopwordsFile.seek(0)
    stopwordsV1 = stopwordsFile.readlines()
    stopwordsV2 = []
    for sent in stopwordsV1:
        sent.replace('\n', '')
        new_word = sent[0:len(sent) - 1]
        stopwordsV2.append(new_word.lower())

    textFilev1 = []
    for sent in textFilev2:
        new_sent = []
        for word in sent:
            if word.lower() not in stopwordsV2:
                new_sent.append(word.lower())
        textFilev1.append(new_sent)

    print('Stop: Stopwords Removal')
    print('Start: Punctuation Removal')

    textFilev2 = []
    for sent in textFilev1:
        new_sent = []
        for word in sent:
            if word not in string.punctuation:
                new_sent.append(word)
        textFilev2.append(new_sent)

    print('Stop: Punctuation Removal')
    print('Start: Phrase Detection')

    textFilev1 = []
    common_terms = ["of", "with", "without", "and", "or", "the", "a", "so", "and"]
    phraseTrainer = Phrases(textFilev2, delimiter=b' ', common_terms=common_terms)
    phraser = Phraser(phraseTrainer)
    for article in textFilev2:
        textFilev1.append((phraser[article]))

    print('Stop: Phrase Detection')

    return textFilev1
import pandas as pd

#Reading the news articles file
nyTimesFile = open('../input/new-york-times-articles/nytimes_news_articles.txt', encoding='latin-1')
nyTimesFile.seek(0)
nyTimesV1 = nyTimesFile.readlines()
nyTimesTemp = []
nyTimesURL = []

for i in range(0, len(nyTimesV1)-1):
    if re.findall('URL', nyTimesV1[i]) == []:
        sent = sent + nyTimesV1[i]
        if (re.findall('URL', nyTimesV1[i+1]) != []) and (i+1 < len(nyTimesV1)):
            nyTimesTemp.append(sent.strip())
    else:
        sent = ''
        nyTimesURL.append(nyTimesV1[i])

for i in range(0, len(nyTimesTemp)):
    nyTimesTemp[i] = nyTimesTemp[i]+'articleID'+str(i)

nytimes = preProcessor(nyTimesTemp)
#Function for creating intermediate index
def file_indexing(file):
    fileIndex = {}
    for index, word in enumerate(file):
        if word in fileIndex.keys():
            fileIndex[word].append(index)
        else:
            fileIndex[word] = [index]
    return fileIndex

#building final index
def fullIndex(intIndex):
    totalindex = {}
    for fileName in intIndex.keys():
        for word in intIndex[fileName].keys():
            if word in totalindex.keys():
                if fileName in totalindex[word].keys():
                    totalindex[word][fileName].extend(intIndex[fileName][word][:])
                else:
                    totalindex[word][fileName] = intIndex[fileName][word]
            else:
                totalindex[word] = {fileName : intIndex[fileName][word]}
    return totalindex

nyTimesIndex = {}
for sent in nytimes:
    nyTimesIndex[' '.join(sent)] = file_indexing(sent)

nyTimesIndexV1 = fullIndex(nyTimesIndex)
#Functions to create one word or phrase query search
def wordSearch(word):
    if word in nyTimesIndexV1.keys():
        return [file for file in nyTimesIndexV1[word].keys()]


def phraseQuery(string):
    lists, result = [], []
    for word in string.split():
        lists.append(wordSearch(word))
    setList = set(lists[0]).intersection(*lists)
    for fileName in setList:
        result.append(fileName)
    return result
searchResult = phraseQuery('white  house')
searchResult1 = []
for file in wordSearch('white'):
    searchResult1.append(file)
for file in wordSearch('house'):
    if file not in searchResult1:
        searchResult1.append(file)
#Making use of TF-IDF ranking to rank the documents given by the searches
#Also using similarity metrics (Cosine similarity) to get the similarity scores between both the documents 
#from both search results
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer().fit(searchResult1)
searchResult1TFIDF = tfidf.transform(searchResult1)
searchResultTFIDF = tfidf.transform(searchResult)
sims = cosine_similarity(searchResult1TFIDF, searchResultTFIDF)

#Now putting the cosine similarity results into a data frame
#Sorting the dataframe by score and getting the most similar and appropriate 30 documents from the search results
cosineSum = []
for ind in range(len(sims)):
    cosineSum.append(sum(sims[ind]))

sumDF = pd.DataFrame({'score':cosineSum})
sumDF['index'] = [i for i in range(len(cosineSum))]
sumDF.sort_values(by='score', inplace=True, ascending=False)

for ind in sumDF['index']:
    print(nyTimesURL[int(searchResult1[ind][str(searchResult1[ind]).find('articleid')+9:])], '\n')

