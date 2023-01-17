from IPython.display import Image

Image(filename="../input/covid19-images/NGC.PNG", width=500, height=500)
from IPython.display import Image

Image(filename="../input/covid19-images/covid-19.jpg", width=500, height=500)
from IPython.display import Image

Image(filename="../input/covid19-images/BERT.png", width=500, height=500)
from IPython.display import Image

Image(filename="../input/covid19-images/Bert_STS.PNG", width=500, height=500)
#Installs needed

!pip install langdetect

!pip install semantic-text-similarity



#Libraries needed

import pandas as pd 

import glob

import json

import re 

import numpy as np

import copy 

import torch 

import matplotlib.pyplot as plt

from langdetect import detect

from semantic_text_similarity.models import ClinicalBertSimilarity

from nltk.tokenize import sent_tokenize, word_tokenize

from gensim.models import Word2Vec

from sklearn.manifold import TSNE

print("Done")
#Read in the metadata

print("Reading in metadata.")

metadata_path = f'../input/cord1933k/metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})

print("Metadata read.")

print()
#Get all file names 

all_json = glob.glob(f'../input/cord1933k/**/*.json', recursive=True)

print(str(len(all_json))+" total file paths fetched.")

print()
#Reader class for files 

class FileReader:

    def __init__(self, file_path):

        with open(file_path) as file:

            content = json.load(file)

            self.paper_id = content['paper_id']

            self.abstract = []

            self.body_text = []

            # Abstract

            for entry in content['abstract']:

                self.abstract.append(entry['text'])

            # Body text

            for entry in content['body_text']:

                self.body_text.append(entry['text'])

            self.abstract = '\n'.join(self.abstract)

            self.body_text = '\n'.join(self.body_text)

        return 
#Read in all text files and store in a Pandas data frame 

print("Reading in text files.")

dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'title': [], 'authors': [], 'journal': [], 'url': []}

for idx, entry in enumerate(all_json[:1000]):

    print(f'Processing index: {idx} of {len(all_json)}', end='\r')

    content = FileReader(entry)

    

    # get metadata information

    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    # no metadata, skip this paper

    if len(meta_data) == 0:

        continue

    

    dict_['paper_id'].append(content.paper_id)

    dict_['abstract'].append(content.abstract)

    dict_['body_text'].append(content.body_text)

    dict_['title'].append(meta_data.iloc[0]['title'])

    dict_['authors'].append(meta_data.iloc[0]['authors'])

    dict_['journal'].append(meta_data.iloc[0]['journal'])

    dict_['url'].append(meta_data.iloc[0]['url'])

df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'title', 'authors', 'journal', 'url'])

dict_ = None

print("Text files read.")

print()
#Get a count for the number of words in articles and abstracts 

print("Getting abstract and body word counts.")

df_covid['abstract_word_count'] = df_covid['abstract'].apply(lambda x: len(x.strip().split()))

df_covid['body_word_count'] = df_covid['body_text'].apply(lambda x: len(x.strip().split()))

print("Counts computed.")

print()
#Saving the dataframe 

print("Saving the dataframe.")

df_covid.to_csv('covidData.csv') 

print("Dataframe saved.")
#Read in the saved data 

print("Loading the dataframe.")

df_covid = pd.read_csv('../input/cord19cleaneddata/covidData.csv')

print("Dataframe loaded.")

print()

df_covid = df_covid.head(1000)
#Remove all articles that have fewer than the number of words specified 

min_word_count = 1000

print("Removing all articles with fewer than "+str(min_word_count)+" words.")

indexNames = df_covid[df_covid['body_word_count'] < min_word_count].index

df_covid = df_covid.drop(indexNames)

df_covid = df_covid.reset_index(drop=True)

print("Articles cleaned.")

print()
#Remove all non-English articles

print("Removing all non-English articles")

index = 0

indexNames = []

while(index < len(df_covid)):

    print(f'Processing index: {index} of {len(df_covid)}', end='\r')

    language = detect(df_covid.iloc[index]['body_text'])

    if(language != 'en'):

        indexNames.append(index)

    index += 1

df_covid = df_covid.drop(indexNames)

df_covid = df_covid.reset_index(drop=True)

print("All non-English articles removed. Total article count is now: "+str(len(df_covid)))

print()
#Save the cleaned dataset 

print("Saving the dataframe.")

df_covid.to_csv('covidDataCleaned.csv') 

print("Dataframe saved.")
#Read in the saved data 

print("Loading the dataframe.")

df_covid = pd.read_csv('../input/cord19cleaneddata/covidDataCleaned.csv')

print("Dataframe loaded.")

print()

df_covid = df_covid.head(5000)
#Get the article text and pre-process by converting to lowercase and removing weird characters 

print("Pre-processing articles by converting to lowercase and removing random characters.")

articleTexts = df_covid.drop(["paper_id", "abstract", "title", "authors", "journal", "url", "abstract_word_count", "body_word_count"], axis=1)

articleTexts['body_text'] = articleTexts['body_text'].apply(lambda x: x.lower())

articleTexts['body_text'] = articleTexts['body_text'].apply(lambda x: re.sub('[^a-z0-9.!?\s]','',x))

text_list = list(articleTexts['body_text'])

print("Pre-processing complete.")

print()
#Extract all sentences - gensim expects a sequence of sentences as input

#Example: sentences = [['first', 'sentence'], ['second', 'sentence']]

print("Tokenizing articles into sentences and words.")

sentences = []

for index, text in enumerate(text_list):

    sentenceList = sent_tokenize(text)

    for sentence in sentenceList:

        wordList = word_tokenize(sentence)

        sentences.append(wordList)

    print("Processing article "+str(index)+" of "+str(len(text_list)), end="\r")

print("A total of "+str(len(sentences))+" sentences have been tokenized for word2vec training.")

print()
#Train & save the word2vec model 

print("Training word2vec.")

model = Word2Vec(sentences, size=100, window=5, min_count=10, workers=4)

print("Word count:", len(list(model.wv.vocab)))

model.save("word2vec.model")

print("Finished training and saving word2vec.")
#Load the trained word2vec model 

print("Loading the pre-trained word2vec model.")

model = Word2Vec.load("word2vec.model")

print("Model loaded.")

print()
#From: https://methodmatters.github.io/using-word2vec-to-analyze-word/

#Define the function to compute the dimensionality reduction and then produce the biplot  

def tsne_plot(model, words):

    "Creates a TSNE model and plots it"

    labels = []

    tokens = []

    

    print("Getting embeddings.")

    for word in model.wv.vocab:  

        if(word in words):

            tokens.append(model[word])

            labels.append(word)

    print("Embeddings extracted.")

    print()

        

    print("Performing dimensionality reduction with t-sne.")

    tsne_model = TSNE(perplexity=5, n_components=2, init='pca', n_iter=2500, verbose=0)

    new_values = tsne_model.fit_transform(tokens)

    print("Dimensioanlity reduction complete.")

    print()

    

    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(8, 8))

    for i in range(len(x)):

        if(labels[i] in words):

            plt.scatter(x[i],y[i])

            plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    plt.show()

    return 
#List of words to visualize in the plot

words = ['china', 'italy', 'taiwan', 'india', 'japan', 'france', 

         'spain', 'canada', 'infection', 'disease', 'pathogen', 

         'organism', 'bacteria', 'virus', 'covid19', 'coronavirus', 

         'healthcase', 'doctor', 'nurse', 'specialist', 'hospital', 

         'novel', 'human', 'sars', 'covid', 'wuhan', 'case', 

         'background', 'dynamic', 'pneumonia', 'outbreak', 'pandemic', 

         'syndrome', 'contact', 'wash', 'hands', 'cough', 

         'respiratory', 'case', 'fear', 'spike', 'curve', 

         'transmission', 'seasonal', 'genome', 'dna', 'testing', 

         'asymptomatic', 'global', 'spread', 'diagnosis']

  

#Call the function on our dataset  

tsne_plot(model, words)
#Word to compare against and number of similar words to print out 

word = 'facemask'

similarCount = 3
#Get and print the results 

results = model.wv.most_similar(positive=word, topn=similarCount)

print("Input word:", word)

print("Top "+str(similarCount)+" similar words are:")

for index, word in enumerate(results):

    print(str(index+1)+". "+word[0]+" --- Score: "+str(word[1]))
#Words to compute cosine similarity over  

word1 = 'china'

word2 = 'wuhan'



#Get the word embeddings 

embedding1 = model.wv[word1]

embedding2 = model.wv[word2]
#Compute the cosine similarity and print the results 

cosineSimilarity = np.sum(embedding1*embedding2) / (np.sqrt(np.sum(np.square(embedding1)))*np.sqrt(np.sum(np.square(embedding2))))

print("Word1: "+word1+" --- Word2: "+word2)

print("Cosine similarity: "+ str(cosineSimilarity))
#Set the GPU device

device = 0

torch.cuda.set_device(device)
#Read in the saved data 

print("Loading the dataframe.")

df_covid = pd.read_csv('../input/cord19cleaneddata/covidDataCleaned.csv')

print("Dataframe loaded.")

print()

df_covid = df_covid.head(500)
#Variable to store the batch size

batchSize = 500 
#Load the model

print("Loading BERT semantic similarity model.")

model = ClinicalBertSimilarity(device='cuda', batch_size=batchSize) #defaults to GPU prediction

print("Model loaded.")

print()
#The primary questions that attempt to be answered  

primaryQuestions = [

    "What is known about transmission, incubation, and environmental stability of coronavirus"

    #"What do we know about coronavirus risk factors"

    #"What do we know about coronavirus genetics, origin, and evolution"

    #"What do we know about vaccines and therapeutics for coronavirus"

    #"What has been published about coronavirus medical care"

    #"What do we know about non-pharmaceutical interventions for coronavirus"

    #"What do we know about diagnostics and surveillance of coronavirus"

    #"In what ways does geography affects virality"

    #"What has been published about ethical and social science considerations regarding coronavirus"

    #"What has been published about information sharing and inter-sectoral collaboration"

]
#Extract the text from the articles 

articleTexts = df_covid.drop(["paper_id", "abstract", "title", "authors", "journal", "url", "abstract_word_count", "body_word_count"], axis=1)

text_list = list(articleTexts['body_text'])
#Get the index of where the prediction ranks among the current best predictions 

def computeScoreIndex(predictionScore, answerScores):

    index = 0

    while(index < len(answerScores)):

        if(predictionScore > answerScores[index]):

            break

        index += 1

    return index 
#Function to make a batch to send through the semantic similarity model 

def makeBatch(query, sentences):

    batch = []

    index = 0

    while(index < len(sentences)):

        batch.append((query, sentences[index]))

        index += 1

    return batch 
#Set scores to strings for saving 

def convertScores(answers):

    for key in answers.keys():

        for scoreIndex, score in enumerate(answers[key]['scores']):

            answers[key]['scores'][scoreIndex] = str(answers[key]['scores'][scoreIndex])

    return answers
#Dictionary to store answers and variable to specify the number of answers to store 

answerCount = 10

questionResponses = {'titles': [], 'sentences': [], 'scores': []}

answers = {}

for query in primaryQuestions:

    answers[query] = copy.deepcopy(questionResponses)
#Function to update the most relevant answers based on model predictions 

def updateAnswersSentences(query, title, sentences, predictions, answers, answerCount):

    #Get the top answerCount prediction scores 

    topIndices = predictions.argsort()[-answerCount:][::-1]

    for index in topIndices:

        #Case where lists are empty

        if(len(answers[query]['scores']) == 0):

            answers[query]['titles'].append(title)

            answers[query]['sentences'].append(sentences[index])

            answers[query]['scores'].append(predictions[index])

        #Case where lists have length shorter than answerCount 

        elif(len(answers[query]['scores']) < answerCount):

            scoreIndex = computeScoreIndex(predictions[index], answers[query]['scores'])

            answers[query]['titles'].insert(scoreIndex, title)

            answers[query]['sentences'].insert(scoreIndex, sentences[index])

            answers[query]['scores'].insert(scoreIndex, predictions[index])

        #Case where lists are full 

        else:

            scoreIndex = computeScoreIndex(predictions[index], answers[query]['scores'])

            #Check to see if an item should be bumped out of the list 

            if(scoreIndex < answerCount):

                answers[query]['titles'].insert(scoreIndex, title)

                answers[query]['sentences'].insert(scoreIndex, sentences[index])

                answers[query]['scores'].insert(scoreIndex, predictions[index])

                answers[query]['titles'] = answers[query]['titles'][:answerCount]

                answers[query]['sentences'] = answers[query]['sentences'][:answerCount]

                answers[query]['scores'] = answers[query]['scores'][:answerCount]

    return answers 
#Loop through all queries   

for queryIndex, query in enumerate(primaryQuestions):

    #Loop through all articles 

    for textIndex, text in enumerate(text_list):

        #Tokenize the article to sentences and loop through all sentences computing prediction scores for each 

        sentences = sent_tokenize(text)

        batchIndex = 0

        while(batchIndex*batchSize < len(sentences)):

            batchSentences = None

            batch = None

            #Check to make sure the batch size won't go out of bounds in regard to the sentences 

            if((batchIndex*batchSize)+batchSize > len(sentences)):

                batchSentences = sentences[batchIndex*batchSize:len(sentences)]

                batch = makeBatch(query, batchSentences)

            else:

                batchSentences = sentences[batchIndex*batchSize:(batchIndex*batchSize)+batchSize]

                batch = makeBatch(query, batchSentences)

            predictions = model.predict(batch)

            answers = updateAnswersSentences(query, df_covid.iloc[textIndex]["title"], batchSentences, predictions, answers, answerCount)

            batchIndex += 1

        print("Processing query "+str(queryIndex)+" of "+str(len(primaryQuestions))+" --- Article "+str(textIndex)+" of "+str(len(text_list)), end='\r')

    print()
#Save the results

answers = convertScores(answers)

jsonData = json.dumps(answers)

f = open("topSentences.json","w")

f.write(jsonData)

f.close()
#Read in the json file of results 

filename = "topSentences.json"

answers = None

with open(filename, 'r') as myfile:

    answers = json.load(myfile)
#Print the results 

for query in answers.keys():

    print("Query: "+query)

    resultCount = len(answers[query]['scores'])

    index = 0

    while(index < resultCount):

        print(str(index+1)+". Article: "+str(answers[query]['titles'][index]))

        print("Sentence: "+answers[query]['sentences'][index])

        print("Score: "+answers[query]['scores'][index])

        print()

        index += 1
#Dictionary to store answers and variable to specify the number of answers to store 

answerCount = 10

questionResponses = {'titles': [], 'scores': []}

answers = {}

for query in primaryQuestions:

    answers[query] = copy.deepcopy(questionResponses)
#Function to update the most relevant answers based on model predictions 

def updateAnswersArticles(query, title, articleScore, answers, answerCount):

    #Case where lists are empty

    if(len(answers[query]['scores']) == 0):

        answers[query]['titles'].append(title)

        answers[query]['scores'].append(articleScore)

    #Case where lists have length shorter than answerCount 

    elif(len(answers[query]['scores']) < answerCount):

        scoreIndex = computeScoreIndex(articleScore, answers[query]['scores'])

        answers[query]['titles'].insert(scoreIndex, title)

        answers[query]['scores'].insert(scoreIndex, articleScore)

    #Case where lists are full 

    else:

        scoreIndex = computeScoreIndex(articleScore, answers[query]['scores'])

        #Check to see if an item should be bumped out of the list 

        if(scoreIndex < answerCount):

            answers[query]['titles'].insert(scoreIndex, title)

            answers[query]['scores'].insert(scoreIndex, articleScore)

            answers[query]['titles'] = answers[query]['titles'][:answerCount]

            answers[query]['scores'] = answers[query]['scores'][:answerCount]

    return answers 
#Loop through all queries   

for queryIndex, query in enumerate(primaryQuestions):

    #Loop through all articles 

    for textIndex, text in enumerate(text_list):

        #Tokenize the article to sentences and loop through all sentences computing prediction scores for each 

        #Create a variable to store all score values 

        sentences = sent_tokenize(text)

        sentenceScores = np.array([])

        batchIndex = 0

        while(batchIndex*batchSize < len(sentences)):

            batchSentences = None

            batch = None

            #Check to make sure the batch size won't go out of bounds in regard to the sentences 

            if((batchIndex*batchSize)+batchSize > len(sentences)):

                batchSentences = sentences[batchIndex*batchSize:len(sentences)]

                batch = makeBatch(query, batchSentences)

            else:

                batchSentences = sentences[batchIndex*batchSize:(batchIndex*batchSize)+batchSize]

                batch = makeBatch(query, batchSentences)

            predictions = model.predict(batch)

            sentenceScores = np.append(sentenceScores, predictions)

            batchIndex += 1

        articleScore = np.mean(sentenceScores)

        answers = updateAnswersArticles(query, df_covid.iloc[textIndex]["title"], articleScore, answers, answerCount)

        print("Processing query "+str(queryIndex)+" of "+str(len(primaryQuestions))+" --- Article "+str(textIndex)+" of "+str(len(text_list)), end='\r')

    print()
#Save the results

answers = convertScores(answers)

jsonData = json.dumps(answers)

f = open("topArticles.json","w")

f.write(jsonData)

f.close()
#Read in the json file of results 

filename = "topArticles.json"

answers = None

with open(filename, 'r') as myfile:

    answers = json.load(myfile)
#Print the results 

for query in answers.keys():

    print("Query: "+query)

    resultCount = len(answers[query]['scores'])

    index = 0

    while(index < resultCount):

        print(str(index+1)+". Article: "+str(answers[query]['titles'][index]))

        print("Score: "+answers[query]['scores'][index])

        print()

        index += 1
#Load in pandas and set options 

import pandas as pd 

import numpy as np

pd.set_option('display.max_colwidth', 10000)

pd.set_option('display.max_rows', 100)



#Read in the saved data 

results = pd.read_csv('../input/cord19results/topSentences_0.csv')

results = results.drop(['Unnamed: 0'], axis=1)

results.index = np.arange(1, len(results)+1)

display(results)



#Read in the saved data 

results = pd.read_csv('../input/cord19results/topArticles_0.csv')

results = results.drop(['Unnamed: 0'], axis=1)

results.index = np.arange(1, len(results)+1)

display(results.head(10))
#Load in pandas and set options 

import pandas as pd 

import numpy as np

pd.set_option('display.max_colwidth', 10000)

pd.set_option('display.max_rows', 100)



#Read in the saved data 

results = pd.read_csv('../input/cord19results/topSentences_11.csv')

results = results.drop(['Unnamed: 0'], axis=1)

results.index = np.arange(1, len(results)+1)

display(results)



#Read in the saved data 

results = pd.read_csv('../input/cord19results/topArticles_11.csv')

results = results.drop(['Unnamed: 0'], axis=1)

results.index = np.arange(1, len(results)+1)

display(results.head(10))
#Load in pandas and set options 

import pandas as pd 

import numpy as np

pd.set_option('display.max_colwidth', 10000)

pd.set_option('display.max_rows', 100)



#Read in the saved data 

results = pd.read_csv('../input/cord19results/topSentences_12.csv')

results = results.drop(['Unnamed: 0'], axis=1)

results.index = np.arange(1, len(results)+1)

display(results)



#Read in the saved data 

results = pd.read_csv('../input/cord19results/topArticles_12.csv')

results = results.drop(['Unnamed: 0'], axis=1)

results.index = np.arange(1, len(results)+1)

display(results.head(10))
#Load in pandas and set options 

import pandas as pd 

import numpy as np

pd.set_option('display.max_colwidth', 10000)

pd.set_option('display.max_rows', 100)



#Read in the saved data 

results = pd.read_csv('../input/cord19results/topSentences_13.csv')

results = results.drop(['Unnamed: 0'], axis=1)

results.index = np.arange(1, len(results)+1)

display(results)



#Read in the saved data 

results = pd.read_csv('../input/cord19results/topArticles_13.csv')

results = results.drop(['Unnamed: 0'], axis=1)

results.index = np.arange(1, len(results)+1)

display(results.head(10))
#Load in pandas and set options 

import pandas as pd 

import numpy as np

pd.set_option('display.max_colwidth', 10000)

pd.set_option('display.max_rows', 100)



#Read in the saved data 

results = pd.read_csv('../input/cord19results/topSentences_15.csv')

results = results.drop(['Unnamed: 0'], axis=1)

results.index = np.arange(1, len(results)+1)

display(results)



#Read in the saved data 

results = pd.read_csv('../input/cord19results/topArticles_15.csv')

results = results.drop(['Unnamed: 0'], axis=1)

results.index = np.arange(1, len(results)+1)

display(results.head(10))
#Load in pandas and set options 

import pandas as pd 

import numpy as np

pd.set_option('display.max_colwidth', 10000)

pd.set_option('display.max_rows', 100)



#Read in the saved data 

results = pd.read_csv('../input/cord19results/topSentences_16.csv')

results = results.drop(['Unnamed: 0'], axis=1)

results.index = np.arange(1, len(results)+1)

display(results)



#Read in the saved data 

results = pd.read_csv('../input/cord19results/topArticles_16.csv')

results = results.drop(['Unnamed: 0'], axis=1)

results.index = np.arange(1, len(results)+1)

display(results.head(10))
#Load in pandas and set options 

import pandas as pd 

import numpy as np

pd.set_option('display.max_colwidth', 10000)

pd.set_option('display.max_rows', 100)



#Read in the saved data 

results = pd.read_csv('../input/cord19results/topSentences_17.csv')

results = results.drop(['Unnamed: 0'], axis=1)

results.index = np.arange(1, len(results)+1)

display(results)



#Read in the saved data 

results = pd.read_csv('../input/cord19results/topArticles_17.csv')

results = results.drop(['Unnamed: 0'], axis=1)

results.index = np.arange(1, len(results)+1)

display(results.head(10))
#Load in pandas and set options 

import pandas as pd 

import numpy as np

pd.set_option('display.max_colwidth', 10000)

pd.set_option('display.max_rows', 100)



#Read in the saved data 

results = pd.read_csv('../input/cord19results/topSentences_3.csv')

results = results.drop(['Unnamed: 0'], axis=1)

results.index = np.arange(1, len(results)+1)

display(results)



#Read in the saved data 

results = pd.read_csv('../input/cord19results/topArticles_3.csv')

results = results.drop(['Unnamed: 0'], axis=1)

results.index = np.arange(1, len(results)+1)

display(results.head(10))