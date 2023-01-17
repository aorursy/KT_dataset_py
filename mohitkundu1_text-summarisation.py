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
# Importing the required Libraries

import nltk

# nltk.download('punkt') # one time execution

import re

#nltk.download('stopwords') # one time execution

import matplotlib.pyplot as plt



from nltk.tokenize import sent_tokenize



from nltk.corpus import stopwords



from sklearn.metrics.pairwise import cosine_similarity



import networkx as nx
# Extract word vectors

word_embeddings = {}

file = open('../input/glove-word-embeddings/glove.6B.100d.txt', encoding='utf-8')

for line in file:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    word_embeddings[word] = coefs

file.close()

len(word_embeddings)
# reading the file

df = pd.read_excel('../input/medicine-descrptions/TASK.xlsx')

df
df.columns
df.rename(columns = {'Unnamed: 1' : 'Introduction' }, inplace=True)

# Deleting the first row

df.drop(0)
# Converting the DataFrame into a dictionary

text_dictionary = {}

for i in range(1,len(df['TEST DATASET'])):

    text_dictionary[i] = df['Introduction'][i]

    

print(text_dictionary[1])
# function to remove stopwords

def remove_stopwords(sen):

    stop_words = stopwords.words('english')

    

    sen_new = " ".join([i for i in sen if i not in stop_words])

    return sen_new
# function to make vectors out of the sentences

def sentence_vector_func (sentences_cleaned) : 

    sentence_vector = []

    for i in sentences_cleaned:

        if len(i) != 0:

            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)

        else:

            v = np.zeros((100,))

        sentence_vector.append(v)

    

    return (sentence_vector)
# function to get the summary of the articles

# NOTE - Remove '#' infront of print statement for displaying the contents at different stages of the text summarisation process

def summary_text (test_text, n = 5):

    sentences = []

    

    # tokenising the text 

    sentences.append(sent_tokenize(test_text))

    # print(sentences)

    sentences = [y for x in sentences for y in x] # flatten list

    # print(sentences)

    

    # remove punctuations, numbers and special characters

    clean_sentences = pd.Series(sentences).str.replace("[^a-z A-Z 0-9]", " ")



    # make alphabets lowercase

    clean_sentences = [s.lower() for s in clean_sentences]

    #print(clean_sentences)



    

    # remove stopwords from the sentences

    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

    #print(clean_sentences)

    

    sentence_vectors = sentence_vector_func(clean_sentences)

    

    # similarity matrix

    sim_mat = np.zeros([len(sentences), len(sentences)])

    #print(sim_mat)

    

    # Finding the similarities between the sentences 

    for i in range(len(sentences)):

        for j in range(len(sentences)):

            if i != j:

                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

    

    

    nx_graph = nx.from_numpy_array(sim_mat)

    scores = nx.pagerank(nx_graph)

    #print(scores)

    

    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)))

    # Extract sentences as the summary

    summarised_string = ''

    for i in range(n):

        

        try:

            summarised_string = summarised_string + str(ranked_sentences[i][1])            

        except IndexError:

            print ("Summary Not Available")

    

    return (summarised_string)
print("Kindly let me know in how many sentences you want the summary - ")

x = 3



summary_dictionary = {}



for key in text_dictionary:

    

    para = text_dictionary[key]    

    summary = summary_text(para,x)

    summary_dictionary[key] = summary

    if key>0 and key<=10 :

        print("Summary of the article - ",key)

        print(summary)

        print('='*50)    

    

print ("*"*10,"The process has been completed successfully","*"*10)
summary_table = pd.DataFrame(list(summary_dictionary.items()),columns = ['TEST DATASET','Summary'])

data_table = pd.DataFrame(list(text_dictionary.items()),columns = ['TEST DATASET','Introduction'])

# Combining the findings into the table

result  = pd.concat([data_table , summary_table['Summary']], axis = 1 , sort = False)

result
# Saving it to a file (remove the '#' to save)

result.to_csv("Summary_File.csv")