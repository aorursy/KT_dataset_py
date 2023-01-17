# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

#read data

df_cluster = pd.read_csv("/kaggle/input/cord19-challenge-finding-relevant-articles/Cluster.csv")

df_cluster.head()
import json



#This function finds the relevant json filename given the sha

def find(name, path):

    filename = name + '.json'

    for root, dirs, files in os.walk(path):

        if filename in files:

            return os.path.join(root, filename)



#This extracts from the json file the body text, title and author

def parse_json(name, path):

    filename = find(name, path)

    try:

        with open(filename, "r") as read_file:

            data = json.load(read_file)

            body_text = data["body_text"]

            title = data["metadata"]["title"]

            author = data["metadata"]["authors"]

    

        return body_text, title, author

    except:

        return "", "", ""





#Load all json files for the selected articles

df_cluster['data'] = df_cluster['sha'].apply(lambda x: parse_json(str(x), "/kaggle/input/"))



#Select a article to evaluate the different ways of extracting the information. We first look at the raw body_text 

article_test = df_cluster['data'][2]

print(article_test)
#combined all the text part of the body_text into a single corpus

def extract_fulltext(json_data):

    body_text = json_data[0]

    corpus = []

    for i in range(len(body_text)):

        corpus.append(body_text[i]["text"]) 

    document = " ".join(corpus)

    return document



#Extract author names from the author part of the full_text

def extract_authors(dict_list):

    name_list = []

    for item in dict_list:

        name_list.append(" ".join([item["first"], item["last"].upper()]))

    return name_list



#Print out extracted info. This could be a typical format for displaying the full_text of an article.

print("Title: ", article_test[1])

print("Authors: ", extract_authors(article_test[2]))

print("Full Text: ", extract_fulltext(article_test))
import re



#This function removes the doi references in the articles

def clean_fulltext(full_text):

    full_text = re.sub(r"\. CC-BY-NC(.*?)preprint(.*?)doi:(.*?)preprint", "", full_text)

    full_text = re.sub(r"The copyright(.*?)preprint(.*?)doi:(.*?)preprint", "", full_text)

    full_text = re.sub(r"CC-BY 4.0 International license author/funder", "", full_text)

    return full_text



clean_text = clean_fulltext(extract_fulltext(article_test))



from gensim.summarization.summarizer import summarize



#Perform extractive summarization with a range of wordcounts

for word_count in range(50,350,50):

    summary = summarize(clean_text, word_count = word_count)

    print("No of words: ", word_count)

    print("Summary: ", summary)

    print("\n")







import string

from nltk.tokenize import word_tokenize



#This code check if a token is a number, and seperate between integer and float

def check_token(token):

    try:

        num = int(str(token))

        return "integer"

    except (ValueError, TypeError):

        try:

            num = float(str(token))

            return "float"

        except (ValueError, TypeError):

            return "string"





def extract_key_sentences(clean_text):

    #Break the text up into sentences

    sentences = re.split(r"(?<=\.) {1,}(?=[A-Z])", clean_text)





    #Empty list to store key sentences and words

    key_sentences = []

    fig_sentences = []

    special_words = []



    #Specify special tokens to look for: common units, comparasion operators, and tokens related to figures and tables

    units = ["kcal", "K", "ml", "mL", "L", "J", "kJ", "mol", "eV"]

    ops = [">", "<", "=", ">=", "<=", "≤"]

    fig_list = ["Figure", "FIGURE", "Fig", "figure", "fig", "figures", "Table", "table", "TABLE", "Graph", "graph"]



    #For each sentence in the full text, check for tokens, and either score according to the number detected in the sentence, select the sentence if it is referring to a figure/table/graph, 

    #or flag as a special term if it contains both letter and numbers



    for s in sentences:

        #First tokenise to seperate into individual tokens

        tokens = word_tokenize(s)

        score = 0

        selected = 0

        for i in range(len(tokens)):

            #check token - int, float or string

            word_type = check_token(tokens[i])

        

            #for integers, only consider numbers that are not years

            #And for other integers, only contribute to score if it is followed by a percent, a unit, or followed or precede by one of the comparison operators

            if word_type == "integer":

                if (int(str(tokens[i])) < 1800) or (int(str(tokens[i])) > 2030):

                    if (i > 0) and (i < len(tokens) - 1):

                        if ((str(tokens[i+1])) == "%") or (str(tokens[i+1]) in units) or (str(tokens[i-1]) in ops) or (str(tokens[i+1]) in ops):

                            score = score + 1

            #All floats are consider important and are scored higher

            if word_type == "float":

                score = score + 3

            #for strings,check if it is a "figure sentence"

            if word_type == "string":

                if (str(tokens[i]) in fig_list):

                    selected = 1



        #Save sentence if they fit criteria

        if score > 0:

            key_sentences.append((s, score))

        if selected > 0:

            fig_sentences.append(s)

    

    #Store empty string if no key sentences are detected

    if len(key_sentences) == 0:

        key_sentences.append(("",0))

    

    #Sort by score before returning it

    key_sentences.sort(key=lambda x:x[1], reverse = True)

        

    return key_sentences, fig_sentences



key_sentences, fig_sentences = extract_key_sentences(clean_text)

      



#Print the top five key sentences

print("Key results extracted:\n")

for i in range(5):

    print(key_sentences[i][0]+"\n")

   

    

#Print figure references

print("Sentences that refer to figures/tables:\n")

for i in range(len(fig_sentences)):

    print(fig_sentences[i] + "\n")
#start by storing summary and key sentenes into dataframe





df_cluster['full_title'] = [x[1] for x in df_cluster['data']]

df_cluster['authors'] = [x[2] for x in df_cluster['data']]

df_cluster['authors'] = df_cluster['authors'].apply(lambda x: extract_authors(x))

df_cluster['text'] = df_cluster['data'].apply(lambda x: extract_fulltext(x))

df_cluster['text'] = df_cluster['text'].apply(lambda x: clean_fulltext(x))

df_cluster['summary'] = df_cluster['text'].apply(lambda x: summarize(x, word_count = 150))

df_cluster['key_sentences'] = df_cluster['text'].apply(lambda x: extract_key_sentences(x))

results_list = [x[0] for x in df_cluster['key_sentences']]

df_cluster['results'] = df_cluster['key_sentences'].apply(lambda x: x[0])

df_cluster['results'] = df_cluster['results'].apply(lambda x: [y[0] for y in x])

df_cluster['figures'] = [x[1] for x in df_cluster['key_sentences']]



def find_information(keyword_list, options = "Summary", num_sentences = 5):

    df_temp = []

    if options == "Summary":

        df_temp =  df_cluster[df_cluster['summary'].apply(lambda x: all(substring in x for substring in keyword_list)) == True]

    elif options == "Full text":

        df_temp =  df_cluster[df_cluster['text'].apply(lambda x: all(substring in x for substring in keyword_list)) == True]

    elif options == "Figures":

        df_temp =  df_cluster[df_cluster['figures'].apply(lambda x: all(substring in x for substring in keyword_list)) == True]

    elif options == "Results":

        df_temp =  df_cluster[df_cluster['results'].apply(lambda x: all(substring in x for substring in keyword_list)) == True]



    if (len(df_temp) == 0):

        print("The Search Cannot find any relevant articles")

    else:

        print("Number of Articles Found: " + str(len(df_temp)) + "\n")

    

        for i in range(len(df_temp)):

            print("Title: ", df_temp['title'].iloc[i], '\n')

            print("Authors: ", ",".join(df_temp['authors'].iloc[i]), '\n')

            print("Summary: ", df_temp['summary'].iloc[i], '\n')

            print("Key Results: \n")

            for j in range(len(df_temp['results'].iloc[i])):

                if j < num_sentences:

                    print(df_temp['results'].iloc[i][j]+"\n")

            print("Sentences that refer to figures/tables:\n")

            for j in range(len(df_temp['figures'].iloc[i])):

                if j < num_sentences:

                    print(df_temp['figures'].iloc[i][j]+"\n")

    

    

    

    return df_temp

    





df_answer = find_information(["vaccine", "SARS"])