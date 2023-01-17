import pandas as pd
import numpy as np 

import re
import os
import json

import nltk

from nltk import word_tokenize, sent_tokenize
# from nltk.stem  import PorterStemmer


from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from transformers import BertForQuestionAnswering

import torch
from transformers import  AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/biobert_v1.1_pubmed_squad_v2")
# Given a character sequence and a defined document unit, tokenization is the task of 
# chopping it up into pieces, called tokens , perhaps at the same time throwing 
# away certain characters, such as punctuation
# AutoTokenizer is a generic tokenizer class that will be instantiated as one of the 
# tokenizer classes of the library when created with the 
# AutoTokenizer.from_pretrained(pretrained_model_name_or_path) class method.
# The from_pretrained() method takes care of returning the correct tokenizer class instance based 
# on the model_type property of the config object, or when it’s missing, 
# falling back to using pattern matching on the pretrained_model_name_or_path string

vectorizer = TfidfVectorizer()
# for converting all the text into vectors, 
# the numerical representations of the text

# filter out all dataframes that do not have any references to covid
def search_focus(df):
    dfa = df[df['abstract'].str.contains('covid')]
    dfb = df[df['abstract'].str.contains('-cov-2')]
    dfc = df[df['abstract'].str.contains('cov2')]
    dfd = df[df['abstract'].str.contains('ncov')]
    frames=[dfa,dfb,dfc,dfd]
    df = pd.concat(frames)
    df=df.drop_duplicates(subset='title', keep="first")
    
    return df


def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    
    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"
    
    return body


def removepunc(my_str):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct


def hasNumbers(inputString):
    return (bool(re.search(r'\d', inputString)))

# This is where the actual call to BERT is made.
# This function will look for the answer to question
# in the context
def ask(question,context):
    input_ids = tokenizer.encode(question, context)
    # convert question and context into a list of numerical IDs
    
    sep_index = input_ids.index(tokenizer.sep_token_id)
    # sep_token is special token separating two different sentences in the same input 
    # sep_token_id is the id of that token

    num_seg_a = sep_index + 1

    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    assert len(segment_ids) == len(input_ids)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)


    start_scores, end_scores = model(torch.tensor([input_ids]),
                                 token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text
    answer_end = 0
    answer_start = torch.argmax(start_scores)
    answer_ends = torch.argsort(end_scores).numpy()[::-1]
    
    for i in answer_ends[0]:
        if answer_start<= i:
            answer_end= i

    answer = ' '.join(tokens[answer_start:answer_end+1])
    answer = answer.replace(" ##","").replace("[CLS] ","")

    pack = [answer,answer_start,answer_end,torch.max(start_scores),end_scores[0][answer_end],(torch.max(start_scores)+end_scores[0][answer_end]),context]

    return pack


def getanswers(question):
    recommendations = []
    #
    
    # Iterate over all the usequeries
    for i in range(len(usequeries)):
        indices = np.argsort(similarity_matrix[i])[-7:][::-1] ## I choose to show N recommended queries from every query
        for t in indices:
            recommendations.append(word_tokenize(df.abstract[t]))
  
    processedQuestion =   " ".join([snowstem.stem(i) for i in word_tokenize(removepunc(question)) if i not in stops])
    vector = vectorizer.transform([processedQuestion])
    questionSimilarityMatrix = cosine_similarity(vector,encArticles)
    indicies = np.argsort(questionSimilarityMatrix[0])[-7:][::-1] 
    for t in indicies:
        recommendations.append(word_tokenize(df.abstract[t]))
          
    questions= []
    contexts= []
    for bigcontext in recommendations:
        # iterate over chunks of 60
        for i in range(int(len(bigcontext)/60)):
            contexts.append(" ".join(bigcontext[i*60:60*(i+1)]))
            questions.append(question)

    answers = []
    for  question, context in zip(questions,contexts):
        result = ask(question,context)
        if len(result[0]) < 7 and "[CLS]" in result[0] :
            continue
        answers.append(result)
    answers = np.array(answers)
    for i in np.argsort(answers[:,5])[-8:][::-1]:
        print(i, answers[i,0])
    return answers


# Load the document meta data
df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv', usecols=['title','journal','abstract','authors','doi','publish_time','sha','pdf_json_files'])
print ('All CORD19 documents ',df.shape)

#
# Clean up the meta data
#
# fill na fields
df = df.fillna('no data provided')

# drop duplicate titles
df = df.drop_duplicates(subset='title', keep="first")

# keep only 2020 dated papers
df = df[df['publish_time'].str.contains('2020')]

# convert abstracts to lowercase
df["abstract"] = df["abstract"].str.lower()+df["title"].str.lower()

# show 5 lines of the new dataframe
df = search_focus(df)
print ("COVID-19 focused documents ",df.shape)

#
# Load the JSONs that contain the relevant articles
#
for index, row in df.iterrows():
    if ';' not in row['sha'] and os.path.exists('/kaggle/input/CORD-19-research-challenge/'+row['pdf_json_files']+'/'+row['pdf_json_files']+'/pdf_json/'+row['sha']+'.json')==True:
        with open('/kaggle/input/CORD-19-research-challenge/'+row['pdf_json_files']+'/'+row['pdf_json_files']+'/pdf_json/'+row['sha']+'.json') as json_file:
            data = json.load(json_file)
            body = format_body(data['body_text'])
            keyword_list=['TB','incidence','age']
            #print (body)
            body = body.replace("\n", " ")

            df.loc[index, 'abstract'] = body.lower()


            
df = df.drop(['pdf_json_files'], axis=1)
df = df.drop(['sha'], axis=1)

df.reset_index(inplace=True)
df.drop("index",axis=1,inplace=True)

df.reset_index(inplace=True)
df.drop("index",axis=1,inplace=True)

nltk.download("punkt")

nltk.download('stopwords')
stops = stopwords.words("english")

snowstem = SnowballStemmer("english")
# Stemming is the process of producing morphological variants of a root/base word. 
# Stemming programs are commonly referred to as stemming algorithms or stemmers. 
# A stemming algorithm reduces the words “chocolates”, “chocolatey”, “choco” to the root word, 
# “chocolate” and “retrieval”, “retrieved”, “retrieves” reduce to the stem “retrieve”.

# portstem = PorterStemmer()

#
# Tokenize both the set of queries and the abstracts.
# This is in preparation for clean up of the texts.
#

usequeries = sent_tokenize("""Smoking, pre-existing pulmonary disease
Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities.
cardiovascular disease , chronic obstructive pulmonary disease and diabetes.
Neonates and pregnant women.
Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.
Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors
Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high risk patient groups
Susceptibility of populations.
Public health mitigation measures that could be effective for control.
immune system disorders.
heart failure.
drinking.
diabetes.

""")
# Although we can ask BERT any question we want, we vectorize a set of questions beforehand
# for the sake of speed. 
# Return a sentence-tokenized copy of text, using NLTK’s recommended sentence tokenizer 

queryarticle = [" ".join([snowstem.stem(removepunc(i.lower())) for i in word_tokenize(x) if i not in stops ]) for x in usequeries]
# Clean up queries

df["usetext"] = df.abstract.apply(lambda x: " ".join([snowstem.stem(i) for i in word_tokenize(removepunc(x.lower())) if not hasNumbers(i) if i not in stops]))
# Clean up abstracts

# Convert both the set of questions and the abstracts 
# into vectors (document-term matrices).  This is for 
# the sake of efficiency so that we can rank the articles 
# that are most similar to the questions

# vectorizer = TfidfVectorizer()
encArticles = vectorizer.fit_transform(df.usetext)
# This fits the data and then transforms it.  "Fit" that
# the vectorizer is being trained on the words of the abstracts.
# After it is trained, the data is transformed into the 
# vector representation of the abtracts.

encQueries = vectorizer.transform(queryarticle)
# It is not necessary to retrain the vectorizer since
# it was just trained in the previous step, so all
# that needs to be done is to convert the questions
# into the vector.

similarity_matrix  = cosine_similarity(encQueries,encArticles)
# determine the similarity of the queries to the abstracts

np.argsort(similarity_matrix[1])[-5:][::-1]
# sort the matrix according to similarity

model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/biobert_v1.1_pubmed_squad_v2")

answers = getanswers("what is the risk for pregnant women?")
answers = getanswers("are smokers at risk")