import os

import numpy as np



import json

import re



from nltk.stem import WordNetLemmatizer #for stemming

from nltk.collocations import *

from nltk.tokenize import word_tokenize #for tokenization

from nltk.corpus import stopwords #for removing stop words

import nltk



import csv #for saving the data after preprocessing, and loading it later



import gensim #for training the model

import joblib #for saving the model on disk



#plotting libraries

import matplotlib.pylab as plt

from sklearn.manifold import TSNE

import seaborn as sns

import random



import heapq
#preparing list of the files path for later use in reading the files itself

root_path = '/kaggle/input/CORD-19-research-challenge'



files_path = []

for dirname, _, filenames in os.walk(f'{root_path}/document_parses/pmc_json/'):

    for filename in filenames:

        files_path.append(os.path.join(dirname, filename))

        

for dirname, _, filenames in os.walk(f'{root_path}/document_parses/pdf_json/'):

    for filename in filenames:

        files_path.append(os.path.join(dirname, filename))

        

files_path = random.sample(files_path, 10000)
def preprocess(text):

    wordnet_lemmatizer = WordNetLemmatizer()

    stop_words = set(stopwords.words('english'))

    

    text = re.sub('<.*?>','',text) #removing html that is read unintentionally when collecting the data

    text = re.sub('https?:\/\/[^\s]+', '', text) #removing the URLs, as they won't make use to us

    text = " ".join([wordnet_lemmatizer.lemmatize(t) for t in text.split()]) #lemmatizing

    

    #removing punctuations

    for punc in '!"#$%&()*+,./:;<=>?@[\\]^_`{|}~':

        text = ' '.join(text.split(punc))

    

    text = ' '.join([word for word in text.split() if word not in stop_words])#removing stop_words

    

    return text.lower().strip()
#counting the number of occurences of the body sections we desire

body_sections = {}



def body_sections_dic(section_name):

    global body_sections

    if section_name in body_sections: 

        body_sections[section_name] += 1

    else: 

        body_sections[section_name] = 1
#saving only the sections that would matter to us

body_sections_matters = ['abstract','introduction','summary','discussion','conclusion','diagnosis', 'method','treatment','result','concluding','method','background','measures','transmission period','incubation']



def fileRead(file_path):

    with open(file_path) as file:

        content = json.load(file)

        

        body_text = []

        for entry in content['body_text']:

            preprocessed_section = preprocess(entry['section'])

            body_sections_dic(preprocessed_section)

            for i in body_sections_matters:

                if i in preprocessed_section or preprocessed_section == '':

                    body_text.append(entry['text'])

                    break

        return preprocess('\n'.join(body_text))
#removing entries with empty body and repeated ones

duplicates_cnt = 0 #counter repeated or duplicated articles 

emptyBodies_cnt = 0 #counter of articles with empty body 



def removeEmptyRows():

    global duplicates_cnt

    global emptyBodies_cnt

    

    empty_body_ind = []

    

    for indx, file in enumerate(files):

        if(file == ''):

            empty_body_ind.append(indx)

        elif file in files[:indx]:

            duplicates_cnt += 1

            empty_body_ind.append(indx)

            

    emptyBodies_cnt = len(empty_body_ind)

    empty_body_ind.reverse()

    

    for ind in empty_body_ind:

        files.pop(ind)

        files_path.pop(ind)
files_flag = 1 #if the flag is == 1, then we should read the files from their locations, otherwise, it should be loaded from csv file



if files_flag:

    #read the files from their locations

    files = [fileRead(eachfile) for eachfile in files_path]

    removeEmptyRows()

    

    #save a csv file of the output

    with open('articles_body.csv', 'w', newline='') as file:

        writer = csv.writer(file)

        for f in files:

            writer.writerow(f)

else: #load the data from a csv file

    files = []

    with open('../input/titles/articles_body_.csv', encoding='latin-1') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:

            files.append(row[0])
import matplotlib.pylab as plt

from matplotlib.pyplot import figure

figure(num=None, figsize=(20, 10), dpi=80, facecolor='w')

plt.rc('xtick', labelsize=20) 

plt.rc('ytick', labelsize=20)

plt.xticks(rotation=90)



lists = sorted(body_sections.items(),key=lambda x: x[1], reverse=True) # sorted by key, return a list of tuples

x, y = zip(*lists[:20]) # unpack a list of pairs into two tuples



plt.plot(x, y)

plt.show()

figure(num=None, figsize=(6, 6), dpi=80, facecolor='w')

plt.rc('xtick', labelsize=10) 

plt.rc('ytick', labelsize=10)

#plt.xticks(rotation=90)

#EDIT: note that there is only two or three duplicates in all articles, I guess we should use another graph type to show it

x, y = ['Nomber of empty body papers', 'Number of duplicated papers'], [emptyBodies_cnt, duplicates_cnt]



plt.bar(x, y, align='center')

plt.show()

#for insertnig the files into the model they should be tonkenized and tagged, so here is a function to do that

def tagFiles(indx, file):

    tokens = word_tokenize(file)

    return gensim.models.doc2vec.TaggedDocument(tokens, [indx])
#tag and tokenize all files

files_flag = 1 #if the flag is == 1, then we should read the files from their locations, otherwise, it should be loaded from csv file

taggedFiles_flag = 1



if taggedFiles_flag:

    #read the files from their locations

    taggedFiles = [tagFiles(indx, file) for indx, file in enumerate(files)]

    

    #save a csv file of the output

    with open('taggedFiles.csv', 'w', newline='') as file:

        writer = csv.writer(file)

        for tf in taggedFiles:

            writer.writerow([tf])

    

else: #load the data from a csv file

    taggedFiles_flag = []

    with open('../input/titles/taggedFiles.csv', encoding='latin-1') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:

            taggedFiles.append(row)
model_flag = 1



if model_flag:

    #build the model

    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)

    model.build_vocab(taggedFiles)

    model.train(taggedFiles, total_examples=model.corpus_count, epochs=model.epochs)

    

    # Save the model to disk

    joblib.dump(model, 'nlp_model.pkl')

else:

    model = joblib.load('nlp_model.pkl') #Load "model.pkl"
def getMostSimilar(question):

    question = preprocess(question) #QUESTION: what infer_vector does

    query_token = model.infer_vector(word_tokenize(question)) #tokenizing the question

    similar_docs = model.docvecs.most_similar([query_token], topn=20) #get the top 20 similar articles to the question

    documents = [files[similar_doc[0]] for similar_doc in similar_docs] #get the data to return from the top 20 articles or documents

    return documents
#storing the questions on task

questions = ["What do we know about potential risks factors?",

             "what is the effect of Smoking, pre-existing pulmonary disease?",

             "Do co-existing respiratory/viral infections make the virus more transmissible or virulent and other comorbidities?",

             "What is the effect on Neonates and pregnant women?",

             "What are the Socio-economic and behavioral factors on COVID-19?",

             "What is the economic impact of the virus?",

             "What are Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors?",

             "Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups",

             "What are the Susceptibility of populations?",

             "What are the Public health mitigation measures that could be effective for control?"]

#getting the answers of the questions

questionsAnswer = [getMostSimilar(q) for q in questions]
queries_token = [model.infer_vector(word_tokenize(question)) for question in questions]

tsne = TSNE(verbose=1, perplexity=100, random_state=42)

Q_embedded = tsne.fit_transform(queries_token)

Q_embedded.shape

documents_vetors = [model.infer_vector(word_tokenize(file)) for file in files]



D_embedded = tsne.fit_transform(documents_vetors)

D_embedded.shape



#Normalization

mean = (0,0)

for q in Q_embedded:

    mean += q

    

normaliz_factor = mean/len(Q_embedded)



D_embedded = D_embedded * normaliz_factor



# sns settings

sns.set(rc={'figure.figsize':(10,8)})



# colors

palette = sns.hls_palette(40, l=.4, s=.9)



# plot

sns.scatterplot(D_embedded[:,0],D_embedded[:,1], palette=palette)

    

# plot

sns.scatterplot(Q_embedded[:,0], Q_embedded[:,1], palette=palette)

plt.show()
similar_docs_percentage = [model.docvecs.most_similar([query_token], topn=20) for query_token in queries_token]

similar_documents = [documents_vetors[similar_doc[0][0]] for similar_doc in similar_docs_percentage]

tsne = TSNE(verbose=1, perplexity=100, random_state=42)

Q_embedded = tsne.fit_transform(queries_token)

Q_embedded.shape

#TODO:DIAA input the questions token

#Normalization

mean = (0,0)

for q in Q_embedded:

    mean += q

normaliz_factor = mean/len(Q_embedded)

# sns settings

sns.set(rc={'figure.figsize':(10,8)})

# colors

palette = sns.hls_palette(40, l=.4, s=.9) 

for doc in similar_documents:

    D_embedded = tsne.fit_transform(similar_documents)

    D_embedded.shape

    D_embedded = D_embedded * normaliz_factor

    # plot

    sns.scatterplot(D_embedded[:,0],D_embedded[:,1], palette=palette)

# plot

sns.scatterplot(Q_embedded[:,0], Q_embedded[:,1], palette=palette)

plt.show()
def text_summery(text):   

    sentence_list = nltk.sent_tokenize(text)

    preprocessed_list = [preprocess(sent) for sent in sentence_list]

    preprocessed_text = ' '.join(preprocessed_list)



    #count vectorizing

    word_frequencies = {}

    for word in nltk.word_tokenize(preprocessed_text):

        if word not in word_frequencies.keys():

            word_frequencies[word] = 1

        else:

            word_frequencies[word] += 1



    maximum_frequncy = max(word_frequencies.values())



    for word in word_frequencies.keys():

        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)



    sentence_scores = {}

    for sent in sentence_list:

        for word in nltk.word_tokenize(sent.lower()):

            if word in word_frequencies.keys():

                if len(sent.split(' ')) < 30:

                    if sent not in sentence_scores.keys():

                        sentence_scores[sent] = word_frequencies[word]

                    else:

                        sentence_scores[sent] += word_frequencies[word]



    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)



    summary = ' '.join(summary_sentences)

    return summary
finalAnswers = [[text_summery(qa) for qa in questionAnswer] for questionAnswer in questionsAnswer]

print("What do we know about potential risks factors?")

print(finalAnswers)