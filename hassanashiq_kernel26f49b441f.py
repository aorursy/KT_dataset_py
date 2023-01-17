from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words

import nltk
import os
import string
import numpy as np
import copy
import pandas as pd
import pickle
import re
import math
with open('IR Project/CISI.ALL') as CISI_file:
    lines = ""
    for l in CISI_file.readlines():
        lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
    lines = lines.lstrip("\n").split("\n")
    
    
print("Done")
doc_set = {}
doc_id = ""
doc_text = ""
for l in lines:
    if l.startswith(".I"):
        doc_id = l.split(" ")[1].strip()
    elif l.startswith(".X"):
        doc_set[doc_id] = doc_text.lstrip(" ")
        doc_id = ""
        doc_text = ""
    else:
        doc_text += l.strip()[3:] + " " # The first 3 characters of a line can be ignored.

# Print something to see the dictionary structure, etc.
print(f"Number of documents = {len(doc_set)}" + ".\n")

doc_set["3"]
with open('IR Project/datasets.QRY') as f:
    lines = ""
    for l in f.readlines():
        lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
    lines = lines.lstrip("\n").split("\n")
    
qry_set = {}
qry_id = ""
for l in lines:
    if l.startswith(".I"):
        qry_id = l.split(" ")[1].strip()
    elif l.startswith(".W"):
        qry_set[qry_id] = l.strip()[3:]
        qry_id = ""
    
# Print something to see the dictionary structure, etc.
print(f"Number of queries = {len(qry_set)}" + ".\n")
print("Query # 2 : ", qry_set["2"]) # note that the dictionary indexes are strings, not numbers. 
qry_set["1"]
def convert_lower_case(data):
    return np.char.lower(data)
def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text
def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data
def remove_apostrophe(data):
    return np.char.replace(data, "'", "")
def stemming(data):
    stemmer= PorterStemmer()
    
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text
def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text
def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    #data = convert_numbers(data)
    data = stemming(data)
    data = remove_punctuation(data)
    #data = convert_numbers(data)
    data = stemming(data) #needed again as we need to stem the words
    data = remove_punctuation(data) #needed again as num2word is giving few hypens and commas fourty-one
    data = remove_stop_words(data) #needed again as num2word is giving stop words 101 - one hundred and one
    return data
processed_set={}
proc_token_id=""
proc_token_text=""

for i in doc_set:
    doc_token_id=i
    processed_set[doc_token_id]=preprocess(doc_set[str(i)])
print("done")
    
doc_set["2"]
processed_set["2"]
tokens_set={}
doc_token_id=""
doct_token_text=""

for i in processed_set:
    doc_token_id=i
    tokens_set[doc_token_id]=word_tokenize(processed_set[str(i)])
print("done")
    
np.array(tokens_set["2"]).T
DF = {}

for i in range(len(tokens_set)):
    tokens = tokens_set[str(i+1)]
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}
for i in DF:
    DF[i] = len(DF[i])
DF

total_vocab_size = len(DF)
total_vocab_size

total_vocab = [x for x in DF]
N=len(total_vocab)
N
def doc_freq(word):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c
doc = 0
N=len(tokens_set)
tf_idf = {}

for i in range(len(tokens_set)):
    if(i>0):
        tokens = tokens_set[str(i)]
    
    counter = Counter(tokens)
    words_count = len(tokens)
    
    for token in np.unique(tokens):
        
        tf = counter[token]/words_count
        df = doc_freq(token)
        idf = np.log((N+1)/(df+1))
        
        tf_idf[doc,token] = tf*idf
    doc += 1

print("tf-idf done")
tf_idf
def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim
D = np.zeros((N, total_vocab_size))   #total_vocab_size is the length of DF
for i in tf_idf:
    try:
        ind = total_vocab.index(i[1])
        D[i[0]][ind] = tf_idf[i]
    except:
        pass
def gen_vector(tokens):

    Q = np.zeros((len(total_vocab)))
    
    counter = Counter(tokens)
    words_count = len(tokens)

    query_weights = {}
    
    for token in np.unique(tokens):
        
        tf = counter[token]/words_count
        df = doc_freq(token)
        idf = math.log((N+1)/(df+1))

        try:
            ind = total_vocab.index(token)
            Q[ind] = tf*idf
        except:
            pass
    return Q
def cosine_similarity(k, query):
    
    preprocessed_query = preprocess(query)
    tokens = word_tokenize(str(preprocessed_query))
    
    #print("\nQuery:", query)
    
    d_cosines = []
    
    query_vector = gen_vector(tokens)
    
    for d in D:
        d_cosines.append(cosine_sim(query_vector, d))
        
    out = np.array(d_cosines).argsort()[-k:][::-1]
    
    
    #print("Most similar Dpocuments-IDs : ")
    
    #print(out)
    
    return out

Q = cosine_similarity(5,qry_set["3"])

print('Related documents to given query :\n \"', qry_set["3"] , '" \n' )
Q
rel_set = {}
with open('IR Project/Datasets.REL') as f:
    for l in f.readlines():
        qry_id = l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[0]
        doc_id = int(l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[-1])
        if qry_id in rel_set:
            rel_set[qry_id].append(doc_id)
        else:
            rel_set[qry_id] = []
            rel_set[qry_id].append(doc_id) 
    
    
print(rel_set["3"]) # note that the dictionary indexes are strings, not numbers. 
    
precision_list=[]
recall_list=[]
precision_list=[]
recall_list=[]
accuracy_list=[]

for i in range(1,len(doc_set)):
    try:
        result_from_cosine=cosine_similarity(6 , qry_set[str(i)]).tolist()
        result_from_ground_truth=rel_set[str(i)]
        
        true_Positive=len(set(result_from_cosine) & set(result_from_ground_truth)) #set(a) & set(b) gives us intersection between a and b
        false_Positive=len(np.setdiff1d(result_from_cosine , result_from_ground_truth))
        false_Negative=len(np.setdiff1d(result_from_ground_truth , result_from_cosine))
        true_negative= ( len(doc_set) -  (true_Positive + false_Negative + false_Positive) )
        #print("true psotive",true_Positive)
        #print("false negative",false_Negative)
        
        try:
            precission= (true_Positive) / ( true_Positive + false_Positive )
            recall= (true_Positive) / (true_Positive + false_Negative)
            
            accuracy= ( true_negative + true_Positive ) / (  true_negative + true_Positive + false_Negative +false_Positive)
           
        except ZeroDivisionError:
            pass

        precision_list.append(precission)
        recall_list.append(recall)
        accuracy_list.append(accuracy)
        
        
        
    except KeyError:
        pass
    
average_precision=sum(precision_list)
average_recall=sum(recall_list)
Accuracy= sum(accuracy_list)
F_Measure = (2 * average_precision * average_recall) / (average_precision + average_recall)
print("Average Precision is : ", average_precision)
print("Average Recall is : ", average_recall)
print("F-score is : " ,F_Measure)
print("Accuracy : " ,Accuracy)

query=input("Enter your query here : ")

Q=cosine_similarity(10,query)

print("\n\nEntered Query is : " , query)
print("\n\nRelated Documents IDs are : ", Q)
print("\nDo you want to retrive the document ? \n press Y to see all related docs \n Press S to see a single document with given id \n Press N to exit ")

entered_option=input()
    
if entered_option == "Y":

    print("\n\n*** You are in All Document Retriveal Mood ***\n\n")

    for i in range(len(Q)):
            print("\n\nDoc-Id :", Q[i] , "\n\t" ,doc_set[str(Q[i])])
           
elif entered_option == "S":
    print("Enter your desired document ID : ")
    doc_id=input()
    print("Doc-Id : ", doc_id, "\n\t" ,doc_set[doc_id])
        

else:
    print("Thank you for using our Information System")
    print("Hassan Ashiq & Usman Ali Abbasi")
qry_set["3"]
