#Loading in the necessary libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import spacy

import string

import re

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#       print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Loading in the metadata (for the abstracts) and creating a copy so I don't have to constantly reload this data.

all_source_metadata = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

metadata = all_source_metadata.copy()
#Subsetting for observations with actual text in the abstract column

#Then I reset the index and dropped the old index column...I forgot why

metadata = metadata[metadata["abstract"].notnull()]

metadata = metadata.reset_index(drop = True)



metadata = metadata[metadata["title"].notnull()]

metadata = metadata.reset_index(drop = True)
#Get shape of metadata 

metadata.shape
#Getting a subset of the data for processing because of computational limitations

reasonable = metadata.sample(n=75000,random_state=0)
#Defining English Stopwords and other additional stopwords

stop_words=set(stopwords.words("english"))

special_stop_words = ["BACKGROUND", "METHODS", "CONCLUSION", "RESULTS", ":", "Abstract", "abstract", "ABSTRACT", "CONCLUSIONS",\

                          "SUPPLEMENTARY", "MATERIAL", "OBJECTIVE", "IMPORTANCE", "METHODOLOGY", "METHODOLOGYPRINCIPAL", "DESIGN",\

                          "Background", "PURPOSE", "MATERIALS", "INTRODUCTION", "ELECTRONIC"]

    

# Initialize spacy 'en' model, keeping only tagger component needed for lemmatization

nlp = spacy.load('en', disable=['parser', 'ner'])



#Defining the punctuation for this project (again, can be done outside of this function)

special_punc = [x for i,x in enumerate(string.punctuation) if x!= "-"]



#Defining a porter stemmer object

#ps = PorterStemmer()
#Creating a function to clean all abstracts: doc_proc

def doc_proc(text):

    

    '''

    doc_proc is a function that takes in a text item (in this case, a scientific journal abstract) and conducts all pre-processing before 

    further analysis. The function automatically does all of the above individual preprocessing steps. This function outputs an object after

    word lemmatization (lemmatized_words). This can be applied over a dataframe through the apply functions.

    

    Keyword Arguments:

    text: the input text to be processed. Must be a continuous string that can be tokenized into words.

    

    '''

    

    ###Starting the text manipulation process###

    

    #Tokenizing into words

    tokenized_word=word_tokenize(text)



    #Filtering out special stopwords first and rejoining the words back together into sentences

    special_filtered_word=[]

    for w in tokenized_word:

        if w not in special_stop_words:

            special_filtered_word.append(w)   

    s = " "

    special_filtered_word = s.join(special_filtered_word)

   

    #Word Tokenizing all text

    text = word_tokenize(special_filtered_word)

    

    #Filtering out all stopwords

    stopword_filtered = []

    for w in text:

        if w.lower() not in stop_words:

            stopword_filtered.append(w)

    

    

    #Filtering out all punctuation, then rejoining all of the resulting sublists into one list

    punc_filtered = [''.join(c for c in s if c not in special_punc) for s in stopword_filtered]

    punc_filtered = [s for s in punc_filtered if s]

    

    #Filtering out stand-alone numbers

    numeric_filtered = [term for term in punc_filtered if term.isdecimal() == False]

    

    #Filtering out any tokens that contain no letters

    numeric_filtered2 = [term for term in numeric_filtered if re.search('[a-zA-Z]', term) is not None]

    

    #Filtering out any tokens left over that are length one (i.e. "C" or "f")

    single_stripped = [term for term in numeric_filtered2 if len(term) > 1]

    

    #Filtering out any tokens left two letters long and lower case (i.e. "sg")

    lower_double_stripped = [term for term in single_stripped if len(term) is not 2 or term.isupper() == True]

    

    #Lower casing all tokens that have a single capital letter and are longer than 2 characters long

    lower_cased_nonsci = []

    for term in lower_double_stripped:

        if len(term) > 2 and sum(1 for c in term if c.isupper()) <= 1:

            lower_cased_nonsci.append(term.lower())

        else:

            lower_cased_nonsci.append(term)

    

    #Removing all specialty punctuation outside of default ones

    for i in range(len(lower_cased_nonsci)):

        if len(lower_cased_nonsci[i]) == 1 and lower_cased_nonsci[i].lower() not in list(string.ascii_lowercase):

            lower_cased_nonsci[i] = ''        

    while('' in lower_cased_nonsci) : 

        lower_cased_nonsci.remove('') 

    

    #Removing all tokens that end in a hyphen

    suffix_hyphen_stripped = [term for term in lower_cased_nonsci if term.endswith("-") == False]

    

    #STEMMING METHOD####################################################################

    # Parse the sentence using the loaded 'en' model object `nlp`

    #final_words = []

    #for term in suffix_hyphen_stripped:

    #    final_words.append(ps.stem(term))

    ####################################################################################

    #LEMMATIZATION METHOD###############################################################

    #

    final_words = []

    for term in suffix_hyphen_stripped:

        doc = nlp(term)

        final_words.append([token.lemma_ for token in doc])

    

    #Recombining sublists containing more than one token

    recombined1 = []

    for term_list in final_words:

        if len(term_list) > 1:

            recombined1.append(["".join(term_list)])

        else:

            recombined1.append(term_list)

    

    #Recombining all sublists into a main list

    recombined2 = [item for sublist in recombined1 for item in sublist]

    #####################################################################################        

    return(recombined2)
# Code to test run times for different versions of doc_proc



#import time

#start_time = time.time()

#test = doc_proc(metadata["abstract"][18])

#end_time = time.time()



#end_time-start_time
#Applying doc_proc to the small dataset into a new column "abstract2" 

reasonable["abstract2"] = reasonable['abstract'].apply(doc_proc)
#Checking the results

print(metadata["abstract"][0])

print('----------------------------------------------')

print(doc_proc(metadata["abstract"][0]))
#Exporting the data for use in other kernels for modeling

reasonable.to_csv("abstract_cleaned.csv", index = False)