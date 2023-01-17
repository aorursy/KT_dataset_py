#Problem---Clean the unstructured data.

#Solution---We will use python's string operation to clean the data



#Creating unstructured Text

text_data=["Interrobang. By Aishwarya Henriette         ",

           "Parking And Going. By Karl Gautier",

           "          Today Is The night. By Jarek Prakash"]
#Stripping the whitespaces or removing the unnecessary spaces

strip_whitespace =[string.strip() for string in text_data]
#Printing the data

strip_whitespace
#Removing unwanted characters or periods

remove_periods =[string.replace(".","") for string in strip_whitespace]
#printing the text

remove_periods
#Applying custom transformation functions

#creating function to capitalize

def capitalizer(string:str)->str:

    return string.upper()



#Apllying the function

[capitalizer(string) for string in remove_periods]
#Using Regular Expression(RE) to apply custom transformation

#importing regular expression library

import re



#creating function

def replace_letters_with_X(string:str)->str:

    return re.sub(r"[a-zA-Z]","X",string)



    

#Applying the function

[replace_letters_with_X(string) for string in remove_periods]
#Problem---The data is in form of HTML and extract the information from it.

#Solution---We will use Beautiful Soup to extract info.



#importing beautiful soup library

from bs4 import BeautifulSoup



#creating some HTML code

html = """

<div class='full_name'><span style='font-weight:bold'>

Masego</span> Azra</div>"

"""



#parsing HTML

soup=BeautifulSoup(html,"lxml")



#Finding the div with the class "full_name" and display text

soup.find("div",{"class" : "full_name"}).text
#Problem---Remove the punctuations from the text

#Solution---We will use python's translate for remving punctuation



#importing libraries

import unicodedata

import sys



#creating text 

text_data = ['Hi!!!! I. Love. This. Song....',

             '10000% Agree!!!! #LoveIT',

             'Right?!?!']



#Creating dictionary of punctuation characters

punctuation=dict.fromkeys(i for i in range(sys.maxunicode)

                          if unicodedata.category(chr(i)).startswith('P'))



#Removing punctuation

[string.translate(punctuation) for string in text_data]
#Problem---Implement tokenization

#Solution---We will ue Natural Language Processing Library NLTK(Natural Language Toolkit)



#importing nltk library

from nltk.tokenize import word_tokenize



#creating text

string = "The science of today is the technology of tomorrow"



#Tokenizing the words

word_tokenize(string)
#Tokenizing in sentences



#importing library

from nltk.tokenize import sent_tokenize



#creating text

string = "The science of today is the technology of tomorrow. Tomorrow is today."



#tokenizing sentence

sent_tokenize(string)
#Problem---Remove the stopwords from the text

#Solution---We will use NLTK



#importing NLTK library

from nltk.corpus import stopwords



#Creating word tokens

tokenized_words = ['i',

                   'am',

                   'going',

                   'to',

                   'go',

                   'to',

                   'the',

                   'store',

                   'and',

                   'park']



#loading stop words

stop_words=stopwords.words('english')



#removing stop words

[word for word in tokenized_words if word not in stop_words]
#List of stop words

stop_words
#Problem---Convert the tokenized words into root words

#Solution---We will use nltk to stem them



#importing nltk library

from nltk.stem.porter import PorterStemmer



#creating tokenized words

tokenized_words = ['i', 'am', 'humbled', 'by', 'this', 'traditional', 'meeting']



#creating stemmer

porter =PorterStemmer()



#Applying stemmer

[porter.stem(word) for word in tokenized_words]
#Problem---Tag each word with part of speech tag

#Solution---We will use NLTK's pretrained tagger



#importing libraries

from nltk import pos_tag

from nltk import word_tokenize



#creating text

text_data ="Koro Muri loves outdoor running"



#using pretrained tagger after tokenizing

text_tagged=pos_tag(word_tokenize(text_data))



#Displaying part of speech

text_tagged
#We can find certain part of speech

[word for word,tag in text_tagged if tag in ['NN','NNS','NNP','NNPS']]
#Problem---Create a bag-of-wordds matrix of th text

#Solution---We will use scikit learn library



#importing libraries

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer



#creating text

text_data = np.array(['I love India. India!',

                      'Sweden is best',

                      'Germany beats both'])



#creating a bag of words

count=CountVectorizer()

bag_of_words=count.fit_transform(text_data)



#displaying data

bag_of_words
#viewing the matrix

bag_of_words.toarray()
#showing feature name

count.get_feature_names()
#Problem---Show the importance of the word

#Solution---We will use sklearn to show the importance of word



#Importing numpy and sklearnlibrary

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer



#creating text

text_data = np.array(['I love India. India!',

                      'Sweden is best',

                      'Germany beats both'])



#creating the tf-idf feature matrix

tfidf=TfidfVectorizer()

feature_matrix=tfidf.fit_transform(text_data)
#Displaying sparse matrix

feature_matrix
#displaying dense matrix

feature_matrix.toarray()
#Displaying the feature name

tfidf.vocabulary_