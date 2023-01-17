import nltk

from nltk.tokenize import RegexpTokenizer

from nltk.stem import PorterStemmer

from nltk.corpus import stopwords

import re



stemmer = PorterStemmer()
sentence = "Welcome to DSW Youtube Channel 10<ok>.DSW stAnds for #Data Science @Wizards.We are welcoming you to Our family"
#Converting to lowercase to make it unified

sentence = sentence.lower()  

print (sentence)
#Compiling regular expression

cleanr = re.compile('<.*?>')
#Substituting compiled pattern with blank space in the string

cleantext = re.sub(cleanr, '', sentence)  

print (cleantext)
#Substituting numbers with blank space

rem_num = re.sub('[0-9]+', '', cleantext)  #Substituting numbers with blank space

print (rem_num)
#Creating a reference variable for Class RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')  

    

#Using tokenize method on processed data

tokens = tokenizer.tokenize(rem_num) 



print (tokens)
#Removing stopwords

filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')] 



print (filtered_words)
#Stemming words after tokenizing

stem_words=[stemmer.stem(w) for w in filtered_words]  



print (stem_words)