!pip install pyttsx3 #used for speech output

!pip install SpeechRecognition #used for speech input

!pip install pyspellchecker #spelling checker
import nltk

from spellchecker import SpellChecker

import urllib

import bs4 as bs

from nltk.corpus import stopwords

nltk.download('stopwords')

nltk.download('punkt')

nltk.download('wordnet')

import warnings

warnings.filterwarnings("ignore")

import random

from sklearn.metrics.pairwise import cosine_similarity

import random

import string 

import pandas as pd

import requests

from bs4 import BeautifulSoup

import pandas as pd

import numpy as np

import os

from pandas import DataFrame

import pyttsx3 

import speech_recognition as sr

from nltk.stem import WordNetLemmatizer

nltk.download('popular', quiet=True) 
page1=requests.get('https://www.timeanddate.com/weather/india') #switch on the Internet option from right side window




def temp(topic):

    

    page = page1

    soup = BeautifulSoup(page.content,'html.parser')



    data = soup.find(class_ = 'zebra fw tb-wt zebra va-m')



    tags = data('a')

    city = [tag.contents[0] for tag in tags]

    tags2 = data.find_all(class_ = 'rbi')

    temp = [tag.contents[0] for tag in tags2]



    indian_weather = pd.DataFrame(

    {

        'City':city,

        'Temperature':temp

    }

    )

    

    df = indian_weather[indian_weather['City'].str.contains(topic.title())] 

    

    return (df['Temperature'])
def wiki_data(topic):

    

    topic=topic.title()

    topic=topic.replace(' ', '_',1)

    url1="https://en.wikipedia.org/wiki/"

    url=url1+topic



    source = urllib.request.urlopen(url).read()



    # Parsing the data/ creating BeautifulSoup object

    soup = bs.BeautifulSoup(source,'lxml')



    # Fetching the data

    text = ""

    for paragraph in soup.find_all('p'):

        text += paragraph.text



    import re

    # Preprocessing the data

    text = re.sub(r'\[[0-9]*\]',' ',text)

    text = re.sub(r'\s+',' ',text)

    text = text.lower()

    text = re.sub(r'\d',' ',text)

    text = re.sub(r'\s+',' ',text)

    

    

    return (text)
def rem_special(text):

    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

    return(text.translate(remove_punct_dict))



sample_text="I am sorry! I don't understand you."

rem_special(sample_text)
from nltk.stem import PorterStemmer 

from nltk.tokenize import word_tokenize 



def stemmer(text):

    words = word_tokenize(text) 

    for w in words:

        text=text.replace(w,PorterStemmer().stem(w))

    return text



stemmer("He is Eating. He played yesterday. He will be going tomorrow.")
lemmer = WordNetLemmatizer()

def LemTokens(tokens):

    return [lemmer.lemmatize(token) for token in tokens]



sample_text="rocks corpora better" #default noun

LemTokens(nltk.word_tokenize(sample_text))
from nltk.tokenize.toktok import ToktokTokenizer

tokenizer = ToktokTokenizer()



stopword_list = nltk.corpus.stopwords.words('english')



def remove_stopwords(text, is_lower_case=False):

    tokens = tokenizer.tokenize(text)

    tokens = [token.strip() for token in tokens]

    if is_lower_case:

        filtered_tokens = [token for token in tokens if token not in stopword_list]

    else:

        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]

    filtered_text = ' '.join(filtered_tokens)    

    return filtered_text



remove_stopwords("This is a sample sentence, showing off the stop words filtration.")
import spacy 

spacy_df=[]

spacy_df1=[]

df_spacy_nltk=pd.DataFrame()

nlp = spacy.load('en_core_web_sm') 

  

# Process whole documents 

sample_text = ("The heavens are above. The moral code of conduct is above the civil code of conduct") 

doc = nlp(sample_text) 

  

# Token and Tag 

for token in doc:

    spacy_df.append(token.pos_)

    spacy_df1.append(token)





df_spacy_nltk['origional']=spacy_df1

df_spacy_nltk['spacy']=spacy_df

#df_spacy_nltk
import spacy 

nlp = spacy.load('en_core_web_sm') 



def ner(sentence):

    doc = nlp(sentence) 

    for ent in doc.ents: 

        print(ent.text, ent.label_) 

    



sentence = "A gangster family epic set in 1919 Birmingham, England; centered on a gang who sew razor blades in the peaks of their caps, and their fierce boss Tommy Shelby."

ner(sentence) 
from textblob import TextBlob



def senti(text):

    testimonial = TextBlob(text)

    return(testimonial.polarity)



sample_text="This apple is good"

print("polarity",senti(sample_text))

sample_text="This apple is not good"

print("polarity",senti(sample_text))


spell = SpellChecker()





def spelling(text):

    splits = sample_text.split()

    for split in splits:

        text=text.replace(split,spell.correction(split))

        

    return (text)

    

    

sample_text="hapenning elephnt texte luckno sweeto"

spelling(sample_text)
#TOkenisation

print(nltk.sent_tokenize("Hey how are you? I am fine."))

print(nltk.word_tokenize("Hey how are you? I am fine."))
city = {} 

city["bangalore"]=["bangalore","bengaluru","blr"]

city["lucknow"] = ["lucknow", "lko"]

city["delhi"]=["new delhi","ndls","delhi"]
def city_name(sentence):

    for word in sentence.split():

        for key, values in city.items():

            

            if word.lower() in values:

                return(key)

                

    

city_name("blr")
from sklearn.feature_extraction.text import TfidfVectorizer

documentA = 'This is about Messi'

documentB = 'This is about TFIDF'

vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform([documentA, documentB])

feature_names = vectorizer.get_feature_names()

dense = vectors.todense()

denselist = dense.tolist()

df = pd.DataFrame(denselist, columns=feature_names)

df
def LemNormalize(text):

    text=rem_special(text) #remove special char

    text=text.lower() # lower case

    text=remove_stopwords(text) # remove stop words

    

    return LemTokens(nltk.word_tokenize(text))
#Generating answer

def response(user_input):

    

    ToGu_response=''

    sent_tokens.append(user_input)

    

    

    

    word_vectorizer = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')   

    all_word_vectors = word_vectorizer.fit_transform(sent_tokens)  

    

   

    similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors) 

    idx=similar_vector_values.argsort()[0][-2]

    



    matched_vector = similar_vector_values.flatten()

    matched_vector.sort()

    vector_matched = matched_vector[-2]

    

    if(vector_matched==0):

        ToGu_response=ToGu_response+"I am sorry! I don't understand you."

        return ToGu_response

    else:

        ToGu_response = ToGu_response+sent_tokens[idx]

        return ToGu_response

#topic=str(input("Please enter the city name you want to ask queries for: ")) #Uncomment this line to take input from command prompt or jupyter notebook.

topic="bangalore" # sample city



topic=city_name(topic) # fetch city name in case of invalid input or any discrepancy in the city name



text=wiki_data(topic) # fetch wiki data about city



sent_tokens = nltk.sent_tokenize(text)# converts to list of sentences 

word_tokens = nltk.word_tokenize(text)# converts to list of words

weather_reading=(temp(topic)).iloc[0] #fetch weather
# greetings Keyword matching

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey")

GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]



def greeting(sentence):

    for word in sentence.split():

        if word.lower() in GREETING_INPUTS:

            return random.choice(GREETING_RESPONSES)
PLACES_INPUTS = ("places", "monuments", "buildings","places", "monument", "building")



import spacy 

nlp = spacy.load('en_core_web_sm') 



def ner(sentence):

    places_imp=""

    doc = nlp(sentence) 

    for ent in doc.ents: 

        if (ent.label_=="FAC"):

            #print(ent.text, ent.label_) 

            places_imp=places_imp+ent.text+","+" "

            

    return(places_imp)

    



places_imp=ner(text) 





s=places_imp

l = s.split() 

k = [] 

for i in l: 

  

    # If condition is used to store unique string  

    # in another list 'k'  

    if (s.count(i)>1 and (i not in k)or s.count(i)==1): 

        k.append(i) 



PLACES_RESPONSES = ' '.join(k)



def places(sentence):

    for word in sentence.split():

        if word.lower() in PLACES_INPUTS:

            return (PLACES_RESPONSES)
WEATHER_INPUTS = ("weather", "temp", "temperature")



WEATHER_RESPONSES =weather_reading



def weather(sentence):

    for word in sentence.split():

        if word.lower() in WEATHER_INPUTS:

            return (WEATHER_RESPONSES)
continue_dialogue=True

print("ToGu: Hello")

#speak("Hello")



while(continue_dialogue==True):

    user_input = input("User:")

    user_input=user_input.lower()

    user_input=spelling(user_input) #spelling check

    print("Sentiment score=",senti(user_input)) #sentiment score

    

    if(user_input!='bye'):

        if(user_input=='thanks' or user_input=='thank you' ):

            print("ToGu: You are welcome..")

            #speak(" You are welcome")

            

        else:

            if(greeting(user_input)!=None):

                tmp=greeting(user_input)

                print("ToGu: "+tmp)

                #speak(tmp)

                

            elif(weather(user_input)!=None):

                tmp=weather(user_input)

                print("ToGu: "+tmp)

                #speak(tmp)

                

                

            elif(places(user_input)!=None):

                tmp=places(user_input)

                print("ToGu: Important places are "+tmp)

                #speak("Important places are")

                #speak(tmp)

                

            else:

                print("ToGu: ",end="")

                temp=response(user_input)

                print(temp) 

                #speak(temp)

                sent_tokens.remove(user_input)

                



    else:

        continue_dialogue=False

        print("ToGu: Goodbye.")

        #speak("goodbye")

        




