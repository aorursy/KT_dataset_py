import nltk

import io

import numpy as np

import random

import string

import warnings

warnings.filterwarnings('ignore')
f=open('../input/covid-19-wikipedia/covid 19.txt','r',errors='ignore')

raw=f.read()

raw=raw.lower()

sent=nltk.sent_tokenize(raw)

word=nltk.word_tokenize(raw)


lem=nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):

    return [lem.lemmatize(token) for token in tokens]

punc_dict=dict((ord(punct),None) for punct in string.punctuation)

def LemNormalize(text):

    return LemTokens(nltk.word_tokenize(text.lower().translate(punc_dict)))
greeting_inputs=('hello','hi','greetings','sup',"what's up",'hey')

greeting_output=('hello','hi','greetings','sup',"what's up",'hey','I am glad, you are talking to me!')

def greeting(sentence):

    for word in sentence.split():

        if word.lower() in greeting_inputs:

            return random.choice(greeting_output)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity
def response(user_response):

    chatbot_response=''

    sent.append(user_response)

    Tf=TfidfVectorizer(tokenizer=LemNormalize,stop_words='english')

    tfidf=Tf.fit_transform(sent)

    vals=cosine_similarity(tfidf[-1],tfidf)

    idx=vals.argsort()[0][-2]

    flat=vals.flatten()

    flat.sort()

    req_tfidf=flat[-2]

    if(req_tfidf==0):

        chatbot_response=chatbot_response+"I am sorry, I don't understand you !"

        return chatbot_response

    else:

        chatbot_response=chatbot_response+sent[idx]

        return chatbot_response
flag=True

print('Chatbot: My name is Chatbot. I will answer your queries about Covid-19. If you wanna exit type "bye"')

while(flag==True):

    user_response=input('Please ask me your query!')

    user_response=user_response.lower()

    if (user_response!='bye'):

        if(user_response=='thanks' or user_response=='thank you'):

            flag=False

            print('Chatbot: You are welcome')

        else:

            if(greeting(user_response)!=None):

                print('Chatbot: '+greeting(user_response))

            else:

                print('Chatbot: ',end='')

                print(response(user_response))

                sent.remove(user_response)

    else:

        flag=False

        print('Chatbot: Bye! Have a nice day!')