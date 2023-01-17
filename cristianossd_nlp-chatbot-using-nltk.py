# https://medium.com/analytics-vidhya/building-a-simple-chatbot-in-python-using-nltk-7c8c8215ac6e

import nltk
import numpy as np
import random
import string
FILE_PATH = '../input/chatbot.txt'

f = open(FILE_PATH, 'r', errors='ignore')
raw = f.read()
raw = raw.lower()
nltk.download('punkt')
nltk.download('wordnet')

sentence_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

[sentence_tokens[:2], word_tokens[:2]]
lemmer = nltk.stem.WordNetLemmatizer()

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def lem_tokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def lem_normalize(text):
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
GREETING_INPUTS = ('hello', 'hi', 'greetings', 'sup', 'what\'s up', 'hey',)
GREETING_RESPONSES = ['hi', 'hey', '*nods*', 'hi there', 'hello', 'I am glad! You are talking to me']

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def response(user_response):
    robo_response = ''
    sentence_tokens.append(user_response)
    
    vectorizer = TfidfVectorizer(tokenizer=lem_normalize, stop_words='english')
    tfidf = vectorizer.fit_transform(sentence_tokens)
    
    values = cosine_similarity(tfidf[-1], tfidf)
    idx = values.argsort()[0][-2]
    flat = values.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if req_tfidf is 0:
        robo_response = '{} Sorry, I don\'t understand you'.format(robo_response)
    else:
        robo_response = robo_response + sentence_tokens[idx]
    return robo_response
flag = True
print('BOT: My name is Robo, I will answer your questions about chatbots. If you want to exit, type Bye')

interactions = [
    'hi',
    'what is chatbot?',
    'describe its design, please',
    'what about alan turing?',
    'and facebook?',
    'sounds awesome',
    'bye',
]

while flag:
    user_response = interactions.pop(0)
    print('USER: {}'.format(user_response))
    if user_response is not 'bye':
        if user_response is 'thanks' or user_response is 'thank you':
            flag = False
            print('BOT: You are welcome...')
        else:
            if greeting(user_response) is not None:
                print('ROBO: {}'.format(greeting(user_response)))
            else:
                print('ROBO: ', end='')
                print(response(user_response))
                sentence_tokens.remove(user_response)
    else:
        flag = False
        print('BOT: bye!')

