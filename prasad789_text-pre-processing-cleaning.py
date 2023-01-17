# numpy and pandas for data manipulation

import numpy as np

import pandas as pd 



# File system manangement

import os



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')



# matplotlib and seaborn for plotting

import matplotlib.pyplot as plt

import seaborn as sns



from tqdm import tqdm

import re
restaurant_reviews=pd.read_csv("/kaggle/input/zomato-restaurants-hyderabad/Restaurant reviews.csv")

restaurant_reviews.head()
def decontracted(phrase):

    # replace '

    phrase = phrase.replace('’', '\'')

    # specific

    phrase = re.sub(r"won't", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)

    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase
print('---------------------------------------- with contraction word --------------------------------------------')

print(restaurant_reviews['Review'][40])

print('---------------------------------------- without contraction word ----------------------------------------')

print(decontracted(restaurant_reviews['Review'][40]))
def line_breaks(phrase_1):

    phrase_1 = phrase_1.replace('\\r', ' ')

    phrase_1 = phrase_1.replace('\\"', ' ')

    phrase_1 = phrase_1.replace('\n', ' ')

    return phrase_1
print('---------------------------------------- with linebreak word --------------------------------------------')

print(restaurant_reviews['Review'][74])

print('---------------------------------------- without linebreak word ----------------------------------------')

print(line_breaks(restaurant_reviews['Review'][74]))
def remove_special_character(phase_2):

    phase_2 = re.sub('[^A-Za-z0-9]+', ' ', phase_2)

    return(phase_2)
print('---------------------------------------- with special character word --------------------------------------------')

print(restaurant_reviews['Review'][1478])

print('---------------------------------------- without special character word ----------------------------------------')

print(remove_special_character(restaurant_reviews['Review'][1478]))
def remove_continues_char(s) :

    senta = ""

    for i in s.split():

        #print(i)

        if len(i) <= 15:

            

            senta += i

            senta += ' '

        else:

            pass

    return(senta)
print('---------------------------------------- with unwanted long word word --------------------------------------------')

print(restaurant_reviews['Review'][238])

print('---------------------------------------- without unwanted long word word ----------------------------------------')

print(remove_continues_char(restaurant_reviews['Review'][238]))
# we are removing the words from the stop words list: 'no', 'nor', 'not'

stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\

            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \

            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\

            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \

            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \

            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \

            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\

            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\

            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\

            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \

            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \

            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\

            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\

            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \

            'won', "won't", 'wouldn', "wouldn't"]
def stop_words(phrase):

    after = ' '.join(e for e in phrase.split() if e.lower() not in stopwords)

    return(after)
print('---------------------------------------- with unwanted long word word --------------------------------------------')

print(restaurant_reviews['Review'][1])

print('---------------------------------------- without unwanted long word word ----------------------------------------')

print(stop_words(restaurant_reviews['Review'][1]))
preprocessed_titles = []

# tqdm is for printing the status bar

for review in tqdm(restaurant_reviews['Review'].values):

    review_1 = str(review)

    review_1 = decontracted(review_1)

    review_1 = line_breaks(review_1)

    review_1 = remove_special_character(review_1)

    review_1 = remove_continues_char(review_1)

    review_1 = stop_words(review_1)

    preprocessed_titles.append(review_1.lower().strip())
print('---------------------------------------- Old review text --------------------------------------------')

print(restaurant_reviews['Review'][1478])

print('---------------------------------------- review text after preprocessing ----------------------------------------')

print(preprocessed_titles[1478])
print('---------------------------------------- Old review text --------------------------------------------')

print(restaurant_reviews['Review'][2078])

print('---------------------------------------- review text after preprocessing ----------------------------------------')

print(preprocessed_titles[2078])
print('---------------------------------------- Old review text --------------------------------------------')

print(restaurant_reviews['Review'][26])

print('---------------------------------------- review text after preprocessing ----------------------------------------')

print(preprocessed_titles[26])