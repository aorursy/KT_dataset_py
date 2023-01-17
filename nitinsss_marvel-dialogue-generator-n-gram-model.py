from nltk import trigrams

from nltk.tokenize import sent_tokenize

from collections import defaultdict

import glob

import re

import random
import nltk

nltk.download('punkt')
txt_files = glob.glob('../input/marvel-cinematic-universe-dialogue-dataset/*.txt')



print(txt_files[:3], end='\n\n')

print("Total number of movie scripts:", len(txt_files))
marvel_corpus = ''



for file in txt_files:

    with open(file, 'r', encoding="ISO-8859-1") as f:

        text = f.read()

        marvel_corpus += text
marvel_corpus = marvel_corpus.lower()

print('Sample text from the corpus:\n\n', marvel_corpus[9500:10000])
#Placeholder tri-gram model

model = defaultdict(lambda: defaultdict(lambda: 0)) 



#Populating model with trigram counts from corpus

for sentence in nltk.sent_tokenize(marvel_corpus):

    words = sentence.split()

    for w1, w2, w3 in trigrams(words, pad_right=True, pad_left=True):

        model[(w1, w2)][w3] += 1

        

#Transforming count to probabilities

for w1_w2 in model:

    total_count = float(sum(model[w1_w2].values()))

    for w3 in model[w1_w2]:

        model[w1_w2][w3] /= total_count
dict(model['iron', 'man'])
def make_it_pretty(text):

    

    #Capitalizes first letter of the text.

    #Also, capitaliszes first letter of each sentence



    

    p = re.compile('(?<=[\.\?\!\)]\s)\w')  

    matches = re.finditer(p, text)

    sentence_start_inds = [match.span()[0] for match in matches]

    



    sentence_start_inds.append(0)

    

    text = list(text)

    

    for i in sentence_start_inds:

        text[i] = text[i].upper() 

    

    return ''.join(text)
def generate(first_w, second_w, max_words):

    

    text  = [first_w, second_w]

    finished = False



    while not finished:



        w1, w2 = text[-2:]



        probable_words = list(model[w1, w2])



        if (not probable_words) or (len(text) > max_words - 1):

            finished = True



        else:

            new_word = random.choice(probable_words)



            text.append(new_word)

    

    text = ' '.join([t for t in text if t])

    text = make_it_pretty(text)

    print(text)
generate('what', 'if', 100)