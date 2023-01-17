!pip install thesaurus
import os

import spacy

import random

from thesaurus import Word

import nltk 

from nltk.corpus import wordnet 

import en_core_web_sm

import re
percent = 50 #input('enter percentage for 0 to 100:')
def synalter_Noun_Verb(word,al,POS):

    max_temp = -1

    flag = 0

    for i in a1:

        try:

            w1 = wordnet.synset(word+'.'+POS+'.01') 

            w2 = wordnet.synset(i+'.'+POS+'.01') # n denotes noun 

            if(max_temp<w1.wup_similarity(w2)):

                max_temp=w1.wup_similarity(w2)

                temp_name = i

                flag =1

        except:

            f = 0

            

    if flag == 0:

        max1 = -1.

        nlp = en_core_web_sm.load()

        for i in a1:

            j=i.replace(' ', '')

            tokens = nlp(u''+j)

            token_main = nlp(u''+word_str)

            for token1 in token_main:

                if max1<float(token1.similarity(tokens)):

                    max1 = token1.similarity(tokens)

                    value = i

        max1 = -1.

        return value 

    else:

        return temp_name
synonyms = [] 

antonyms = []   

all_files = os.listdir("../input/")

txt_files = filter(lambda x: x[-4:] == '.txt', all_files)

print(txt_files)

for i in txt_files:

    textfile = i

    print("Input File: "+ textfile)

    print(" ")

    path = '../input/'+textfile

    exists = os.path.isfile(path)

    if exists: 

        file_open = open(path,"r")

        text = file_open.read()

        output_text = text

        print("Sentence: "+text)

        words = text.split()

        counts = {}

        for word in words:

            if word not in counts:

                counts[word] = 0

            counts[word] += 1

        one_word = []

        for key, value in counts.items():

            if value == 1 and key.isalpha() and len(key)>2:

                one_word.append(key)

        noun = []

        verb = []

        nlp = spacy.load('en_core_web_sm')

        doc = nlp(u''+' '.join(one_word))

        for token in doc:

            if  token.pos_ == 'VERB':

                verb.append(token.text)

            if  token.pos_ == 'NOUN':

                noun.append(token.text)

            

        all_main =verb + noun

        len_all = len(noun)+len(verb)

        final_value = int(len_all * percent /100)

        random.seed(4)

        temp = random.sample(range(0, len_all), final_value)

        for i in temp:

            try:

                word_str = all_main[i]

                w = Word(word_str)

                a1= list(w.synonyms())

                if i<len(verb):

                    change_word=synalter_Noun_Verb(word_str,a1,'v')

                    try:

                        search_word = re.search(r'\b('+word_str+r')\b', output_text)

                        Loc = search_word.start()

                        output_text = output_text[:int(Loc)] + change_word + output_text[int(Loc) + len(word_str):]

                    except:

                        f=0



                else:

                    change_word=synalter_Noun_Verb(word_str,a1,'n')

                    try:

                        search_word = re.search(r'\b('+word_str+r')\b', output_text)

                        Loc = search_word.start()

                        output_text = output_text[:int(Loc)] + change_word + output_text[int(Loc) + len(word_str):]

                    except:

                        f=0



            except:

                f=0

        print('')

        print('Output:')

        print(output_text)

        f = open(textfile, "a")

        f.write(str(output_text))

        print('')