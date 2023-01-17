

import numpy as np

import os

import sys

import json

import nltk.data



tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



# If there is both 'incubation' and 'COVID-19' or '2019-nCoV' then the text is considered about COVID-19 incubation period

def is_covid_incubation(body_text):

    last_paragraph = body_text[-1]['text']

    is_incubation = False

    is_covid = False

    for paragraph in body_text:

        if 'incubation' in paragraph['text']:

            is_incubation = True

        if 'COVID-19' in paragraph['text'] or '2019-nCoV' in paragraph['text']:

            is_covid = True

    return is_covid and is_incubation



def is_number(s):

    try:

        float(s)

        return True

    except ValueError:

        return False

    

def has_number(sentence):

    words = nltk.word_tokenize(sentence)

    for word in words:

        if is_number(word):

            return True

    return False



def get_known_incubation_sentence(body_text):   # Finding sentence about both 'incubation' and mean/median/average that does not contains mentions of SARS or MERS

    for paragraph in body_text:

        sentences = tokenizer.tokenize(paragraph['text'])

        for sentence in sentences:

            if 'SARS' not in sentence and 'MERS' not in sentence:

                if 'incubation' in sentence.lower() and has_number(sentence) and ('average' in sentence.lower() or 'median' in sentence.lower() or 'mean' in sentence.lower()):

                    return sentence

    return ''



end_print = '' # The mentions of 'incubation' where there is no mean/average/median



for dirname, _, filenames in os.walk('/kaggle/input'):   # Going through all the files

    for filename in filenames:

        if filename.endswith('json'):

            with open(os.path.join(dirname, filename)) as f:

                data = json.load(f)

                if is_covid_incubation(data['body_text']):   # If the text is about COVID-19 incubation perios

                    printout = dirname + '/' + filename + '\n' + data['metadata']['title']

                    incubation_sentence = get_known_incubation_sentence(data['body_text'])    # Get the sentence about the incubation period

                    if not incubation_sentence:                      # If no sentence about the exact incubation period is found print all paragraphs where incubation is mentioned

                        end_print = end_print + printout + '\n'      # Save the paragraphs to be printed at the end

                        for paragraph in data['body_text']:

                            if 'incubation' in paragraph['text']:

                                end_print = end_print + paragraph['text'] + '\n\n'

                    else:

                        print(printout)                   # If menntion of the exact incubation period is found print it immediately

                        print(incubation_sentence)

                        print()

print(end_print)  # Print the mentions of 'incubation' where there is no mean/average/median after all else