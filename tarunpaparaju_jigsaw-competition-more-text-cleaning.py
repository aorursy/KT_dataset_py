import os

import gc



import re

import numpy as np

import pandas as pd



import nltk

from nltk.corpus import wordnet, stopwords

from nltk.stem import WordNetLemmatizer

from nltk.stem.porter import PorterStemmer



from colorama import Fore, Back, Style
train_df = pd.read_csv('../input/train.csv')
train_df.head()
comments = train_df['comment_text']
def example_cleaning_results(function):

    select_comments = []

    for i, comment in enumerate(comments):

        if comment != function(comment):

            select_comments.append(comment)

        if len(select_comments) == 5:

            break

    

    print("                          " +\

          f'{Style.DIM}'+\

          "EXAMPLE WORKING OF TEXT CLEANING FUNCTION"+\

          f'{Style.RESET_ALL}')

    print("                          " +\

          f'{Style.DIM}'+\

          "-------------------------------------------"+\

          f'{Style.RESET_ALL}')

    print("")



    for comment in select_comments:

        print(f'{Fore.YELLOW}{Style.DIM}' + comment + f'{Style.RESET_ALL}' +\

              '\n\n' + "                                     "+\

              'CHANGES TO' + '\n\n' +\

              f'{Fore.CYAN}{Style.DIM}' + function(comment) + f'{Style.RESET_ALL}')

        

        print("")

        

        print(f'{Fore.WHITE}{Style.DIM}' +\

              "-------------------------"+\

              "-------------------------"+\

              "-------------------------"+\

              "------------------" +\

              f'{Style.RESET_ALL}')
def remove_numbers(text):

    """ Removes integers """

    text = ''.join([i for i in text if not i.isdigit()])         

    return text
example_cleaning_results(remove_numbers)
def replace_multi_exclamation_mark(text):

    """ Replaces repetitions of exlamation marks """

    text = re.sub(r"(\!)\1+", ' multiExclamation ', text)

    return text



def replace_multi_question_mark(text):

    """ Replaces repetitions of question marks """

    text = re.sub(r"(\?)\1+", ' multiQuestion ', text)

    return text



def replace_multi_stop_mark(text):

    """ Replaces repetitions of stop marks """

    text = re.sub(r"(\.)\1+", ' multiStop ', text)

    return text
example_cleaning_results(lambda x: replace_multi_exclamation_mark(replace_multi_question_mark(replace_multi_stop_mark(x))))
contraction_patterns = [(r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'),\

                        (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'),\

                        (r'(\w+)n\'t', '\g<1> not'),\

                        (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'),\

                        (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'),\

                        (r'&', 'and'), (r'dammit', 'damn it'), (r'dont', 'do not'),\

                        (r'wont', 'will not')]



def replace_contraction(text):

    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]

    for (pattern, repl) in patterns:

        (text, count) = re.subn(pattern, repl, text)

    return text
example_cleaning_results(replace_contraction)
def replace(word, pos=None):

    """ Creates a set of all antonyms for the word and if there is only one antonym, it returns it """

    antonyms = set()

    for syn in wordnet.synsets(word, pos=pos):

        for lemma in syn.lemmas():

            for antonym in lemma.antonyms():

                antonyms.add(antonym.name())

    if len(antonyms) == 1:

        return antonyms.pop()

    else:

        return None



def replace_negations(text):

    """ Finds "not" and antonym for the next word and if found, replaces not and the next word with the antonym """

    i, l = 0, len(text)

    words = []

    while i < l:

        word = text[i]

        if word == 'not' and i+1 < l:

            ant = replace(text[i+1])

            if ant:

                words.append(ant)

                i += 2

                continue

        words.append(word)

        i += 1

    return words



def tokenize_and_replace_negations(text):

    tokens = nltk.word_tokenize(text)

    tokens = replace_negations(tokens)

    text = " ".join(tokens)

    return text
example_cleaning_results(tokenize_and_replace_negations)
stoplist = stopwords.words('english')



def remove_stop_words(text):

    finalTokens = []

    tokens = nltk.word_tokenize(text)

    for w in tokens:

        if (w not in stoplist):

            finalTokens.append(w)

    text = " ".join(finalTokens)

    return text
example_cleaning_results(remove_stop_words)
def replace_elongated(word):

    """ Replaces an elongated word with its basic form, unless the word exists in the lexicon """



    repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')

    repl = r'\1\2\3'

    if wordnet.synsets(word):

        return word

    repl_word = repeat_regexp.sub(repl, word)

    if repl_word != word:      

        return replace_elongated(repl_word)

    else:       

        return repl_word

    

def replace_elongated_words(text):

    finalTokens = []

    tokens = nltk.word_tokenize(text)

    for w in tokens:

        finalTokens.append(replace_elongated(w))

    text = " ".join(finalTokens)

    return text
example_cleaning_results(replace_elongated_words)
stemmer = PorterStemmer()



def stem_words(text):

    finalTokens = []

    tokens = nltk.word_tokenize(text)

    for w in tokens:

        finalTokens.append(stemmer.stem(w))

    text = " ".join(finalTokens)

    return text
example_cleaning_results(stem_words)
lemmatizer = WordNetLemmatizer()



def lemmatize_words(text):

    finalTokens = []

    tokens = nltk.word_tokenize(text)

    for w in tokens:

        finalTokens.append(lemmatizer.lemmatize(w))

    text = " ".join(finalTokens)

    return text
example_cleaning_results(lemmatize_words)