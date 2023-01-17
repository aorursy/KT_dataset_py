!pip install spacy
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import spacy

nlp = spacy.load('en_core_web_sm')
introduction_text = ('This tutorial is about Natural'\

                     ' Language Processing in Spacy.')

introduction_doc = nlp(introduction_text)

# Extract tokens for the given doc

print ([token.text for token in introduction_doc])
!echo "This tutorial is about Natural Language Processing in Spacy." >> introduction.txt
file_name = 'introduction.txt'

introduction_file_text = open(file_name).read()

introduction_file_doc = nlp(introduction_file_text)

# Extract tokens for the given doc

print ([token.text for token in introduction_file_doc])
about_text = ('Syed Riaz is a Applied AI developer currently' \

              ' working for a Indian-based Anuncio' \

              ' Technologies. He is interested in exploring' \

              ' Natural Language Processing.')

about_doc = nlp(about_text)

sentences = list(about_doc.sents)

len(sentences)
for sentence in sentences:

    print (sentence)
def set_custom_boundaries(doc):

    # Adds support to use `...` as the delimiter for sentence detection

    for token in doc[:-1]:

        if token.text == '...':

            doc[token.i+1].is_sent_start = True

    return doc



ellipsis_text = ('Syed, can you, ... never mind, I forgot' \

                 ' what I was saying. So, do you think' \

                 ' we should ...')

# Load a new model instance

custom_nlp = spacy.load('en_core_web_sm')

custom_nlp.add_pipe(set_custom_boundaries, before='parser')

custom_ellipsis_doc = custom_nlp(ellipsis_text)

custom_ellipsis_sentences = list(custom_ellipsis_doc.sents)

for sentence in custom_ellipsis_sentences:

    print(sentence)
# Sentence Detection with no customization

ellipsis_doc = nlp(ellipsis_text)

ellipsis_sentences = list(ellipsis_doc.sents)

for sentence in ellipsis_sentences:

    print(sentence)
for token in about_doc:

    print (token, token.idx)
for token in about_doc:

    print (token, token.idx, token.text_with_ws,

           token.is_alpha, token.is_punct, token.is_space,

           token.shape_, token.is_stop)
import re

import spacy

from spacy.tokenizer import Tokenizer

custom_nlp = spacy.load('en_core_web_sm')

prefix_re = spacy.util.compile_prefix_regex(custom_nlp.Defaults.prefixes)

suffix_re = spacy.util.compile_suffix_regex(custom_nlp.Defaults.suffixes)

infix_re = re.compile(r'''[-~]''')

def customize_tokenizer(nlp):

    # Adds support to use `-` as the delimiter for tokenization

    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,

                     suffix_search=suffix_re.search,

                     infix_finditer=infix_re.finditer,

                     token_match=None

                    )



custom_nlp.tokenizer = customize_tokenizer(custom_nlp)

custom_tokenizer_about_doc = custom_nlp(about_text)

print([token.text for token in custom_tokenizer_about_doc])
import spacy

spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

len(spacy_stopwords)
for stop_word in list(spacy_stopwords)[:10]:

    print(stop_word)
for token in about_doc:

    if not token.is_stop:

        print (token)
about_no_stopword_doc = [token for token in about_doc if not token.is_stop]

print (about_no_stopword_doc)
conference_help_text = ('Syed Riaz is helping organize a developer'

                        ' conference on Applications of Natural Language'

                        ' Processing. He keeps organizing local AI meetups'

                        ' and several internal talks at his workplace.')

conference_help_doc = nlp(conference_help_text)

for token in conference_help_doc:

    print (token, token.lemma_)
from collections import Counter

complete_text = ('Syed Riaz is a Applied AI research engineer currently'

                 'working for a Bangalore-based Anuncio Technologies. He is'

                 ' interested in exploring Natural Language Processing.'

                 ' There is a developer conference happening on 13 March'

                 ' 2020 in Bangalore. It is titled "Applications of Natural'

                 ' Language Processing". There is a helpline number '

                 ' available at +1-1234567891. Syed is helping organize it.'

                 ' He keeps organizing local AI meetups and several'

                 ' internal talks at his workplace. Syed is also presenting'

                 ' a talk. The talk will introduce the reader about "Use'

                 ' cases of Natural Language Processing in AI industry".'

                 ' Apart from his work, he is very passionate about travelling.'

                 ' Syed would like to see whole world. He has planned '

                 ' to travel different countries one at a time.'

                 ' He is also planning to create travel videos'

                 ' for which he is planning to join a film making course.')



complete_doc = nlp(complete_text)

# Remove stop words and punctuation symbols

words = [token.text for token in complete_doc

         if not token.is_stop and not token.is_punct]



word_freq = Counter(words)



# 5 commonly occurring words with their frequencies

common_words = word_freq.most_common(5)

print (common_words)
# Unique words

unique_words = [word for (word, freq) in word_freq.items() if freq == 1]

print (unique_words)
words_all = [token.text for token in complete_doc if not token.is_punct]

word_freq_all = Counter(words_all)

# 5 commonly occurring words with their frequencies

common_words_all = word_freq_all.most_common(5)

print (common_words_all)
for token in about_doc:

    print (token, token.tag_, token.pos_, spacy.explain(token.tag_))
nouns = []

adjectives = []

for token in about_doc:

    if token.pos_ == 'NOUN':

        nouns.append(token)

    if token.pos_ == 'ADJ':

        adjectives.append(token)



print(nouns)

print(adjectives)
from spacy import displacy

about_interest_text = ('He is interested in learning'

                       ' Natural Language Processing.')

about_interest_doc = nlp(about_interest_text)

displacy.render(about_interest_doc, style='dep', jupyter=True)

def is_token_allowed(token):

    '''

    Only allow valid tokens which are not stop words

    and punctuation symbols.

    '''

    if (not token or not token.string.strip() or

        token.is_stop or token.is_punct):

        return False

    return True



def preprocess_token(token):

    # Reduce token to its lowercase lemma form

    return token.lemma_.strip().lower()



complete_filtered_tokens = [preprocess_token(token)

                            for token in complete_doc if is_token_allowed(token)]

print(complete_filtered_tokens)
from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)



def extract_full_name(nlp_doc):

    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]

    matcher.add('FULL_NAME', None, pattern)

    matches = matcher(nlp_doc)

    for match_id, start, end in matches:

        span = nlp_doc[start:end]

        return span.text



extract_full_name(about_doc)
from spacy.matcher import Matcher



matcher = Matcher(nlp.vocab)



conference_org_text = ('There is a developer conference'

                       'happening on 13 March 2020 in Bangalore. It is titled'

                       ' "Applications of Natural Language Processing".'

                       ' There is a helpline number available'

                       ' at (123) 456-789')



def extract_phone_number(nlp_doc):

    pattern = [{'ORTH': '('}, {'SHAPE': 'ddd'},

               {'ORTH': ')'}, {'SHAPE': 'ddd'},

               {'ORTH': '-', 'OP': '?'},

               {'SHAPE': 'ddd'}]

    matcher.add('PHONE_NUMBER', None, pattern)

    matches = matcher(nlp_doc)

    for match_id, start, end in matches:

        span = nlp_doc[start:end]

        return span.text



conference_org_doc = nlp(conference_org_text)

extract_phone_number(conference_org_doc)
travel_text = 'Syed is planning to travel planet earth.'

travel_doc = nlp(travel_text)

for token in travel_doc:

    print (token.text, token.tag_, token.head.text, token.dep_)
#displacy.serve(travel_doc, style='dep')

displacy.render(travel_doc, style='dep', jupyter=True)
one_line_about_text = ('Syed Riaz is a Applied AI research engineer'

                       ' currently working for a Bangalore-based Anuncio Technologies')



one_line_about_doc = nlp(one_line_about_text)

# Extract children of `engineer`

print([token.text for token in one_line_about_doc[7].children])
# Extract previous neighboring node of `engineer`

print (one_line_about_doc[7].nbor(-1))
# Extract next neighboring node of `engineer`

print (one_line_about_doc[7].nbor())
# Extract all tokens on the left of `engineer`

print([token.text for token in one_line_about_doc[7].lefts])
# Extract tokens on the right of `engineer`

print([token.text for token in one_line_about_doc[7].rights])
# Print subtree of `engineer`

print (list(one_line_about_doc[7].subtree))
def flatten_tree(tree):

    return ''.join([token.text_with_ws for token in list(tree)]).strip()



# Print flattened subtree of `engineer`

print (flatten_tree(one_line_about_doc[7].subtree))
conference_text = ('There is a AI developer conference'

                   ' happening on 13 March 2020 in Bangalore.')

conference_doc = nlp(conference_text)

# Extract Noun Phrases

for chunk in conference_doc.noun_chunks:

    print (chunk)
!pip install textacy
import textacy

about_talk_text = ('The talk will introduce reader about Use'

                   ' cases of Natural Language Processing in'

                   ' AI industry')

pattern = r'(<VERB>?<ADV>*<VERB>+)'

about_talk_doc = textacy.make_spacy_doc(about_talk_text,

                                        lang='en_core_web_sm')

verb_phrases = textacy.extract.pos_regex_matches(about_talk_doc, pattern)

# Print all Verb Phrase

for chunk in verb_phrases:

    print(chunk.text)
# Extract Noun Phrase to explain what nouns are involved

for chunk in about_talk_doc.noun_chunks:

    print (chunk)
anuncio_class_text = ('Anuncio Technologies is situated'

                      ' near Manyatha Tech Park or the City of Bangalore and has'

                      ' world-class AI developers.')

anuncio_class_doc = nlp(anuncio_class_text)

for ent in anuncio_class_doc.ents:

    print(ent.text, ent.start_char, ent.end_char,

          ent.label_, spacy.explain(ent.label_))
#displacy.serve(anuncio_class_doc, style='ent')

displacy.render(anuncio_class_doc, style='dep', jupyter=True)
survey_text = ('Out of 5 people surveyed, Syed Riaz,'

               ' Satyadev Shetty and Praveen Kumar like'

               ' apples. Gourav Sinha and Vikrant Dharmshi'

               ' like oranges.')



def replace_person_names(token):

    if token.ent_iob != 0 and token.ent_type_ == 'PERSON':

        return '[REDACTED] '

    return token.string



def redact_names(nlp_doc):

    for ent in nlp_doc.ents:

        ent.merge()

        tokens = map(replace_person_names, nlp_doc)

        return ''.join(tokens)



survey_doc = nlp(survey_text)

redact_names(survey_doc)