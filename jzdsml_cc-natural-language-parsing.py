import os

import nltk

from nltk import pos_tag, RegexpParser

from nltk.tokenize import PunktSentenceTokenizer, word_tokenize

from collections import Counter



def word_sentence_tokenize(text):

  

  # create a PunktSentenceTokenizer

  sentence_tokenizer = PunktSentenceTokenizer(text)

  

  # sentence tokenize text

  sentence_tokenized = sentence_tokenizer.tokenize(text)

  

  # create a list to hold word tokenized sentences

  word_tokenized = list()

  

  # for-loop through each tokenized sentence in sentence_tokenized

  for tokenized_sentence in sentence_tokenized:

    # word tokenize each sentence and append to word_tokenized

    word_tokenized.append(word_tokenize(tokenized_sentence))

    

  return word_tokenized



# function that pulls chunks out of chunked sentence and finds the most common chunks

def np_chunk_counter(chunked_sentences):



    # create a list to hold chunks

    chunks = list()



    # for-loop through each chunked sentence to extract noun phrase chunks

    for chunked_sentence in chunked_sentences:

        for subtree in chunked_sentence.subtrees(filter=lambda t: t.label() == 'NP'):

            chunks.append(tuple(subtree))



    # create a Counter object

    chunk_counter = Counter()



    # for-loop through the list of chunks

    for chunk in chunks:

        # increase counter of specific chunk by 1

        chunk_counter[chunk] += 1



    # return 30 most frequent chunks

    return chunk_counter.most_common(30)



# function that pulls chunks out of chunked sentence and finds the most common chunks

def vp_chunk_counter(chunked_sentences):



    # create a list to hold chunks

    chunks = list()



    # for-loop through each chunked sentence to extract verb phrase chunks

    for chunked_sentence in chunked_sentences:

        for subtree in chunked_sentence.subtrees(filter=lambda t: t.label() == 'VP'):

            chunks.append(tuple(subtree))



    # create a Counter object

    chunk_counter = Counter()



    # for-loop through the list of chunks

    for chunk in chunks:

        # increase counter of specific chunk by 1

        chunk_counter[chunk] += 1



    # return 30 most frequent chunks

    return chunk_counter.most_common(30)
text = open("../input/the_wizard_of_oz.txt", encoding="utf-8").read().lower()

# print(text)
sentence_tokenizer = PunktSentenceTokenizer(text)

sentence_tokenized = sentence_tokenizer.tokenize(text)

# look at a sentence

print(sentence_tokenized[10])

# number of sentences

print(len(sentence_tokenized))
word_tokenized = list()

for sentence in sentence_tokenized:

    word_tokenized.append(word_tokenize(sentence))

print(word_tokenized[10])

print(len(word_tokenized))
pos_tagged_text = list()

for sentence in word_tokenized:

    pos_tagged_text.append(pos_tag(sentence))

print(pos_tagged_text[10])

# Alphabetical list of part-of-speech tags:

# https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
chunk_grammar = "NP: {<DT>?<JJ>*<NN>}"

vp_chunk_grammar = "VP: {<VB.*><DT>?<JJ>*<NN><RB.?>?}"



chunk_parser = RegexpParser(chunk_grammar)

vp_chunk_parser = RegexpParser(vp_chunk_grammar)



chunked_sentence = chunk_parser.parse(pos_tagged_text[10])

print(chunked_sentence)

vp_chunked_sentence = vp_chunk_parser.parse(pos_tagged_text[10])

print(vp_chunked_sentence)
np_chunked_sentences = list()

vp_chunked_sentences = list()

for sentence in pos_tagged_text:

    np_chunked_sentences.append(chunk_parser.parse(sentence))

    vp_chunked_sentences.append(vp_chunk_parser.parse(sentence))

print(np_chunked_sentences[222])

print(vp_chunked_sentences[222])
most_common_np_chunks = np_chunk_counter(np_chunked_sentences)

print("NP chunks")

print(most_common_np_chunks)

most_common_vp_chunks = vp_chunk_counter(vp_chunked_sentences)

print("VP chunks")

print(most_common_vp_chunks)