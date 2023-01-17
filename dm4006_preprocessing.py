import os

import sys

import random

import argparse

import json

import nltk

import numpy as np

from tqdm import tqdm

from six.moves.urllib.request import urlretrieve

#import importlib

#importlib.reload(sys)

#sys.setdefaultencoding('utf8')

random.seed(42)

np.random.seed(42)



SQUAD_BASE_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"

import base64

def write_to_file(out_file, line):

    line= base64.b64encode(line)

    out_file.write(line.encode('utf8') + b'\n')





def data_from_json(filename):

    """Loads JSON data from filename and returns"""

    with open(filename) as data_file:

        data = json.load(data_file)

    return data





def tokenize(sequence):

    tokens = [token.replace("``", '"').replace("''", '"').lower() for token in nltk.word_tokenize(sequence)]

    return tokens





def total_exs(dataset):

    """

    Returns the total number of (context, question, answer) triples,

    given the data read from the SQuAD json file.

    """

    total = 0

    for article in dataset['data']:

        for para in article['paragraphs']:

            total += len(para['qas'])

    return total





def reporthook(t):

    """https://github.com/tqdm/tqdm"""

    last_b = [0]



    def inner(b=1, bsize=1, tsize=None):

        """

        b: int, optional

            Number of blocks just transferred [default: 1].

        bsize: int, optional

            Size of each block (in tqdm units) [default: 1].

        tsize: int, optional

            Total size (in tqdm units). If [default: None] remains unchanged.

        """

        if tsize is not None:

            t.total = tsize

        t.update((b - last_b[0]) * bsize)

        last_b[0] = b



    return inner







def get_char_word_loc_mapping(context, context_tokens):

    """

    Return a mapping that maps from character locations to the corresponding token locations.

    If we're unable to complete the mapping e.g. because of special characters, we return None.



    Inputs:

      context: string (unicode)

      context_tokens: list of strings (unicode)



    Returns:

      mapping: dictionary from ints (character locations) to (token, token_idx) pairs

        Only ints corresponding to non-space character locations are in the keys

        e.g. if context = "hello world" and context_tokens = ["hello", "world"] then

        0,1,2,3,4 are mapped to ("hello", 0) and 6,7,8,9,10 are mapped to ("world", 1)

    """

    acc = '' # accumulator

    current_token_idx = 0 # current word loc

    mapping = dict()



    for char_idx, char in enumerate(context): # step through original characters

        if char != u' ' and char != u'\n': # if it's not a space:

            acc += char # add to accumulator

            context_token = str(context_tokens[current_token_idx]) # current word token

            if acc == context_token: # if the accumulator now matches the current word token

                syn_start = char_idx - len(acc) + 1 # char loc of the start of this word

                for char_loc in range(syn_start, char_idx+1):

                    mapping[char_loc] = (acc, current_token_idx) # add to mapping

                acc = '' # reset accumulator

                current_token_idx += 1



    if current_token_idx != len(context_tokens):

        return None

    else:

        return mapping





def preprocess_and_write(dataset, tier, out_dir):

    """Reads the dataset, extracts context, question, answer, tokenizes them,

    and calculates answer span in terms of token indices.

    Note: due to tokenization issues, and the fact that the original answer

    spans are given in terms of characters, some examples are discarded because

    we cannot get a clean span in terms of tokens.



    This function produces the {train/dev}.{context/question/answer/span} files.



    Inputs:

      dataset: read from JSON

      tier: string ("train" or "dev")

      out_dir: directory to write the preprocessed files

    Returns:

      the number of (context, question, answer) triples written to file by the dataset.

    """

    error=0

    num_exs = 0 # number of examples written to file

    num_mappingprob, num_tokenprob, num_spanalignprob = 0, 0, 0

    examples = []



    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):



        article_paragraphs = dataset['data'][articles_id]['paragraphs']

        for pid in range(len(article_paragraphs)):



            context =str(article_paragraphs[pid]['context']) # string



            # The following replacements are suggested in the paper

            # BidAF (Seo et al., 2016)

            context = context.replace("''", '" ')

            context = context.replace("``", '" ')



            context_tokens = tokenize(context) # list of strings (lowercase)

            context = context.lower()



            qas = article_paragraphs[pid]['qas'] # list of questions



            charloc2wordloc = get_char_word_loc_mapping(context, context_tokens) # charloc2wordloc maps the character location (int) of a context token to a pair giving (word (string), word loc (int)) of that token



            if charloc2wordloc is None: # there was a problem

                num_mappingprob += len(qas)

                continue # skip this context example



            # for each question, process the question and answer and write to file

            for qn in qas:



                # read the question text and tokenize

                question = str(qn['question']) # string

                question_tokens = tokenize(question) # list of strings



                # of the three answers, just take the first

                try:

                    

                    ans_text = str(qn['answers'][0]['text']).lower() # get the answer text

                    ans_start_charloc = qn['answers'][0]['answer_start'] # answer start loc (character count)

                except:

                    error=error+1

                ans_end_charloc = ans_start_charloc + len(ans_text) # answer end loc (character count) (exclusive)



                # Check that the provided character spans match the provided answer text

                if context[ans_start_charloc:ans_end_charloc] != ans_text:

                  # Sometimes this is misaligned, mostly because "narrow builds" of Python 2 interpret certain Unicode characters to have length 2 https://stackoverflow.com/questions/29109944/python-returns-length-of-2-for-single-unicode-character-string

                  # We should upgrade to Python 3 next year!

                  num_spanalignprob += 1

                  continue



                # get word locs for answer start and end (inclusive)

                ans_start_wordloc = charloc2wordloc[ans_start_charloc][1] # answer start word loc

                ans_end_wordloc = charloc2wordloc[ans_end_charloc-1][1] # answer end word loc

                assert ans_start_wordloc <= ans_end_wordloc



                # Check retrieved answer tokens match the provided answer text.

                # Sometimes they won't match, e.g. if the context contains the phrase "fifth-generation"

                # and the answer character span is around "generation",

                # but the tokenizer regards "fifth-generation" as a single token.

                # Then ans_tokens has "fifth-generation" but the ans_text is "generation", which doesn't match.

                ans_tokens = context_tokens[ans_start_wordloc:ans_end_wordloc+1]

                if "".join(ans_tokens) != "".join(ans_text.split()):

                    num_tokenprob += 1

                    continue # skip this question/answer pair



                examples.append((' '.join(context_tokens), ' '.join(question_tokens), ' '.join(ans_tokens), ' '.join([str(ans_start_wordloc), str(ans_end_wordloc)])))



                num_exs += 1



    print("Number of (context, question, answer) triples discarded due to char -> token mapping problems: ", num_mappingprob)

    print("Number of (context, question, answer) triples discarded because character-based answer span is unaligned with tokenization: ", num_tokenprob)

    print("Number of (context, question, answer) triples discarded due character span alignment problems (usually Unicode problems): ", num_spanalignprob)

    print("Processed %i examples of total %i\n" % (num_exs, num_exs + num_mappingprob + num_tokenprob + num_spanalignprob))



    # shuffle examples

    indices = range(len(examples))

    np.random.shuffle(list(indices))



    context_file=open(tier +'.context', 'w')

    question_file=open(tier +'.question', 'w')

    ans_text_file=open(tier +'.answer', 'w')

    span_file=open(tier +'.span', 'w')



    for i in indices:

        (context, question, answer, answer_span) = examples[i]



        # write tokenized data to file

        context_file.write(context+'\n')

        question_file.write(question+'\n')

        ans_text_file.write(answer+'\n')

        span_file.write(answer_span+'\n')

#main function



train_data = data_from_json("../input/train-v1.1.json")

print("Train data has %i examples total" % total_exs(train_data))



# preprocess train set and write to file

preprocess_and_write(train_data, 'train', 'data_out')



# download dev set

#maybe_download(SQUAD_BASE_URL, dev_filename, args.data_dir, 4854279)



# read dev set

dev_data = data_from_json("../input/dev-v1.1.json")

print("Dev data has %i examples total" % total_exs(dev_data))



# preprocess dev set and write to file

preprocess_and_write(dev_data, 'dev', 'data_out')




