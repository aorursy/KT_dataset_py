# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os, glob, json, logging, re, string, pprint, sys, pickle, threading

from collections import defaultdict, Counter

from typing import Callable

import multiprocessing as mp



import numpy as np

import pandas as pd



import tensorflow as tf

from tensorflow import keras



from gensim import corpora

import nltk.data

from nltk.tokenize import RegexpTokenizer



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



# All this since Jupyter creates its own log

logger = logging.getLogger()

handler = logging.StreamHandler(stream=sys.stdout)

formatter = logging.Formatter(fmt='%(asctime)s : %(levelname)s : %(message)s')

handler.setFormatter(formatter)

handler.setLevel(logging.INFO)

logger.addHandler(handler)



import matplotlib.pyplot as plt

%matplotlib inline
REREAD_DATA = False
meta = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')

meta.head()
# Sort entries into DF's by their text format

pmc_parsed = meta.loc[meta['has_pmc_xml_parse'] & (meta['has_pdf_parse']==False),['title','source_x','pmcid', 'cord_uid']]

pdf_parsed = meta.loc[meta['has_pdf_parse'] & (meta['has_pmc_xml_parse']==False),['title','source_x','sha', 'cord_uid']]

both_parsed = meta.loc[meta['has_pdf_parse'] & meta['has_pmc_xml_parse'],['title','source_x','pmcid','sha', 'cord_uid']]

print("Number of PMC only articles: {0}".format(len(pmc_parsed)))

print("Number of PDF only articles: {0}".format(len(pdf_parsed)))

print("Number of PMC/PDF articles: {0}".format(len(both_parsed)))

print("Total articles: {0}".format(len(pmc_parsed)+len(pdf_parsed)+len(both_parsed)))



# Examples files

pmc = glob.glob('/kaggle/input/CORD-19-research-challenge/*/*/pmc_json/{pmc_id}.xml.json'.format(pmc_id=both_parsed['pmcid'].iloc[0]))[0]

pdf = glob.glob('/kaggle/input/CORD-19-research-challenge/*/*/pdf_json/{sha}.json'.format(sha=both_parsed['sha'].iloc[0]))[0]



both_parsed.head()
# Redefine stopwords to remove punctuiation, as we do in the below extraction

def letters_only(word):

    """Return the word with only the ASCII letters included"""

    return ''.join([i for i in word if i in string.ascii_letters])



PREPROC_STOPWORDS = [letters_only(word) for word in STOPWORDS]



def get_pdf_filename(sha):

    if ';' in sha:

        sha = sha.split(';')[0].strip()

    pdf = glob.glob('/kaggle/input/CORD-19-research-challenge/*/*/pdf_json/{sha}.json'.format(sha=sha))

    if not len(pdf):

        logging.info("Unable to find pdf with sha: {0}".format(sha))

        return False

    return pdf[0]



def get_pmc_filename(pmcid):

    if ';' in pmcid:

        pmcid = pmcid.split(';')[0].strip()

    pmc = glob.glob('/kaggle/input/CORD-19-research-challenge/*/*/pmc_json/{pmcid}.xml.json'.format(pmcid=pmcid))

    if not len(pmc):

        logging.info("Unable to find PMC file with PMCID: {0}".format(pmcid))

        return False

    return pmc[0]



def extract_content(file_name):

    """

    Given a JSON file name, return a dictionary with keys as the 

    section headers and values as a list of the section's paragraphs

    """

    with open(file_name) as fin:

        paper = json.load(fin)

    sections = defaultdict(list)

    if 'abstract' in paper:

        for paragraph in paper['abstract']:

            sections['abstract'].append(paragraph['text'])

    for paragraph in paper['body_text']:

        sections[paragraph['section']].append(paragraph['text'])

    return sections



def extract_bib(file_name):

    with open(file_name) as fin:

        paper = json.load(fin)

    references = []

    for entry in paper['bib_entries'].values():

        if len(entry['authors']):

            references.append((entry['title'], entry['year'], entry['authors'][0]['last']))

        else:

            references.append((entry['title'], entry['year'], None))

    return references



def abstract_to_document(file_name):

    """Given a JSON file name, return a gensim document containing the paragraphs of the abstract."""

    with open(file_name) as fin:

        paper = json.load(fin)

    document = []

    for paragraph in paper['abstract']:

        text = [''.join([i for i in word if i in string.ascii_letters]) for word in paragraph['text'].lower().split()]

        document += [word for word in text if word not in PREPROC_STOPWORDS and len(word)]

    return document



# Build DF organizing paper titles with their full file paths

if REREAD_DATA:

    file_pointer= pd.DataFrame(

        {

        'title': both_parsed.title,

        'file_name': [get_pdf_filename(sha) for sha in both_parsed.sha],

        'cord_uid': both_parsed.cord_uid

        }

    )



    file_pointer.append(

        pd.DataFrame(

            {

            'title': pdf_parsed.title,

            'file_name': [get_pdf_filename(sha) for sha in pdf_parsed.sha],

            'cord_uid': pdf_parsed.cord_uid

            }

        )

    )



    file_pointer.append(

        pd.DataFrame(

            {

            'title': pmc_parsed.title,

            'file_name': [get_pmc_filename(pmcid) for pmcid in pmc_parsed.pmcid],

            'cord_uid': pmc_parsed.cord_uid

            }

        )

    )

    file_pointer.to_csv('/kaggle/working/paper_filenames.csv')



else:

    file_pointer = pd.read_csv(

        '/kaggle/input/cord-abstract-corpus/paper_filenames.csv', 

        usecols=['title','file_name', 'cord_uid']

    )





file_pointer.head()
# Display a random file

sample = file_pointer.sample()

print(sample.iloc[0,0])

content = extract_content(sample.iloc[0,1])

references = extract_bib(sample.iloc[0,1])

pprint.pprint(content)

pprint.pprint(references)
# URL regex thanks to https://gist.github.com/winzig/8894715

URLS = r"""			# Entire matched URL

  (?:

  https?:				# URL protocol and colon

    (?:

      /{1,3}				# 1-3 slashes

      |					#   or

      [a-z0-9%]				# Single letter or digit or '%'

                                       # (Trying not to match e.g. "URI::Escape")

    )

    |					#   or

                                       # looks like domain name followed by a slash:

    [a-z0-9.\-]+[.]

    (?:[a-z]{2,13})

    /

  )

  (?:					# One or more:

    [^\s()<>{}\[\]]+			# Run of non-space, non-()<>{}[]

    |					#   or

    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)

    |

    \([^\s]+?\)				# balanced parens, non-recursive: (...)

  )+

  (?:					# End with:

    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)

    |

    \([^\s]+?\)				# balanced parens, non-recursive: (...)

    |					#   or

    [^\s`!()\[\]{};:'".,<>?«»“”‘’]	# not a space or one of these punct chars

  )

  |					# OR, the following to match naked domains:

  (?:

  	(?<!@)			        # not preceded by a @, avoid matching foo@_gmail.com_

    [a-z0-9]+

    (?:[.\-][a-z0-9]+)*

    [.]

    (?:[a-z]{2,13})

    \b

    /?

    (?!@)			        # not succeeded by a @,

                            # avoid matching "foo.na" in "foo.na@example.com"

  )

"""
# Finds all references of the form [4], [6,7,10], etc.

bracket_refs = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")

# Finds all references of the form (4), (6, 7, 10), etc.

paren_refs = re.compile(r"\((\d+(?:,\s*\d+)*)\)")

# Finds all references of the form (Li et al., 2016), (Jeremy et. al., 2008), etc.

author_refs = re.compile(r"\((\w+) et\.? al\.?, (\d+)\)")

# matches the end of sentences.

end_of_sentence = re.compile(r'\s([?.!"](?:\s|$))')

# Matches arbitrary URLs

url_re = re.compile(URLS)



def space_sub(sentence: str) -> str:

    sentence = ' '.join(sentence.split())

    return end_of_sentence.sub(r'\1', sentence)



def remove_references(sentence: str) -> str:

    sentence = bracket_refs.sub(r"", sentence)

    sentence = paren_refs.sub(r"", sentence)

    sentence = author_refs.sub(r"", sentence)

    return sentence



def remove_urls(sentence: str) -> str:

    return url_re.sub(r'', sentence)



def pre_tokenize(sentence: str) -> str:

    pre_proc = remove_urls(remove_references(sentence.lower()))

    words = space_sub(pre_proc).split()

    words = [''.join([i for i in word if i in string.ascii_letters]) for word in words]

    words = [word for word in words if word not in PREPROC_STOPWORDS and len(word)]

    return ' '.join(words)



def split_sentences(paragraph):

    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentences = sentence_tokenizer.tokenize(paragraph)

    return sentences



def match_references(sentences: list, references: list, tokenizer: Callable[[str], str]=None) -> list:

    matched = []

    if tokenizer is None:

        tokenizer = lambda x: x

    for sentence in sentences:

        pos_refs = bracket_refs.findall(sentence) + paren_refs.findall(sentence)

        auth_refs = author_refs.findall(sentence)

        if pos_refs:

            for ref_list in pos_refs:

                for ref in ref_list.split(','):

                    try:

                        # All positional references within the paper are 1 indexed

                        referenced_title = references[int(ref.strip())-1][0]

                        matched.append((tokenizer(referenced_title), tokenizer(sentence)))

                    except:

                        logging.info("Unable to parse group {0} from sentence: '''{1}'''".format(ref, sentence))

        if auth_refs:

            for (author, year) in auth_refs:

                for (ref_title, ref_year, ref_author) in references:

                    if author == ref_author and str(ref_year) == year:

                        matched.append((tokenizer(ref_title), tokenizer(sentence)))

    return matched

# Test out string processing functions

sentences = split_sentences(list(content.values())[1][0])

print('Sentence split:')

print(sentences, '\n')



print('References:')

print(references, '\n')



matched = match_references(sentences, references)

print('Matched references:')

print(matched, '\n')



print('Pre-tokenized:')

for sentence in sentences:

    print(pre_tokenize(sentence))
def _worker_get_vocab_stats(vocab_entry: tuple) -> dict:

    word, frequency = vocab_entry

    pairs = defaultdict(int)

    symbols = word.split()

    for i in range(len(symbols)-1):

        pairs[symbols[i], symbols[i+1]] += frequency

    return pairs



def mp_get_vocab_stats(vocab: dict) -> dict:

    """Multiprocessing form of get_vocab_stats"""

    pairs = defaultdict(int)

    with mp.Pool() as pool:

        it = pool.imap(_worker_get_vocab_stats, vocab.items(), chunksize=10)



        for pair in it:

            for p, freq in pair.items():

                pairs[p] += freq

            

    return Counter(pairs)

    



def get_vocab_stats(vocab: dict) -> dict:

    """Get counts of pairs of consecutive symbols from a vocabulary"""

    pairs = defaultdict(int)

    for word, frequency in vocab.items():

        symbols = word.split()



        # Counting up occurrences of pairs

        for i in range(len(symbols) - 1):

            pairs[symbols[i], symbols[i + 1]] += frequency



    return pairs



def merge_vocab(pair: tuple, v_in: dict) -> dict:

    """Merge all occurrences of the most frequent pair"""

    

    v_out = {}

    bigram = re.escape(' '.join(pair))

    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

    

    for word in v_in:

        # replace most frequent pair in all vocabulary

        w_out = p.sub(''.join(pair), word)

        v_out[w_out] = v_in[word]



    return v_out



def threaded_update_vocab(file_name, counter_list, index, sentence_tokenizer):

    """Threaded worker to be called by ReferenceCorpus.update_vocab."""

    content = extract_content(file_name)

    

    word_counts = counter_list[index]

            

    for section_name, section in content.items():

        for paragraph in section:

            sentences = sentence_tokenizer.tokenize(paragraph)

            for sentence in sentences:

                tokens = [" ".join(word) + " </w>" for word in pre_tokenize(sentence).split()]

                word_counts.update(tokens)

class ReferenceCorpus(object):

    """Handler of CORD19 documents designed to initially read data, train tokenizer and encodings and

    efficiently batch data to train an ML model on the documents.

    

    Constructor args:

        corpus_df - Data frame of CORD19 paper titles and file paths.

        vocab_file - Pkl file containing prebuild ReferenceCorpus.

        batch_size - Training batch size to return on training iterations.

    

    Attributes:

        enc_vocab - Dictionary mapping pre-trained BPE tokens to their sparse encoding.

        word_counts - Counter of non-BPE words from the corpus.

        default_token - If an unknown token is encountered during encoding, provide this default.

        mask_token - Reserved token used as mask during ML training.

    

    """

    def __init__(

        self, 

        corpus_df=None, 

        vocab_file=None,

        batch_size=4096,

    ):

        

        self.corpus_df = corpus_df

        self.batch_size = batch_size

        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        

        self.enc_vocab = dict()

        self.word_counts = Counter()



        self._vocab_counts = Counter() # Counter of encoded words from the Corpus

        self._tokenizer_dict = dict() # Dictionary of pre-tokenized words (to speed up encoding)

        self._sorted_tokens = list() # List of all symbols from BPE, sorted from longest to shortest



        self.default_token = '<SKIP>'

        self.mask_token = '<MSK>'

        self.seperation_token = '<SEP>'

        self.start_token = '<CLS>'

        

        self.special_tokens = {

            'default_token': self.default_token,

            'mask_token': self.mask_token,

            'seperation_token': self.seperation_token,

            'start_token': self.start_token

        }

        

        if vocab_file is not None:

            self.load(vocab_file)

        elif self.corpus_df is not None:

            self.update_vocab(self.corpus_df)

            

    def update_vocab(self, file_pntr_df):

        """Update the raw vocabulary with a DF of files."""



        for _, row in file_pntr_df.iterrows():

            content = extract_content(row['file_name'])

            for section_name, section in content.items():

                for paragraph in section:

                    sentences = self.sentence_tokenizer.tokenize(paragraph)

                    for sentence in sentences:

                        tokens = [" ".join(word) + " </w>" for word in pre_tokenize(sentence).split()]

                        self.word_counts.update(tokens)

                        

        # Encode the new words with existing encoding

        if len(self.enc_vocab):

            self.tokenize_vocab()

            

        self._update_encoding()



    def tokenize_vocab(self,):

        """

        Substitutes the set of symbols from self.enc_vocab into self._vocab_counts dictionary.

        

        self.enc_vocab is a dict mapping tokenized symbols to their sparse encoding (an integer).

        self._vocab_counts has keys of tokenized words from the corpus (the words are split by spaces into 

        their constituent tokens) and values as the frequency of the word in the corpus.

        

        This method prepares the corpus for further rounds of byte_pair_encode.

        """

        if not len(self._sorted_tokens) or len(self._sorted_tokens) != self.enc_vocab:

            self._sorted_tokens = list(self.enc_vocab.keys())

            self._sorted_tokens.sort(key=lambda x:len(x), reverse=True)

        self._vocab_counts = self.word_counts.copy()

        for symbol in self._sorted_tokens:

            symbol_regex = r'(?:\ |^)' + re.escape(re.sub(r'(\</w\>|\S)', r' \1', symbol).strip()) + r'(?:\ |$)'

            self._vocab_counts = {re.sub(symbol_regex, ' ' + symbol + ' ', word).strip(): count for word,count in self._vocab_counts.items()}

            

    def _update_encoding(self,):

        """To be run after building byte pair encodings to update other private attributes.

        

        Updates enc_vocab (dict mapping tokens to sparse encodings) based on tokens in _vocab_counts (Counter

        storing.

        """

        self._tokenizer_dict = dict()

        self.enc_vocab = dict()

        

        # Add default and mask tokens:

        self.enc_vocab[self.default_token] = 0

        self.enc_vocab[self.mask_token] = 1

        self.enc_vocab[self.seperation_token] = 2

        self.enc_vocab[self.start_token] = 3

        

        for word in self._vocab_counts.keys():

            split_word = word.split()

            self._tokenizer_dict[''.join(split_word)] = split_word

            for symbol in split_word:

                if symbol not in self.enc_vocab:

                    self.enc_vocab[symbol] = len(self.enc_vocab)

        self._sorted_tokens = list(self.enc_vocab.keys())

        self._sorted_tokens.sort(key=lambda x:len(x), reverse=True)

        

        

    def build_byte_pair_encoding(self, num_merges=200, merges_per_round=50, refresh=False):

        """Trains the byte pair encoding based on previously trained encoding and previously read corpus.

        

        Args:

            num_merges - Total number of token merges to complete (default: 200)

            merges_per_round - How many token merges to make per pair counting from the corpus (default: 50)

            refresh - If True, discard any previously trained encoding (default: False)

        """

        if refresh:

            self.enc_vocab = dict()

            self._vocab_counts = self.word_counts.copy()

        elif not self._vocab_counts:

            # NOTE: This isn't perfect in recreating self._vocab_counts, though it will bring you back to a "similar" state

            self.tokenize_vocab()

                

                

        try:

            merge_rounds = (num_merges + merges_per_round - 1) // merges_per_round

            for i in range(merge_rounds):

                n = min(num_merges - i*merges_per_round, merges_per_round)

                pairs = mp_get_vocab_stats(self._vocab_counts)



                if not pairs:

                    break



                #best = max(pairs, key=pairs.get)

                best = pairs.most_common(n)

                for pair, freq in best:

                    self._vocab_counts = merge_vocab(pair, self._vocab_counts)

        except KeyboardInterrupt:

            logging.info("User stopped BPE training early.")

            

        self._update_encoding()

        

                

    def tokenize_word(self, word):

        """Use the pretrained byte pair encoding to tokenize word."""

        if word in self._tokenizer_dict:

            return self._tokenizer_dict[word]

        

        logging.debug("Tokenizing previously unseen word: {0}".format(word))

        tokenized = word

        for symbol in self._sorted_tokens:

            if symbol == self.default_token:

                continue

            if symbol == self.mask_token:

                continue

            symbol_regex = r'(?:\ |^)' + re.escape(re.sub(r'(\</w\>|\S)', r' \1', symbol).strip()) + r'(?:\ |$)'

            tokenized = re.sub(symbol_regex, ' ' + symbol + ' ', tokenized).strip()

            # Check if you are done tokenizing (all portions of the remaining word are tokens themselves)

            if all([i in self.enc_vocab for i in tokenized.split()]):

                break

        # Save it in case you see the word again

        self._tokenizer_dict[word] = tokenized.split()

        return tokenized.split()

        

    def tokenize_document(self, document):

        """Given a raw document (usually a sentence), return the tokens to be fed into the model."""

        words = ["{0}</w>".format(word) for word in pre_tokenize(document).split()]

        return [token for w in words for token in self.tokenize_word(w)] # Flatten into a single list

    

    def sparse_encode_tokens(self, tokens):

        """Given a list of tokens, return the sparse encoding"""

        return [self.enc_vocab.get(token, self.enc_vocab.get(self.default_token, 0)) for token in tokens]

    

    def sparse_encode_document(self, document):

        tokens = self.tokenize_document(document)

        return self.sparse_encode_tokens(tokens)

        

    def extract_files(self, file_pntr_df, augment=False):

        """Extract a list of documents from a DF of files and store them in self.documents."""

        if not augment:

            self.documents = []

        for _, row in file_pntr_df.iterrows():

            content = extract_content(row['file_name'])

            references = extract_bib(row['file_name'])

            

            for section_name, section in content.items():

                for paragraph in section:

                    sentences = self.sentence_tokenizer.tokenize(paragraph)

                    if len(sentences) > 1:

                        self.documents.append([pre_tokenize(s) for s in sentences])

                    pairs = match_references(sentences, references, tokenizer=pre_tokenize)

                    for pair in pairs:

                        self.documents.append([pair[0], pair[1]])

                        self.documents.append([pair[1], pair[0]])

                        

    def load(self, vocab_file):

        """Reload previously build Corpus."""

        with open(vocab_file, 'rb') as fin:

            saving = pickle.load(fin)

        (self.enc_vocab, self.word_counts, self._vocab_counts, self._tokenizer_dict, self._sorted_tokens) = saving

            

    def save(self, vocab_file):

        """Save Corpus to disk for future use."""

        saving = (self.enc_vocab, self.word_counts, self._vocab_counts, self._tokenizer_dict, self._sorted_tokens)

        with open(vocab_file, 'wb') as fout:

            pickle.dump(saving, fout)

            

    def write_encoded_docs(self, file_pntr_df, root_dir):

        """

        Extract tokens from all files in file_pntr_df and write them to root_dir.

        

        Writes the following files:

            root_dir/token_params.pkl - Contains self.enc_vocab and self._tokenizer_dict.

            root_dir/reference_matching/ - Directory containing one serialized file per paper with reference matching documents.

            root_dir/base_documents/ - Directory containing one serialized file per paper with documents organized by paragraph.

        """

        reference_dir = os.path.join(root_dir, 'reference_matching')

        doc_dir = os.path.join(root_dir, 'base_documents')

        

        if not os.path.isdir(root_dir):

            os.mkdir(root_dir, mode=0o770)

        if not os.path.isdir(reference_dir):

            os.mkdir(reference_dir, mode=0o770)

        if not os.path.isdir(doc_dir):

            os.mkdir(doc_dir, mode=0o770)

            

        token_params = os.path.join(root_dir, 'token_params.pkl')

        with open(token_params, 'wb') as fout:

            pickle.dump((self.enc_vocab, self.special_tokens, self._tokenizer_dict), fout)

            

        for _, row in file_pntr_df.iterrows():

            cord_uid = row['cord_uid']

            content = extract_content(row['file_name'])

            references = extract_bib(row['file_name'])

            

            reference_matches = []

            documents = []

            

            for section_name, section in content.items():

                for paragraph in section:

                    sentences = self.sentence_tokenizer.tokenize(paragraph)

                    documents.append([self.sparse_encode_document(s) for s in sentences])

                    

                    pairs = match_references(sentences, references, tokenizer=pre_tokenize)

                    for pair in pairs:

                        reference_matches.append([self.sparse_encode_document(s) for s in pair])



            doc_name = os.path.join(doc_dir, cord_uid)

            with open(doc_name, 'wb') as fout:

                pickle.dump(documents, fout)

                

            if len(reference_matches):

                ref_name = os.path.join(reference_dir, cord_uid)

                with open(ref_name, 'wb') as fout:

                    pickle.dump(reference_matches, fout)

            

    def __getitem__(self, index):

        """Return a batch of tokenized sentence pairs.

        

        The elements of the batch are tuples of two lists:

            1. input_tokens := The start token, then tokens from sentence 0, then the seperation token, then tokens from sentence 1

            2. sentence_mask := List of 0's and 1's corresponding to the position of the tokens from sentence 0 or 1.

        """

        ret = []

        #num_papers = max(10, self.batch_size//100)

        num_papers = 10

        self.extract_files(self.corpus_df.sample(num_papers))

        # Grab batch_size random sentence pairs from documents

        for doc_num in np.random.randint(0, len(self.documents), size=self.batch_size):

            pair_start = np.random.randint(0, len(self.documents[doc_num])-1)

            sentence0_tokens = self.tokenize_document(self.documents[doc_num][pair_start])

            sentence1_tokens = self.tokenize_document(self.documents[doc_num][pair_start+1])



            input_tokens = [self.start_token] + sentence0_tokens + [self.seperation_token] + sentence1_tokens

            sentence_mask = [0] * (len(sentence0_tokens) + 2) + [1] * len(sentence1_tokens)



            ret.append((input_tokens, sentence_mask))



        return ret
# Build new Corpus

#ref_corpus = ReferenceCorpus(corpus_df=file_pointer)

#ref_corpus.build_byte_pair_encoding(num_merges=500, merges_per_round=50)

#ref_corpus.save('/kaggle/working/reference_corpus.pkl')





# Read previously build corpus

ref_corpus = ReferenceCorpus(

    vocab_file='/kaggle/input/cord-abstract-corpus/reference_corpus.pkl',

    corpus_df=file_pointer,

    batch_size=16

)
ref_corpus.write_encoded_docs(file_pointer, '/tmp/tokenized_docs')
!tar -czf /kaggle/working/tokenized_docs.tar.gz /tmp/tokenized_docs