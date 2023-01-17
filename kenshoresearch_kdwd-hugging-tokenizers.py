from collections import Counter

import json

import os

from pprint import pprint

import string

import time



import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from tqdm import tqdm



sns.set()

sns.set_context('talk')



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
NUM_KLAT_LINES = 5_343_564

kdwd_path = os.path.join("/kaggle/input", "kensho-derived-wikimedia-data")

vocab_path = os.path.join("/kaggle/input", "hugging-face-tokenizer-vocabs")
!pip install tokenizers

!pip install transformers
import tokenizers    # Rust implementations

import transformers  # Python implementations
vocab_file = os.path.join(vocab_path, "bert-base-uncased-vocab.txt")

rust_bert_wp = tokenizers.BertWordPieceTokenizer(vocab_file)

pyth_bert_wp = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

pprint("Rust tokenizer class: {}".format(rust_bert_wp))

print()

pprint("Python tokenizer class: {}".format(pyth_bert_wp))
encoded = rust_bert_wp.encode("Do you feel like I feel?")

pprint("encoded={}".format(encoded))

pprint("encoded.tokens={}".format(encoded.tokens))
tokens = pyth_bert_wp.convert_ids_to_tokens(pyth_bert_wp.encode("Do you feel like I feel?"))

pprint("tokens={}".format(tokens))
class KdwdLinkAnnotatedText:

    def __init__(self, file_path, max_pages):

        self.num_lines = NUM_KLAT_LINES

        self.file_path = file_path

        self.max_pages = max_pages

        self.pages_to_parse = min(self.num_lines, self.max_pages)

    def __iter__(self):

        with open(self.file_path) as fp:

            for ii_line, line in enumerate(fp):

                if ii_line == self.pages_to_parse:

                    break

                yield json.loads(line)
NUM_PAGES = 500
table = str.maketrans('', '', string.punctuation)

def simple_tokenizer(text):

    tokens = [tok.lower().strip() for tok in text.split()]

    tokens = [tok.translate(table) for tok in tokens]

    tokens = [tok for tok in tokens if tok != ""]

    return tokens
file_path = os.path.join(kdwd_path, "link_annotated_text.jsonl")

klat = KdwdLinkAnnotatedText(file_path, max_pages=NUM_PAGES)
t0 = time.time()

for page in tqdm(klat, total=klat.pages_to_parse, desc='just iteration'):

    for section in page['sections']:

        first = section['text'][0]

dt_iter = time.time() - t0

print("dt: {}".format(dt_iter))
unigrams_simple = Counter()

unigrams_hf_rust = Counter()

unigrams_hf_pyth = Counter()
t0 = time.time()

for page in tqdm(klat, total=klat.pages_to_parse, desc='simple tokenizer'):

    for section in page['sections']:

        tokens = simple_tokenizer(section['text'])

        unigrams_simple.update(tokens)

dt_simple = time.time() - t0

print("dt: {}".format(dt_simple))
t0 = time.time()

for page in tqdm(klat, total=klat.pages_to_parse, desc='hugging face Rust tokenizer'):

    for section in page['sections']:

        encoded = rust_bert_wp.encode(section['text'])

        unigrams_hf_rust.update(encoded.tokens)

dt_hf_rust = time.time() - t0

print("dt: {}".format(dt_hf_rust))
t0 = time.time()

for page in tqdm(klat, total=klat.pages_to_parse, desc='hugging face Python tokenizer'):

    for section in page['sections']:

        tokens = pyth_bert_wp.convert_ids_to_tokens(pyth_bert_wp.encode(section['text']))

        unigrams_hf_pyth.update(tokens)

dt_hf_pyth = time.time() - t0

print("dt: {}".format(dt_hf_pyth))
labels = ["just iteration", "simple", "hugging rust", "hugging python"]

times = np.array([dt_iter, dt_simple, dt_hf_rust, dt_hf_pyth])

rates = np.array([

    sum(unigrams_simple.values()) / dt_simple,

    sum(unigrams_hf_rust.values()) / dt_hf_rust,

    sum(unigrams_hf_pyth.values()) / dt_hf_pyth,

])

yy = np.arange(len(labels)) 



width = 0.5

figsize = (16, 8)

fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)



ax = axes[0]

rects1 = ax.barh(yy, times, width) 

ax.set_yticks(yy)

ax.set_yticklabels(labels)

ax.set_xlabel('seconds')

ax.set_ylabel('Tokenizer')

ax.set_title('Total Parse Time')



ax = axes[1]

rects2 = ax.barh(yy[1:], rates/1000, width, color="orange") 

ax.set_xlabel('Thousands of Tokens / s')

ax.set_title('Token Parse Rate')



fig.suptitle('Tokenizer Performance on {} Wikipedia Pages'.format(NUM_PAGES));
print("times: {}".format(list(zip(labels, times))))



# normalize by Hugging Face Python

print("times normalized by Hugging Face Python: {}".format(times/times[3]))
print("rates: {}".format(list(zip(labels[1:], rates))))



# normalize by Hugging Face Python

print("rates normalized by Hugging Face Python: {}".format(rates/rates[2]))
unigrams_simple.most_common(25)
unigrams_hf_rust.most_common(25)
unigrams_hf_pyth.most_common(25)