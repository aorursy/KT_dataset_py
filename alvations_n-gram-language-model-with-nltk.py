!pip install -U pip

!pip install -U dill

!pip install -U nltk==3.4
from nltk.util import pad_sequence

from nltk.util import bigrams

from nltk.util import ngrams

from nltk.util import everygrams

from nltk.lm.preprocessing import pad_both_ends

from nltk.lm.preprocessing import flatten
text = [['a', 'b', 'c'], ['a', 'c', 'd', 'c', 'e', 'f']]
list(bigrams(text[0]))
list(ngrams(text[1], n=3))
from nltk.util import pad_sequence

list(pad_sequence(text[0],

                  pad_left=True, left_pad_symbol="<s>",

                  pad_right=True, right_pad_symbol="</s>",

                  n=2)) # The n order of n-grams, if it's 2-grams, you pad once, 3-grams pad twice, etc. 
padded_sent = list(pad_sequence(text[0], pad_left=True, left_pad_symbol="<s>", 

                                pad_right=True, right_pad_symbol="</s>", n=2))

list(ngrams(padded_sent, n=2))
list(pad_sequence(text[0],

                  pad_left=True, left_pad_symbol="<s>",

                  pad_right=True, right_pad_symbol="</s>",

                  n=3)) # The n order of n-grams, if it's 2-grams, you pad once, 3-grams pad twice, etc. 
padded_sent = list(pad_sequence(text[0], pad_left=True, left_pad_symbol="<s>", 

                                pad_right=True, right_pad_symbol="</s>", n=3))

list(ngrams(padded_sent, n=3))
from nltk.lm.preprocessing import pad_both_ends

list(pad_both_ends(text[0], n=2))

list(bigrams(pad_both_ends(text[0], n=2)))
from nltk.util import everygrams

padded_bigrams = list(pad_both_ends(text[0], n=2))

list(everygrams(padded_bigrams, max_len=2))
from nltk.lm.preprocessing import flatten

list(flatten(pad_both_ends(sent, n=2) for sent in text))
from nltk.lm.preprocessing import padded_everygram_pipeline

train, vocab = padded_everygram_pipeline(2, text)
training_ngrams, padded_sentences = padded_everygram_pipeline(2, text)

for ngramlize_sent in training_ngrams:

    print(list(ngramlize_sent))

    print()

print('#############')

list(padded_sentences)
try: # Use the default NLTK tokenizer.

    from nltk import word_tokenize, sent_tokenize 

    # Testing whether it works. 

    # Sometimes it doesn't work on some machines because of setup issues.

    word_tokenize(sent_tokenize("This is a foobar sentence. Yes it is.")[0])

except: # Use a naive sentence tokenizer and toktok.

    import re

    from nltk.tokenize import ToktokTokenizer

    # See https://stackoverflow.com/a/25736515/610569

    sent_tokenize = lambda x: re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', x)

    # Use the toktok tokenizer that requires no dependencies.

    toktok = ToktokTokenizer()

    word_tokenize = word_tokenize = toktok.tokenize
import os

import requests

import io #codecs





# Text version of https://kilgarriff.co.uk/Publications/2005-K-lineer.pdf

if os.path.isfile('language-never-random.txt'):

    with io.open('language-never-random.txt', encoding='utf8') as fin:

        text = fin.read()

else:

    url = "https://gist.githubusercontent.com/alvations/53b01e4076573fea47c6057120bb017a/raw/b01ff96a5f76848450e648f35da6497ca9454e4a/language-never-random.txt"

    text = requests.get(url).content.decode('utf8')

    with io.open('language-never-random.txt', 'w', encoding='utf8') as fout:

        fout.write(text)
# Tokenize the text.

tokenized_text = [list(map(str.lower, word_tokenize(sent))) 

                  for sent in sent_tokenize(text)]
tokenized_text[0]
print(text[:500])
# Preprocess the tokenized text for 3-grams language modelling

n = 3

train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)
from nltk.lm import MLE

model = MLE(n) # Lets train a 3-grams model, previously we set n=3
len(model.vocab)
model.fit(train_data, padded_sents)

print(model.vocab)
len(model.vocab)
print(model.vocab.lookup(tokenized_text[0]))
# If we lookup the vocab on unseen sentences not from the training data, 

# it automatically replace words not in the vocabulary with `<UNK>`.

print(model.vocab.lookup('language is never random lah .'.split()))
print(model.counts)
model.counts['language'] # i.e. Count('language')
model.counts[['language']]['is'] # i.e. Count('is'|'language')
model.counts[['language', 'is']]['never'] # i.e. Count('never'|'language is')
model.score('language') # P('language')
model.score('is', 'language'.split())  # P('is'|'language')
model.score('never', 'language is'.split())  # P('never'|'language is')
model.score("<UNK>") == model.score("lah")
model.score("<UNK>") == model.score("leh")
model.score("<UNK>") == model.score("lor")
model.logscore("never", "language is".split())
print(model.generate(20, random_seed=7))
from nltk.tokenize.treebank import TreebankWordDetokenizer



detokenize = TreebankWordDetokenizer().detokenize



def generate_sent(model, num_words, random_seed=42):

    """

    :param model: An ngram language model from `nltk.lm.model`.

    :param num_words: Max no. of words to generate.

    :param random_seed: Seed value for random.

    """

    content = []

    for token in model.generate(num_words, random_seed=random_seed):

        if token == '<s>':

            continue

        if token == '</s>':

            break

        content.append(token)

    return detokenize(content)
generate_sent(model, 20, random_seed=7)
print(model.generate(28, random_seed=0))
generate_sent(model, 28, random_seed=0)
generate_sent(model, 20, random_seed=1)
generate_sent(model, 20, random_seed=30)
generate_sent(model, 20, random_seed=42)
import dill as pickle 



with open('kilgariff_ngram_model.pkl', 'wb') as fout:

    pickle.dump(model, fout)
with open('kilgariff_ngram_model.pkl', 'rb') as fin:

    model_loaded = pickle.load(fin)
generate_sent(model_loaded, 20, random_seed=42)
import pandas as pd

df = pd.read_csv('../input/Donald-Tweets!.csv')

df.head()
trump_corpus = list(df['Tweet_Text'].apply(word_tokenize))
# Preprocess the tokenized text for 3-grams language modelling

n = 3

train_data, padded_sents = padded_everygram_pipeline(n, trump_corpus)
from nltk.lm import MLE

trump_model = MLE(n) # Lets train a 3-grams model, previously we set n=3

trump_model.fit(train_data, padded_sents)
generate_sent(trump_model, num_words=20, random_seed=42)
generate_sent(trump_model, num_words=10, random_seed=0)
generate_sent(trump_model, num_words=50, random_seed=10)
print(generate_sent(trump_model, num_words=100, random_seed=52))