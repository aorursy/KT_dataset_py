example_text = """

An explosion targeting a tourist bus has injured at least 16 people near the Grand Egyptian Museum, 

next to the pyramids in Giza, security sources say E.U.



South African tourists are among the injured. Most of those hurt suffered minor injuries, 

while three were treated in hospital, N.A.T.O. say.



http://localhost:8888/notebooks/Text%20preprocessing.ipynb



@nickname of twitter user and his email is email@gmail.com . 



A device went off close to the museum fence as the bus was passing on 16/02/2012.

"""
from nltk.tokenize import sent_tokenize, word_tokenize



nltk_words = word_tokenize(example_text)

display(f"Tokenized words: {nltk_words}")
import spacy

import en_core_web_sm



nlp = en_core_web_sm.load()



doc = nlp(example_text)

spacy_words = [token.text for token in doc]

display(f"Tokenized words: {spacy_words}")
display(f"In spacy but not in nltk: {set(spacy_words).difference(set(nltk_words))}")

display(f"In nltk but not in spacy: {set(nltk_words).difference(set(spacy_words))}")
import string



display(f"Punctuation symbols: {string.punctuation}")
text_with_punct = "@nickname of twitter user, and his email is email@gmail.com ."
text_without_punct = text_with_punct.translate(str.maketrans('', '', string.punctuation))

display(f"Text without punctuation: {text_without_punct}")
doc = nlp(text_with_punct)

tokens = [t.text for t in doc]

# python 

tokens_without_punct_python = [t for t in tokens if t not in string.punctuation]

display(f"Python based removal: {tokens_without_punct_python}")



tokens_without_punct_spacy = [t.text for t in doc if t.pos_ != 'PUNCT']

display(f"Spacy based removal: {tokens_without_punct_spacy}")
text = "This movie is just not good enough"
spacy_stop_words = spacy.lang.en.stop_words.STOP_WORDS



display(f"Spacy stop words count: {len(spacy_stop_words)}")
text_without_stop_words = [t.text for t in nlp(text) if not t.is_stop]

display(f"Spacy text without stop words: {text_without_stop_words}")
import nltk



nltk_stop_words = nltk.corpus.stopwords.words('english')

display(f"nltk stop words count: {len(nltk_stop_words)}")
text_without_stop_words = [t for t in word_tokenize(text) if t not in nltk_stop_words]

display(f"nltk text without stop words: {text_without_stop_words}")
import en_core_web_sm



nlp = en_core_web_sm.load()



customize_stop_words = [

    'not'

]



for w in customize_stop_words:

    nlp.vocab[w].is_stop = False



text_without_stop_words = [t.text for t in nlp(text) if not t.is_stop]

display(f"Spacy text without updated stop words: {text_without_stop_words}")
from normalise import normalise



text = """

On the 13 Feb. 2007, Theresa May announced on MTV news that the rate of childhod obesity had 

risen from 7.3-9.6% in just 3 years , costing the N.A.T.O £20m

"""



user_abbr = {

    "N.A.T.O": "North Atlantic Treaty Organization"

}



normalized_tokens = normalise(word_tokenize(text), user_abbrevs=user_abbr, verbose=False)

display(f"Normalized text: {' '.join(normalized_tokens)}")
from nltk.stem import PorterStemmer

import numpy as np



text = ' '.join(normalized_tokens)

tokens = word_tokenize(text)
porter=PorterStemmer()

stem_words = np.vectorize(porter.stem)

stemed_text = ' '.join(stem_words(tokens))

display(f"Stemed text: {stemed_text}")
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

lemmatize_words = np.vectorize(wordnet_lemmatizer.lemmatize)

lemmatized_text = ' '.join(lemmatize_words(tokens))

display(f"nltk lemmatized text: {lemmatized_text}")
lemmas = [t.lemma_ for t in nlp(text)]

display(f"Spacy lemmatized text: {' '.join(lemmas)}")
import numpy as np

import multiprocessing as mp



import string

import spacy 

import en_core_web_sm

from nltk.tokenize import word_tokenize

from sklearn.base import TransformerMixin, BaseEstimator

from normalise import normalise



nlp = en_core_web_sm.load()





class TextPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self,

                 variety="BrE",

                 user_abbrevs={},

                 n_jobs=1):

        """

        Text preprocessing transformer includes steps:

            1. Text normalization

            2. Punctuation removal

            3. Stop words removal

            4. Lemmatization

        

        variety - format of date (AmE - american type, BrE - british format) 

        user_abbrevs - dict of user abbreviations mappings (from normalise package)

        n_jobs - parallel jobs to run

        """

        self.variety = variety

        self.user_abbrevs = user_abbrevs

        self.n_jobs = n_jobs



    def fit(self, X, y=None):

        return self



    def transform(self, X, *_):

        X_copy = X.copy()



        partitions = 1

        cores = mp.cpu_count()

        if self.n_jobs <= -1:

            partitions = cores

        elif self.n_jobs <= 0:

            return X_copy.apply(self._preprocess_text)

        else:

            partitions = min(self.n_jobs, cores)



        data_split = np.array_split(X_copy, partitions)

        pool = mp.Pool(cores)

        data = pd.concat(pool.map(self._preprocess_part, data_split))

        pool.close()

        pool.join()



        return data



    def _preprocess_part(self, part):

        return part.apply(self._preprocess_text)



    def _preprocess_text(self, text):

        normalized_text = self._normalize(text)

        doc = nlp(normalized_text)

        removed_punct = self._remove_punct(doc)

        removed_stop_words = self._remove_stop_words(removed_punct)

        return self._lemmatize(removed_stop_words)



    def _normalize(self, text):

        # some issues in normalise package

        try:

            return ' '.join(normalise(text, variety=self.variety, user_abbrevs=self.user_abbrevs, verbose=False))

        except:

            return text



    def _remove_punct(self, doc):

        return [t for t in doc if t.text not in string.punctuation]



    def _remove_stop_words(self, doc):

        return [t for t in doc if not t.is_stop]



    def _lemmatize(self, doc):

        return ' '.join([t.lemma_ for t in doc])
import pandas as pd



df_bbc = pd.read_csv('../input/bbc-text.csv')
%%time

text = TextPreprocessor(n_jobs=-1).transform(df_bbc['text'])
print(f"Performance of transformer on {len(df_bbc)} texts and {mp.cpu_count()} processes")