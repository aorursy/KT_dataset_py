["2019-ncov", "2019 novel coronavirus", "coronavirus 2019", "coronavirus disease 19", "covid-19", "covid 19", "ncov-2019", "sars-cov-2", "wuhan coronavirus", "wuhan pneumonia", "wuhan virus"]
# Install scispacy package
!pip install scispacy
import spacy
import scispacy

nlp = spacy.load("../input/scispacymodels/en_core_sci_sm/en_core_sci_sm-0.2.4")
nlp.max_length = 2000000
!pip install contractions
import re

CURRENCIES = {'$': 'USD', 'zł': 'PLN', '£': 'GBP', '¥': 'JPY', '฿': 'THB',
              '₡': 'CRC', '₦': 'NGN', '₩': 'KRW', '₪': 'ILS', '₫': 'VND',
              '€': 'EUR', '₱': 'PHP', '₲': 'PYG', '₴': 'UAH', '₹': 'INR'}

RE_NUMBER = re.compile(
    r"(?:^|(?<=[^\w,.]))[+–-]?"
    r"(([1-9]\d{0,2}(,\d{3})+(\.\d*)?)|([1-9]\d{0,2}([ .]\d{3})+(,\d*)?)|(\d*?[.,]\d+)|\d+)"
    r"(?:$|(?=\b))")

RE_URL = re.compile(
    r'((http://www\.|https://www\.|http://|https://)?' +
    r'[a-z0-9]+([\-.][a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(/.*)?)')

# English Stop Word List (Standard stop words used by Apache Lucene)
STOP_WORDS = {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
              "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
              "they", "this", "to", "was", "will", "with"}
import string
from typing import List
import ftfy
import contractions

def clean_tokenized_sentence(tokens: List[str],
                             unicode_normalization="NFC",
                             unpack_contractions=False,
                             replace_currency_symbols=False,
                             remove_punct=True,
                             remove_numbers=False,
                             lowercase=True,
                             remove_urls=True,
                             remove_stop_words=True) -> str:
    if remove_stop_words:
        tokens = [token for token in tokens if token not in STOP_WORDS]

    sentence = ' '.join(tokens)

    if unicode_normalization:
        sentence = ftfy.fix_text(sentence, normalization=unicode_normalization)

    if unpack_contractions:
        sentence = contractions.fix(sentence, slang=False)

    if replace_currency_symbols:
        for currency_sign, currency_tok in CURRENCIES.items():
            sentence = sentence.replace(currency_sign, f'{currency_tok} ')

    if remove_urls:
        sentence = RE_URL.sub('_URL_', sentence)

    if remove_punct:
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))

    # strip double spaces
    sentence = re.sub(r' +', ' ', sentence)

    if remove_numbers:
        sentence = RE_NUMBER.sub('_NUMBER_', sentence)

    if lowercase:
        sentence = sentence.lower()

    return sentence

def clean_sentence(sentence) -> str:
    doc = nlp(sentence)
    tokens = [str(token) for token in doc]
    return clean_tokenized_sentence(tokens)
print(clean_sentence("Let's clean this sentence!"))
import pandas as pd
sentences_df = pd.read_csv('../input/covid19sentencesmetadata/sentences_with_metadata.csv')
sentences_df.head()
print(f"Sentence count: {len(sentences_df)}")
from gensim.models.phrases import Phraser
bigram_model = Phraser.load("../input/covid19phrasesmodels/covid_bigram_model_v0.pkl")
bigram_model["despite social media often vehicle fake news boast news hype also worth noting tremendous effort scientific community provide free uptodate information ongoing studies well critical evaluations".split()]
trigram_model = Phraser.load("../input/covid19phrasesmodels/covid_trigram_model_v0.pkl")
def clean_sentence(sentence) -> str:
    doc = nlp(sentence)
    tokens = [str(token) for token in doc]
    cleaned_sentence = clean_tokenized_sentence(tokens)
    sentence_with_bigrams = bigram_model[cleaned_sentence.split(' ')]
    sentence_with_trigrams = trigram_model[sentence_with_bigrams]
    return ' '.join(sentence_with_trigrams)
print(clean_sentence("On 23 January 2020, the Coalition for Epidemic Preparedness Innovations (CEPI) announced that they will fund vaccine development programmes with Inovio"))
import os
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from matplotlib import pylab
%matplotlib inline
fasttext_model_dir = '../input/fasttext-no-subwords-trigrams'
num_points = 400

first_line = True
index_to_word = []
with open(os.path.join(fasttext_model_dir, "word-vectors-100d.txt"),"r") as f:
    for line_num, line in enumerate(f):
        if first_line:
            dim = int(line.strip().split()[1])
            word_vecs = np.zeros((num_points, dim), dtype=float)
            first_line = False
            continue
        line = line.strip()
        word = line.split()[0]
        vec = word_vecs[line_num-1]
        for index, vec_val in enumerate(line.split()[1:]):
            vec[index] = float(vec_val)
        index_to_word.append(word)
        if line_num >= num_points:
            break
word_vecs = normalize(word_vecs, copy=False, return_norm=False)
tsne = TSNE(perplexity=40, n_components=2, init='pca', n_iter=10000)
two_d_embeddings = tsne.fit_transform(word_vecs[:num_points])
labels = index_to_word[:num_points]
def plot(embeddings, labels):
    pylab.figure(figsize=(20,20))
    for i, label in enumerate(labels):
        x, y = embeddings[i,:]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                       ha='right', va='bottom')
    pylab.show()

plot(two_d_embeddings, labels)
from pprint import pprint
import gensim.models.keyedvectors as word2vec

fasttext_model = word2vec.KeyedVectors.load_word2vec_format(os.path.join(fasttext_model_dir, "word-vectors-100d.txt"))
def print_most_similar(search_term):
    print(f"Synonyms of '{search_term}':")
    synonyms = fasttext_model.most_similar(search_term)
    pprint(synonyms)
print_most_similar("new_coronavirus")
print_most_similar("fake_news")
print_most_similar("pathogen")
[(0, '0.079"•" + 0.019"blood" + 0.015"associated" + 0.013"cells" + ' '0.012"ace2" + 0.012"protein" + 0.011"important" + 0.011"levels" + ' '0.010"diseases" + 0.010"cell"'), (1, '0.110"who" + 0.088"it" + 0.056"response" + 0.043"could" + 0.036"under" ' '+ 0.035"available" + 0.032"major" + 0.032"as" + 0.030"without" + ' '0.024"muscle"'), (2, '0.173"■" + 0.020"some" + 0.013"drugs" + 0.010"transmission" + ' '0.009"surgery" + 0.009"must" + 0.009"drug" + 0.009"there" + ' '0.008"increased" + 0.008"high"'), (3, '0.071"de" + 0.036"were" + 0.025"patient" + 0.023"1" + 0.022"after" + ' '0.018"a" + 0.018"more" + 0.015"all" + 0.015"when" + 0.014"cause"'), (4, '0.044"the" + 0.035"from" + 0.028"should" + 0.019"other" + 0.018"risk" ' '+ 0.017"oral" + 0.017"which" + 0.017"in" + 0.013"use" + 0.013"cases"'), (5, '0.069"may" + 0.033"can" + 0.031"have" + 0.029"disease" + 0.028"dental" ' '+ 0.022"also" + 0.020"has" + 0.020"been" + 0.018"health" + ' '0.016"virus"'), (6, '0.051"la" + 0.031"en" + 0.025"2" + 0.023"3" + 0.016"que" + 0.016"el" ' '+ 0.016"y" + 0.014"los" + 0.014"4" + 0.013"les"'), (7, '0.045"s" + 0.041"et" + 0.031"during" + 0.023"al" + 0.022"had" + ' '0.021"people" + 0.020"à" + 0.018"local" + 0.017"days" + 0.016"2020"'), (8, '0.062"patients" + 0.030"treatment" + 0.028"care" + 0.020"used" + ' '0.014"clinical" + 0.014"infection" + 0.013"common" + 0.013"severe" + ' '0.013"respiratory" + 0.012"dentistry"'), (9, '0.030"using" + 0.020"areas" + 0.018"ct" + 0.014"described" + ' '0.014"performed" + 0.013"lesions" + 0.013"above" + 0.012"day" + ' '0.011"learning" + 0.011"reactions"')]
def create_articles_metadata_mapping(sentences_df: pd.DataFrame) -> dict:
    sentence_id_to_metadata = {}
    for row_count, row in sentences_df.iterrows():
        sentence_id_to_metadata[row_count] = dict(
            paper_id=row['paper_id'],
            cord_uid=row['cord_uid'],
            source=row['source'],
            url=row['url'],
            publish_time=row['publish_time'],
            authors=row['authors'],
            section=row['section'],
            sentence=row['sentence'],
        )
    return sentence_id_to_metadata
sentence_id_to_metadata = create_articles_metadata_mapping(sentences_df)
import operator
from datetime import datetime

class SearchEngine:
    def __init__(self,
                 sentence_id_to_metadata: dict,
                 sentences_df: pd.DataFrame,
                 bigram_model,
                 trigram_model,
                 fasttext_model):
        self.sentence_id_to_metadata = sentence_id_to_metadata
        self.cleaned_sentences = sentences_df['cleaned_sentence'].tolist()
        print(f'Loaded {len(self.cleaned_sentences)} sentences')

        self.bigram_model = bigram_model
        self.trigram_model = trigram_model
        self.fasttext_model = fasttext_model

    def _get_search_terms(self, keywords, synonyms_threshold):
        # clean tokens
        cleaned_terms = [clean_tokenized_sentence(keyword.split(' ')) for keyword in keywords]
        # remove empty terms
        cleaned_terms = [term for term in cleaned_terms if term]
        # create bi-grams
        terms_with_bigrams = self.bigram_model[' '.join(cleaned_terms).split(' ')]
        # create tri-grams
        terms_with_trigrams = self.trigram_model[terms_with_bigrams]
        # expand query with synonyms
        search_terms = [self.fasttext_model.most_similar(token) for token in terms_with_trigrams]
        # filter synonyms above threshold (and flatten the list of lists)
        search_terms = [synonym[0] for synonyms in search_terms for synonym in synonyms
                        if synonym[1] >= synonyms_threshold]
        # expand keywords with synonyms
        search_terms = list(terms_with_trigrams) + search_terms
        return search_terms

    def search(self,
               keywords: List[str],
               optional_keywords=None,
               top_n: int = 10,
               synonyms_threshold=0.7,
               keyword_weight: float = 3.0,
               optional_keyword_weight: float = 0.5) -> List[dict]:
        if optional_keywords is None:
            optional_keywords = []

        search_terms = self._get_search_terms(keywords, synonyms_threshold)

        optional_search_terms = self._get_search_terms(optional_keywords, synonyms_threshold) \
            if optional_keywords else []

        print(f'Search terms after cleaning, bigrams, trigrams and synonym expansion: {search_terms}')
        print(f'Optional search terms after cleaning, bigrams, trigrams and synonym expansion: {optional_search_terms}')

        date_today = datetime.today()

        # calculate score for each sentence. Take only sentence with at least one match from the must-have keywords
        indexes = []
        match_counts = []
        days_diffs = []
        for sentence_index, sentence in enumerate(self.cleaned_sentences):
            sentence_tokens = sentence.split(' ')
            sentence_tokens_set = set(sentence_tokens)
            match_count = sum([keyword_weight if keyword in sentence_tokens_set else 0
                               for keyword in search_terms])
            if match_count > 0:
                indexes.append(sentence_index)
                if optional_search_terms:
                    match_count += sum([optional_keyword_weight if keyword in sentence_tokens_set else 0
                                       for keyword in optional_search_terms])
                match_counts.append(match_count)
                article_date = self.sentence_id_to_metadata[sentence_index]["publish_time"]

                if article_date == "2020":
                    article_date = "2020-01-01"

                article_date = datetime.strptime(article_date, "%Y-%m-%d")
                days_diff = (date_today - article_date).days
                days_diffs.append(days_diff)

        # the bigger the better
        match_counts = [float(match_count)/sum(match_counts) for match_count in match_counts]

        # the lesser the better
        days_diffs = [(max(days_diffs) - days_diff) for days_diff in days_diffs]
        days_diffs = [float(days_diff)/sum(days_diffs) for days_diff in days_diffs]

        index_to_score = {}
        for index, match_count, days_diff in zip(indexes, match_counts, days_diffs):
            index_to_score[index] = 0.7 * match_count + 0.3 * days_diff

        # sort by score descending
        sorted_indexes = sorted(index_to_score.items(), key=operator.itemgetter(1), reverse=True)

        # take only the sentence IDs
        sorted_indexes = [item[0] for item in sorted_indexes]

        # limit results
        sorted_indexes = sorted_indexes[0: min(top_n, len(sorted_indexes))]

        # get metadata for each sentence
        results = []
        for index in sorted_indexes:
            results.append(self.sentence_id_to_metadata[index])
        return results
search_engine = SearchEngine(sentence_id_to_metadata, sentences_df, bigram_model, trigram_model, fasttext_model)
def search(keywords, optional_keywords=None, top_n=10, synonyms_threshold=0.8, only_sentences=False):
    print(f"\nSearch for terms {keywords}\n\n")
    results = search_engine.search(
        keywords, optional_keywords=optional_keywords, top_n=top_n, synonyms_threshold=synonyms_threshold
    )
    print("\nResults:\n")
    
    if only_sentences:
        for result in results:
            print(result['sentence'] + "\n")
    else:
        pprint(results)
search(keywords=["spillover", "bats", "snakes", "exotic animals", "seafood"],
       optional_keywords=["new coronavirus", "coronavirus", "covid19"],
      top_n=3)
task_id = 1
import json

with open(f"../input/covid19seedsentences/{task_id}.json") as in_fp:
    seed_sentences_json = json.load(in_fp)

print(seed_sentences_json['taskName'])
corpus_index_path = "../input/covid19corpusindex/"
import six
import io


class WordFreq:
    """
    A dict-like object to hold word frequencies.

    Usage example:

    freqs = WordFreq.from_counts('/path/to/word_freq.txt')
    freqs['the']
    0.0505408583229405

    Once created you can use it for weighted average sentence encoding:

    encoder = SentenceEncoder(..., word_freq=freqs.__getitem__)
    """

    def __init__(self, word_freq):
        self.word_freq = word_freq

    def __getitem__(self, arg):
        return self.word_freq.get(arg, 0.0)

    @classmethod
    def from_counts(cls, fname):
        total = 0
        cnts = dict()
        with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
            for line in fin:
                word, cnt = line.rstrip().split(' ')
                cnt = int(cnt)
                total += cnt
                cnts[word] = cnt
        word_freq = {word: cnt / total for word, cnt in cnts.items()}
        return cls(word_freq)


class SimpleEncoder:
    def __init__(self,
                 word_embeddings: dict,
                 word_embedding_dim: int = 100,
                 preprocessor: callable = lambda s: s,
                 tokenizer: callable = lambda s: s.split(),
                 word_freq: callable = lambda w: 0.0,
                 weighted: bool = True,
                 alpha: float = 1e-3):
        """
        Sentence encoder as a smooth average of word vectors.

        Args:
            word_embeddings (dict): map words to their vector representation.
            word_embedding_dim (int): word embedding size. default is 200.
            preprocessor (callable): optional, a callable to pre-process sentence before tokenizing into words.
            tokenizer (callable): optional, a callable which splits a sentence into words.
            word_freq (callable): optional, a callable which map a word to its frequency in range [0 - 1]
            weighted (bool): optional, whether or not to use weighted average. default is True.
            alpha (bool): smoothing alpha for Out-of-Vocab tokens.

        Usage example (1 - bag-of-words average):
        -----------------
            w2v_path = '/path/to/vectors.txt'
            encoder = SimpleEncoder.from_w2v(w2v_path)
            encoder.encode('a sentence is here')

        Usage example (2 - Smooth Inverse Frequency average):
        -----------------
            w2v_path = '/path/to/vectors.txt'
            word_freq = WordFreq.from_counts('/path/to/word_freq.txt')
            encoder = SimpleEncoder.from_w2v(w2v_path, weighted=True, word_freq=word_freq.__getitem__)
            encoder.encode('a sentence is here')

        Usage example (3 - Smooth Inverse Frequency average + removing 1st component):
        -----------------
            w2v_path = '/path/to/vectors.txt'
            word_freq = WordFreq.from_counts('/path/to/word_freq.txt')
            encoder = SimpleEncoder.from_w2v(w2v_path, weighted=True, word_freq=word_freq.__getitem__)
            corpus = ['sentence a', 'sentence b']
            emb = encoder.encode(corpus)
            encoder.components_ = svd_components(emb, n_components=1)
            emb = encoder.encode(corpus)  # re-calculate embeddings
            encoder.encode('a sentence is here')

        """
        # word embeddings (filename)
        self.word_embeddings = word_embeddings

        # word embedding dim (e.g 200)
        self.word_embedding_dim = word_embedding_dim

        # sentence tokenizer (callable)
        self.tokenizer = tokenizer

        # preprocessor (callable)
        self.preprocessor = preprocessor

        # word frequency (callable)
        self.word_freq = word_freq

        # yes/no: tf-idf weighted average
        self.weighted = weighted

        # smoothing alpha
        self.alpha = alpha

        # principal components (pre-calc)
        self.components_ = None

    def __str__(self):
        components_dim = self.components_.shape if self.components_ is not None else None
        return (f"<SimpleEncoder(dim={self.word_embedding_dim}, "
                f"weighted={self.weighted}, "
                f"alpha={self.alpha}, "
                f"components_dim={components_dim})>")

    @classmethod
    def load(cls, w2v_path, word_count_path, principal_components_path):
        """Initialize an instance of `cls`.

        Returns:
            SimpleEncoder
        """
        encoder = cls.from_w2v(w2v_path,
                               tokenizer=lambda s: s.split(),
                               preprocessor=lambda s: s)

        encoder.load_word_counts(word_count_path)

        encoder.load_components(principal_components_path)

        return encoder

    @classmethod
    def from_w2v(cls, w2v_path, **init_kwargs):
        """Create a sentence encoder from word embeddings saved to disk.

        Args:
            w2v_path (str): filename of the word vectors.
            init_kwargs: additional keyword arguments to ```init``` method.

        Returns:
            SimpleEncoder
        """
        word_embeddings = {}
        with open(w2v_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
            _, dim = map(int, fin.readline().split())
            for line in fin:
                tokens = line.rstrip().split(' ')
                word_embeddings[tokens[0]] = np.array(tokens[1:], np.float32)
        return cls(word_embeddings=word_embeddings, word_embedding_dim=dim, **init_kwargs)

    def load_word_counts(self, fname):
        """Load word count file and use it for td-idf weighted average.

        Notice that ```weighted`` must be set to ```True`` in order to use it.

        Args:
            fname (str): filename.
        """
        word_freq = WordFreq.from_counts(fname)
        self.word_freq = word_freq.__getitem__

    def load_components(self, fname):
        """Load pre-computed principal components from a file.

        Args:
            fname (str): filename (e.g 'components.npy').
        """
        fd = io.open(fname, mode="rb")
        self.components_ = np.load(fd)

    def encode(self, sentences) -> np.array:
        if isinstance(sentences, six.string_types):
            sentences = [sentences]
        emb = np.stack([self._encode(sentence) for sentence in sentences])
        if self.components_ is not None:
            emb = emb - emb.dot(self.components_.transpose()).dot(self.components_)
        return emb

    def _encode(self, sent: str) -> np.array:
        count = 0
        sent_vec = np.zeros(self.word_embedding_dim, dtype=np.float32)
        sent = self.preprocessor(sent)
        words = self.tokenizer(sent)
        for word in words:
            word_vec = self.word_embeddings.get(word)
            if word_vec is None:
                continue
            norm = np.linalg.norm(word_vec)
            if norm > 0:
                word_vec *= (1.0 / norm)
            if self.weighted:
                freq = self.word_freq(word)
                word_vec *= self.alpha / (self.alpha + freq)
            sent_vec += word_vec
            count += 1
        if count > 0:
            sent_vec *= (1.0 / count)
        return sent_vec
import nmslib
import numpy as np
from sklearn.decomposition import TruncatedSVD


def linalg_pca(X):
    """PCA transformation with ```np.linalg``` (Singular Value Decomposition).

    Args:
        X (np.array): 2d array.

    Returns:
        np.array (2d)
    """
    # reduce mean
    X -= np.mean(X, axis=0)
    # compute covariance matrix
    cov = np.cov(X, rowvar=False)
    # compute eigen values & vectors
    eigen_vals, eigen_vectors = np.linalg.eigh(cov)
    # sort eigen vectors by eigen values
    idx = np.argsort(eigen_vals)[::-1]
    eigen_vectors = eigen_vectors[:, idx]
    return np.dot(X, eigen_vectors)


class Aggregation:
    UNION = 'union'
    AVG = 'average'
    MEAN = 'mean'
    PC_1 = 'pc_1'
    PC_2 = 'pc_2'

    
def nn_iter(indices, distances, black=None):
    """Helper method to unpack nearest neighbors lists.

    Args:
        indices (list): neighbor indices.
        distances (list): neighbor distances.
        black (list): black list.

    Returns:
        list(tuple)
    """
    for idx, dist in zip(indices, distances):
        if dist <= 0.0:
            continue
        if black is not None and idx in black:
            continue
        yield int(idx), float(dist)


class NMSLibCorpusIndex:
    def __init__(self, dim, metric='cosinesimil', **index_params):
        """Init ```nmslib.FloatIndex```.

        References
        -----------

        1) Installation

        https://github.com/nmslib/nmslib/tree/master/python_bindings#installation

        2) Supported metrics

        https://github.com/nmslib/nmslib/blob/master/manual/spaces.md

        3) Index params

        https://github.com/nmslib/nmslib/blob/master/manual/methods.md#graph-based-search-methods-sw-graph-and-hnsw
        """
        self.dim = dim
        self.index = nmslib.init(method='hnsw', space=metric)
        self.index_params = index_params or {'post': 0}  # {'post': 2, 'efConstruction': 200, 'M': 25}
        self._knn_batch_method = frozenset([Aggregation.UNION, Aggregation.AVG, Aggregation.MEAN, Aggregation.PC_1,Aggregation.PC_2])

    def __len__(self):
        return len(self.index)

    def __repr__(self):
        return f"<NMSLibCorpusIndex(size={self.__len__()})>"

    def load(self, fname, **kwargs):
        """Load an index from disk.

        Args:
            fname (str): filename.
            kwargs: additional keyword arguments.
        """
        self.index.loadIndex(fname, **kwargs)

    def save(self, fname, **kwargs):
        """Save index to disk.

        Args:
            fname (str): filename.
            kwargs: additional keyword arguments.
        """
        self.create_index()
        self.index.saveIndex(fname, save_data=True)

    def create_index(self):
        """Create ANN Index."""
        self.index.createIndex(self.index_params, print_progress=True)

    def get_vector_by_id(self, idx):
        """Get vector from index by id.

        Args:
            idx (int): vector id.

        Returns:
            np.array.
        """
        return np.array(self.index[idx], np.float32)

    def add_dense(self, dense, ids=None):
        """Add a batch of vectors to the index.

        Args:
            dense (array-like): array like of vectors (each is a ``np.array``).
            ids (array-like): array like of indices (each is a ``int``).
        """
        self._check_dim(dense)

        index_len = self.__len__()
        self.index.addDataPointBatch(
            data=dense,
            ids=ids if ids is not None else np.arange(index_len, index_len + dense.shape[0]))

    def knn_query(self, vec, ids=None, limit=10):
        """Find a set of approximate nearest neighbors to ``vec``.

        Args:
            vec (np.array): input vector.
            ids (list): optional, list of indices to filter out from result.
            limit (int): optional, limit result set size.

        Returns:
            list[tuple] = (neighbor_id, distance)
        """
        self._check_dim(vec)

        indices, distances = self.index.knnQuery(vec, k=limit * 2)
        return sorted(nn_iter(indices, distances, black=ids), key=operator.itemgetter(1))[:limit]

    def _check_batch_method(self, method):
        assert method in self._knn_batch_method, f"Invalid KNN batch method: {method}"

    def knn_query_batch(self, dense, ids=None, limit=10, method='union'):
        """Find a set of approximate nearest neighbors to ``dense``.

        If ```method``` is 'union', than this set will be the top-slice of the union set of all nearest neighbors.

        If ```method``` is 'mean', than this set equals to
            ```self.knn_query(np.mean(dense, axis=0), ids=ids, limit=limit)```.

        If ```method``` is 'pc_1', than this set equals to
            ```self.knn_query(_linalg_pca(dense)[0], ids=ids, limit=limit)```.

        If ```method``` is 'pc_2', than this set equals to
            ```self.knn_query(_linalg_pca(dense)[1], ids=ids, limit=limit)```.

        Args:
            dense (array-like): array like of vectors (each is a ``np.array``).
            ids (iterable): optional, list of indices to filter out from result.
            limit (int): optional, limit result set size.
            method (str): optional

        Returns:
            list[tuple] = (neighbor_id, distance)
        """
        self._check_batch_method(method)

        if method == 'mean':
            return self.knn_query(np.mean(dense, axis=0), ids=ids, limit=limit)
        elif method == 'pc_1':
            return self.knn_query(linalg_pca(dense)[0], ids=ids, limit=limit)
        elif method == 'pc_2':
            return self._knn_query_batch(linalg_pca(dense)[:1], ids=ids, limit=limit)
        elif method == 'union':
            return self._knn_query_batch(dense, ids=ids, limit=limit)
        else:  # union
            return self._knn_query_batch(dense, ids=ids, limit=limit)

    def _knn_query_batch(self, dense, ids=None, limit=10):
        """Find the union set of approximate nearest neighbors to ``dense``.

        Args:
            dense (array-like): array like of vectors (each is a ``np.array``).
            ids (list): optional, list of indices to filter out from result.
            limit (int): optional, limit result set size.

        Returns:
            list[tuple] = (neighbor_id, distance)
        """
        self._check_dim(dense)

        nearest_neighbors = []
        for indices, distances in self.index.knnQueryBatch(dense, k=limit):
            for idx, dist in nn_iter(indices, distances, black=ids):
                nearest_neighbors.append((idx, dist))
        return sorted(nearest_neighbors, key=operator.itemgetter(1))[:limit]

    def _check_dim(self, dense):
        dim = getattr(self, 'dim', None)
        if dim:
            if len(dense.shape) == 2:
                dense_dim = dense.shape[1]
            else:
                dense_dim = dense.shape[0]

            assert dim == dense_dim, f"expected dense vectors shape to be {dim}, got {dense_dim} instead."
def infer_dimension_from_corpus_name(fname):
    match = re.search(r'(\d+)d', fname)
    if not match:
        raise ValueError(f'Could not detect index dimension from {fname}.')
    dim = int(match.group(1))
    return dim


def load_corpus_index(fname, dim=None, **load_kwargs):
    index = None
    if dim is None:
        dim = infer_dimension_from_corpus_name(fname)
    if 'nmslib' in fname:
        index = NMSLibCorpusIndex(dim=dim)
    index.load(fname, **load_kwargs)
    return index
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String

Base = declarative_base()

def get_session(conn):
    """Init DB Session.
    """
    connect_args = {}
    if conn.startswith('sqlite:///'):
        connect_args.update({'check_same_thread': False})
    engine = create_engine(conn, connect_args=connect_args)
    Session = sessionmaker(bind=engine)
    return Session()

class Sentence(Base):
    __tablename__ = 'sentences'

    id = Column(Integer, name='sentence_id', primary_key=True)
    sentence = Column(String)
    paper_id = Column(String, name='paper_id')
    cord_uid = Column(String, name='cord_uid')
    publish_time = Column(String, name='publish_time')

    def __repr__(self):
        return f"<Sentence(id={self.id}, sentence=\"{self.sentence}\")>"

    def to_dict(self):
        return {
            'id': self.id,
            'text': self.sentence,
            'paper_id': self.paper_id,
            'cord_uid': self.cord_uid,
            'publish_time': self.publish_time
        }
import murmurhash
import numpy as np
import scipy.spatial


class CovidSimilarity:
    def __init__(self, corpus_index, sentence_encoder, db_session, bigram_model=None, trigram_model=None):
        self.corpus_index = corpus_index
        self.sentence_encoder = sentence_encoder
        self.db_session = db_session
        self.bigram_model = bigram_model
        self.trigram_model = trigram_model

    def similar_k(self, input_sentences, limit=10, method='union', group_by='cosine'):
        """Find similar sentences.

        Args:
            input_sentences (str/list[str]): one or more input sentences.
            sentence_encoder  : encoder
            limit (int): limit result set size to ``limit``.
            corpus_index : type of corpus where to fetch the suggestions from
            db_session  : Database to get neighbors from
            method (str): aggregation method ('union', 'mean', 'pc1', 'pc2').
            group_by (str): distance metric to use to group the result set. Default is 'cosine'.

        Returns:
            list<dict>
        """
        res = []
        nearest = dict()

        cleaned_sentences = [clean_sentence(sentence) for sentence in input_sentences]
        
        if self.bigram_model and self.trigram_model:
            tokenzied_sentences = [sentence.split(' ') for sentence in cleaned_sentences]
            sentences_with_bigrams = self.bigram_model[tokenzied_sentences]
            sentences_with_trigrams = self.trigram_model[sentences_with_bigrams]
            cleaned_sentences = [' '.join(sentence) for sentence in sentences_with_trigrams]

        embeddings = self.sentence_encoder.encode(cleaned_sentences)
        indices = [murmurhash.hash(sent) for sent in cleaned_sentences]

        for idx, dist in self.corpus_index.knn_query_batch(embeddings, ids=indices, limit=limit, method=method):
            if idx not in nearest:
                nearest[idx] = dist
            else:
                nearest[idx] = min(nearest[idx], dist)

        for sentence in self.db_session.query(Sentence).filter(Sentence.id.in_(nearest.keys())).all():
            sentence_dict = sentence.to_dict()
            encoding = sentence_encoder.encode(sentence.sentence)
            distances = scipy.spatial.distance.cdist(encoding, embeddings, group_by)
            nearest_idx = int(np.argmax(distances))
            sentence_dict['nearest'] = indices[nearest_idx]
            sentence_dict['dist'] = nearest[sentence.id]
            res.append(sentence_dict)

        return {
            'results': sorted(res, key=lambda x: x['dist']),
            'sentences': [
                {
                    'id': sent_id,
                    'text': sent
                } for sent_id, sent in zip(indices, cleaned_sentences)
            ]
        }
db_session = get_session(conn=f"sqlite:///{os.path.join(corpus_index_path, 'covid19.sqlite')}")
corpus_index = load_corpus_index(os.path.join(corpus_index_path, 'simple-encoder-nmslib-100d.bin'))
sentence_encoder = SimpleEncoder.load(
    os.path.join(fasttext_model_dir, "word-vectors-100d.txt"),
    os.path.join(fasttext_model_dir, "word-counts.txt"),
    os.path.join(corpus_index_path, "simple-encoder-100d-components.npy")
)
covid_similarity = CovidSimilarity(corpus_index, sentence_encoder, db_session, bigram_model, trigram_model)
sentences= ["Origination of 2019-nCoV from bats has been strongly supported but the presumed intermediate host remain to be identified initial reports that 2019-nCoV had an origin in snakes have not been verified",
           "For example, farmed palm civets were suggested to be an intermediate host for SARS to be spilled over to humans although the details on how to link bat and farmed palm civets are unclear [15, 16, 17]"]
covid_similarity.similar_k(sentences, limit=3, method="union")
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

class BartSummarizer:
    def __init__(self):
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_name = 'bart-large-cnn'
        print(f'Initializing BartTokenizer with model: {model_name} ...')
        self.tokenizer_summarize = BartTokenizer.from_pretrained(model_name)
        print(f'Finished initializing BartTokenizer with model: {model_name}')

        print(f'Initializing BartForConditionalGeneration with model: {model_name} ...')
        self.model_summarize = BartForConditionalGeneration.from_pretrained(model_name)
        print(f'Finished initializing BartForConditionalGeneration with model: {model_name}')
        self.model_summarize.to(self.torch_device)
        self.model_summarize.eval()

    def create_summary(self, text: str,
                       repetition_penalty=1.0) -> str:
        text_input_ids = self.tokenizer_summarize.batch_encode_plus(
            [text], return_tensors='pt', max_length=1024)['input_ids'].to(self.torch_device)
        summary_ids = self.model_summarize.generate(text_input_ids,
                                                    num_beams=4,
                                                    length_penalty=1.2,
                                                    max_length=1024,
                                                    min_length=124,
                                                    no_repeat_ngram_size=4,
                                                    repetition_penalty=repetition_penalty)
        summary = self.tokenizer_summarize.decode(summary_ids.squeeze(), skip_special_tokens=True)
        return summary
# This will take time on the first time since it downloads the model
bart_summarizer = BartSummarizer()
answers_results = []
for sub_task_json in seed_sentences_json["subTasks"]:
    sub_task_description = sub_task_json["description"]
    print(f"Working on task: {sub_task_description}")
    best_sentences = sub_task_json["bestSentences"]
    relevant_sentences = covid_similarity.similar_k(best_sentences)
    relevant_sentences_texts = [result["text"] for result in relevant_sentences["results"]]
    sub_task_summary = bart_summarizer.create_summary(" ".join(best_sentences))
    answers_results.append(dict(sub_task_description=sub_task_description, relevant_sentences=relevant_sentences, sub_task_summary=sub_task_summary))
from IPython.display import display, HTML
pd.set_option('display.max_colwidth', 0)
def display_summary(summary: str):
    return display(HTML(f"<div>{summary}</div>"))

def display_sub_task_description(sub_task_description):
    return display(HTML(f"<h2>{sub_task_description}</h2>"))

def display_task_name(task_name):
    return display(HTML(f"<h1>{task_name}</h1>"))
def visualize_output(seed_sentences, sub_task_json, sentence_id_to_metadata):
    """
    Prints output for each sub-task
    """
    # print description
    display_sub_task_description(sub_task_json.get("sub_task_description"))
    display_summary(sub_task_json.get("sub_task_summary"))
    
    # print output sentences
    sentence_output = pd.DataFrame(sub_task_json.get('relevant_sentences').get('results'))
    sentence_output.rename(columns={"text": "Relevant Sentence","cord_id": "CORD UID",
                                    "publish_time": "Publish Time", "id": "row_id"}, inplace=True)
    sentence_output["URL"] = sentence_output["row_id"].apply(lambda row_id: sentence_id_to_metadata[row_id]["url"])
    sentence_output["Source"] = sentence_output["row_id"].apply(lambda row_id: sentence_id_to_metadata[row_id]["source"])
    
    display(sentence_output[['cord_uid', 'Source', 'Publish Time', 'Relevant Sentence', 'URL']])
display_task_name(seed_sentences_json["taskName"])
for idx, sub_task_json in enumerate(answers_results):
    visualize_output(seed_sentences_json["subTasks"][idx]["bestSentences"], sub_task_json, sentence_id_to_metadata)

def save_output(seed_sentences, sub_task_json, sentence_id_to_metadata):
    """
    Saves output for each sub-task
    """
    
    # print output sentences
    sentence_output = pd.DataFrame(sub_task_json.get('relevant_sentences').get('results'))
    sentence_output.rename(columns={"text": "Relevant Sentence","cord_id": "CORD UID",
                                    "publish_time": "Publish Time", "id": "row_id"}, inplace=True)
    sentence_output["URL"] = sentence_output["row_id"].apply(lambda row_id: sentence_id_to_metadata[row_id]["url"])
    sentence_output["Source"] = sentence_output["row_id"].apply(lambda row_id: sentence_id_to_metadata[row_id]["source"])
    
    return sentence_output[['cord_uid', 'Source', 'Publish Time', 'Relevant Sentence', 'URL']]
relevant_sentences = []
for idx, sub_task_json in enumerate(answers_results):
    task_sentences = save_output(seed_sentences_json["subTasks"][idx]["bestSentences"], sub_task_json, sentence_id_to_metadata)
    relevant_sentences.append(task_sentences)
all_relevant_sentences = pd.concat(relevant_sentences).reset_index()
all_relevant_sentences.head(1)
all_relevant_sentences.shape
all_relevant_sentences['Relevant Sentence'][0]
lemmatized_sentences = []

for i in range(len(all_relevant_sentences['Relevant Sentence'])):
    remove_stop_words = clean_sentence(all_relevant_sentences['Relevant Sentence'][i])
    
    lemmatized_sentences.append(bigram_model[remove_stop_words.split(' ')])
lemmatized_sentences[:1]
import gensim.corpora as corpora
# Create Dictionary
id2word = corpora.Dictionary(lemmatized_sentences)

# Create Corpus
texts = lemmatized_sentences

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])
# Create Dictionary
id2word = corpora.Dictionary(lemmatized_sentences)

# Create Corpus
texts = lemmatized_sentences

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])
# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=15, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=lemmatized_sentences, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
%matplotlib inline
# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis