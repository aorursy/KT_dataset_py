import os
import numpy as np
from os import path
from urllib.request import urlretrieve
import sklearn
from sklearn.manifold import TSNE
%matplotlib inline
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize
import unidecode
EPSILON = 1e-15
nltk.download('punkt')
class PretrainedEmbeddings():
    def __init__(self, language, embeddings):
        self.vec_file = None
        if language == 'en':
            if embeddings == 'glove':
                self.vec_file = 'glove_100k.en.vec'
            elif embeddings == 'ft':
                self.vec_file = 'ft_300k.en.vec'
        elif language == 'fr':
            if embeddings == 'glove':
                print('No GloVe french embeddings!')
                return None
            elif embeddings == 'ft':
                self.vec_file = 'ft_50k.fr.vec'
        self.language = language
        self.url = "https://github.com/ECE-Deep-Learning/courses_labs/releases/download/0.1/" + self.vec_file
        self.file_location = os.path.join('../input/', self.vec_file)
        self.embeddings_index = None
        self.embeddings_index_inversed = None
        self.embeddings_vectors = None
        self.voc_size = None
        self.dim = None
    
    @staticmethod
    def _normalize(array):
        return array / np.linalg.norm(array, axis=-1, keepdims=True)
        
    def download(self):
        if not path.exists(self.file_location):
            print('Downloading from %s to %s...' % (self.url, self.file_location))
            urlretrieve(self.url, self.file_location)
            print('Downloaded embeddings')        
            
    # Note that you can choose to normalize directly the embeddings 
    # to make the cosine similarity computation easier afterward
    def load(self, normalize=False):
        self.embeddings_index, self.embeddings_index_inversed = {}, {}
        self.embeddings_vectors = []
        file = open(self.file_location, encoding='utf-8')
        header = next(file)
        self.voc_size, self.dim = [int(token) for token in header.split()]
        print('Vocabulary size: {0}\nEmbeddings dimension: {1}'.format(self.voc_size, self.dim))
        print('Loading embeddings in memory...')
        for idx, line in enumerate(file):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = idx
            self.embeddings_index_inversed[idx] = word
            self.embeddings_vectors.append(vector)
        self.embeddings_vectors = np.asarray(self.embeddings_vectors)
        print('Embeddings loaded')
        if normalize:
            self.embeddings_vectors = self._normalize(self.embeddings_vectors)
            print('Embeddings normalized')
        file.close()
        
    # Return an embedding vector associated to a given word
    # For this you are supposed to used the objects defined in the load function
    # Be sure to handle the case where the received word is not found in the embeddings' vocabulary
    def word_to_vec(self, word):
        # TODO:
        if word in self.embeddings_index:
            return self.embeddings_vectors[self.embeddings_index[word]]
        else:
            return None
        #else:
            #print("Error : word is not found in the embeddings' vocabulary")
    
    # Return the closest word associated to a given embedding vector
    # The vector passed as argument might not be in self.embeddings_vectors
    # In other terms, you have to compute every cosine similarity between the vec argument
    # and the embeddings found in self.embeddings_vectors. Then determine the embedding in 
    # self.embeddings_vectors with the highest similarity and return its associated string word
    def vec_to_word(self, vec, n=1):
        # TODO:
        #First Technique # seem to have the same result than the second one.
        #w2 = self.embeddings_vectors
        #resultat = []
        #cosine = np.dot(w2, vec.T) / (np.linalg.norm(w2) * np.linalg.norm(vec))
        #topResultats = cosine.argsort()[::-1][:n]
        #for vec in topResultats:
        #    resultat.append(self.embeddings_index_inversed[vec])
        #return resultat
    
        cos = np.dot(self.embeddings_vectors, np.transpose(vec))/(np.linalg.norm(self.embeddings_vectors)*np.linalg.norm(vec))
        words = [self.embeddings_index_inversed[item] for item in reversed(np.argsort(cos)[-n:])]
        if n==1: return words[0]
        else: return words

    # Return the n top similar words from a given string input
    # The similarities are based on the cosine similarities between the embeddings vectors
    # Note that the string could be a full sentence composed of several words
    # Split the sentence, map the words that can be found in self.embeddings_vectors to vectors and
    # average them. Then return the top (default: top=10) words associated to the top embeddings 
    # in self.embeddings_vectors that have the highest cosine similarity with the previously computed average
    def most_similar(self, query, top=10):
        # TODO:
        vecs = []
        for word in query.split():
            vecs.append(self.word_to_vec(word))
        return self.vec_to_word(np.mean(vecs, axis=0), top)
    
    def project_and_visualize(self, sample=1000):
        embeddings_tsne = TSNE(perplexity=30).fit_transform(self.embeddings_vectors[:sample])
        plt.figure(figsize=(40, 40))
        axis = plt.gca()
        np.set_printoptions(suppress=True)
        plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], marker=".", s=1)
        for idx in range(sample):
            plt.annotate(
                self.embeddings_index_inversed[idx],
                xy=(embeddings_tsne[idx, 0], embeddings_tsne[idx, 1]),
                xytext=(0, 0), textcoords='offset points'
            )
pretrained_embeddings = PretrainedEmbeddings(language='en', embeddings='glove')
#pretrained_embeddings.download()
pretrained_embeddings.load(normalize=True)
vec = pretrained_embeddings.word_to_vec("annual")
pretrained_embeddings.vec_to_word(vec)
pretrained_embeddings.project_and_visualize()
pretrained_embeddings.most_similar('french city')
class LanguageModel():
    def __init__(self):
        self.corpus_path = None
        self.corpus = None
    
    def load_data(self, corpus_path):
        self.corpus_path = os.path.join('../input/', corpus_path)
        file = open(self.corpus_path, encoding="utf-8")
        self.corpus = unidecode.unidecode(file.read().lower().replace("\n", " "))
        print('Corpus length: {0} characters'.format(len(self.corpus)))
        file.close()
    
    def get_contiguous_sample(self, size):
        index = np.random.randint(1, len(self.corpus) - size)
        return self.corpus[index:index+size]
sample_size = 500

language_model = LanguageModel()
language_model.load_data('rousseau.txt')
print('Sample of {0} characters:\n{1}'.format(
    sample_size, language_model.get_contiguous_sample(sample_size)
))
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

class CharLanguageModel(LanguageModel):
    def __init__(self):
        super(LanguageModel, self).__init__()
        self.char_index = None
        self.char_index_inversed = None
        self.vocabulary_size = None
        self.max_length_sequence = None
        self.X = None
        self.y = None
        self.model = None
    
    def extract_vocabulary(self):
        chars = sorted(set(self.corpus))
        self.char_index = dict((c, i) for i, c in enumerate(chars))
        self.char_index_inversed = dict((i, c) for i, c in enumerate(chars))
        self.vocabulary_size = len(self.char_index)
        print('Vocabulary size: {0}'.format(self.vocabulary_size))
        
    def plot_vocabulary_distribution(self):
        counter = Counter(self.corpus)
        chars, counts = zip(*counter.most_common())
        indices = np.arange(len(counts))
        plt.figure(figsize=(16, 5))
        plt.bar(indices, counts, 0.8)
        plt.xticks(indices, chars)
        
    """
    Convert X and y into one-hot encoded matrices
    
    Importante note: if the sequence length if smaller than max_length_sequence, 
    we pad the input with zeros vectors at the beginning of the one-hot encoded matrix
    """
    def _one_hot_encoding(self, X, y):
        X_one_hot = np.zeros(
            (len(X), self.max_length_sequence, self.vocabulary_size), 
            dtype=np.float32
        )
        y_one_hot = np.zeros(
            (len(X), self.vocabulary_size), 
            dtype=np.float32
        )
        # Leave above code as it is, change X_one_hot and y_one_hot below
        # TODO:
        for i in range(len(X)):
            for j in range(len(X[i])):
                X_one_hot[i][self.max_length_sequence - len(X[i]) + j][self.char_index.get(X[i][j])] = 1
        
        if y is not None:
            for i in range(len(y)):
                y_one_hot[i][self.char_index.get(y[i])] = 1
        return X_one_hot, y_one_hot 
    
    """
    The matrices X and y are created in this method
    It consists of sampling sentences in the corpus as training vectors with the next character as target
    """
    def build_dataset(self, 
                      max_length_sequence=40, min_length_sentence=5, max_length_sentence=200, 
                      step=3):
        self.X, self.y = [], []
        
        sentences = sent_tokenize(self.corpus)
        sentences = filter(
            lambda x: len(x) >= min_length_sentence and len(x) <= max_length_sentence, 
            sentences
        )
        for sentence in sentences:
            for i in range(0, max(len(sentence) - max_length_sequence, 1), step):
                last_index = min(i+max_length_sequence, i+len(sentence)-1)
                self.X.append(sentence[i:last_index])
                self.y.append(sentence[last_index])

        self.max_length_sequence = max_length_sequence
        self.X, self.y = sklearn.utils.shuffle(self.X, self.y)
        print('Number of training sequences: {0}'.format(len(self.X)))
        self.X, self.y = self._one_hot_encoding(self.X, self.y)
        print('X shape: {0}\ny shape: {1}'.format(self.X.shape, self.y.shape))
    
    """
    Define, compile, and fit a Keras model on (self.X, self.y)
    It should be composed of :
        - one recurrent LSTM layer projecting into hidden_size dimensions
        - one Dense layer with a softmax activation projecting into vocabulary_size dimensions
    """
    def train(self, hidden_size=128, batch_size=128, epochs=10):
        # TODO:
        self.model = Sequential()
        self.model.add(LSTM(units = hidden_size))
        self.model.add(Dense(units = self.vocabulary_size, activation="softmax"))
        self.model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

        self.model.fit(x=self.X, y=self.y, batch_size=batch_size, epochs=epochs,verbose=1)

    
    """
    Return the prediction of our model, meaning the next token given an input sequence
    
    If preprocessed is specified as True, we consider X as an array of strings and we will transform
    it to a one-hot encoded matrix
    Importante note: if the sequence length if smaller than max_length_sequence, 
    we pad the input with zeros vectors at the beginning of the one-hot encoded matrix
    
    If preprocessed is specified as False, we apply the model predict on X as it is
    """
    def predict(self, X, verbose=1, preprocessed=True):
        if not preprocessed:
            X_one_hot = np.zeros(
                (len(X), self.max_length_sequence, self.vocabulary_size), dtype=np.float32
            )
            # Leave above code as it is, change X_one_hot below
            # TODO:
            X_one_hot, _ = self._one_hot_encoding(X, None)
        else:
            X_one_hot = X
        return self.model.predict(X_one_hot, verbose=verbose)
    
    # Perplexity metric used to appreciate the performance of our model
    def perplexity(self, y_true, y_pred):
        likelihoods = np.sum(y_pred * y_true, axis=1)
        return 2 ** -np.mean(np.log2(likelihoods + EPSILON))
    
    """
    Sample the next character according to the predictions.
    
    Use a lower temperature to force the model to output more
    confident predictions: more peaky distribution.
    """
    def _sample_next_char(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds + EPSILON) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds + EPSILON)
        probs = np.random.multinomial(1, preds, size=1)
        return np.argmax(probs)
    
    def generate_text(self, seed_string, length=300, temperature=1.0):
        if self.model is None:
            print('The language model has not been trained yet!')
            return None
        generated = seed_string
        prefix = seed_string
        for i in range(length):
            predictions = np.ravel(self.predict([prefix], verbose=0, preprocessed=False))
            next_index = self._sample_next_char(predictions, temperature)
            next_char = self.char_index_inversed[next_index]
            generated += next_char
            prefix = prefix[1:] + next_char
        return generated
language_model = CharLanguageModel()
language_model.load_data('rousseau.txt')
language_model.extract_vocabulary()
language_model.char_index
language_model.plot_vocabulary_distribution()
language_model.build_dataset()
epochs = 5
language_model.train(epochs=epochs)
if language_model.model is not None:
    print('Perplexity after {0} epochs: {1}'.format(
        epochs, language_model.perplexity(language_model.y, language_model.model.predict(language_model.X))
    ))
language_model.generate_text("l'etat n'est pas au-dessus de la loi", temperature=0.25)
language_model.generate_text("la republique", temperature=0.25)
from spacy.lang.fr import French
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

class WordLanguageModel(LanguageModel):
    def __init__(self):
        super(LanguageModel, self).__init__()
        self.pretrained_embeddings = PretrainedEmbeddings(language='fr', embeddings='ft')
        self.pretrained_embeddings.download()
        self.pretrained_embeddings.load()
        self.parser = None
        self.word_index = None
        self.word_index_inversed = None
        self.vocabulary_size = None
        self.max_length_sequence = None
        self.tokens = None
        self.X = None
        self.y = None
        self.model = None
        
    def extract_vocabulary(self, max_vocabulary=1500000):
        self.parser = French(max_length=max_vocabulary)
        self.tokens = [token.orth_ for token in self.parser(self.corpus) if token.is_alpha]
        unique_tokens = set(self.tokens)
        self.word_index = dict((w, i) for i, w in enumerate(unique_tokens))
        self.word_index_inversed = dict((i, w) for i, w in enumerate(unique_tokens))
        self.vocabulary_size = len(self.word_index)
        print('Vocabulary size: {0}'.format(self.vocabulary_size))
        
    """
    Convert X and y into embedded matrices
    Hint: use the self.pretrained_embeddings.word_to_vec method for each token found
    
    Importante notes: 
    - if the sequence length if smaller than max_length_sequence, 
    we pad the input with zeros vectors at the beginning of the embedded matrix
    - if a word is not found in self.pretrained_embeddings.word_to_vec then word should be
    mapped to a vector of zeros instead
    """
    def _token_embedding(self, X, y):
        X_embedding = np.zeros(
            (len(X), self.max_length_sequence, self.pretrained_embeddings.dim), 
            dtype=np.float32
        )       
        y_one_hot = np.zeros(
            (len(X), self.vocabulary_size), 
            dtype=np.float32
        )
        # Leave above code as it is, change X_embedding and y_one_hot below
        # TODO:
        for i in range(len(X)):
            for j in range(len(X[i])):
                word = X[i][j]
                if word is not None:
                    if self.pretrained_embeddings.word_to_vec(word) is not None:
                        X_embedding[i][j] = self.pretrained_embeddings.word_to_vec(word)
                    
        if y is not None:
            for i in range(len(y)):
                y_one_hot[i][self.word_index.get(y[i])] = 1
        
        return X_embedding, y_one_hot
        
    def build_dataset(self, max_length_sequence=40, step=3):
        self.X, self.y = [], []
        for i in range(0, len(self.tokens) - max_length_sequence, step):
            self.X.append(self.tokens[i:i+max_length_sequence])
            self.y.append(self.tokens[i+max_length_sequence])
        self.max_length_sequence = max_length_sequence
        self.X, self.y = sklearn.utils.shuffle(self.X, self.y)
        print('Number of training sequences: {0}'.format(len(self.X)))
        self.X, self.y = self._token_embedding(self.X, self.y)
        print('X shape: {0}\ny shape: {1}'.format(self.X.shape, self.y.shape))
        
    """
    Define, compile, and fit a Keras model on (self.X, self.y)
    It should be composed of :
        - one or many recurrent LSTM layers projecting into hidden_size dimensions
        - one Dense layer with a relu activation projecting into hidden_size dimensions
        - one Dense layer with a softmax activation projecting into vocabulary_size dimensions
    """
    def train(self, hidden_size=128, batch_size=128, epochs=10):
        # TODO:
        self.model = Sequential()
        self.model.add(LSTM(units = hidden_size, return_sequences=True))
        self.model.add(LSTM(units = hidden_size))
        self.model.add(Dense(units = hidden_size, activation="relu"))
        self.model.add(Dense(units = self.vocabulary_size, activation="softmax"))
        self.model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
        self.model.fit(x=self.X,y=self.y, batch_size=batch_size, epochs=epochs,verbose=1)
        
    """
    Return the prediction of our model, meaning the next token given an input sequence
    
    If preprocessed is specified as True, we consider X as an array of strings and we will transform
    it to an embedded matrix using self.pretrained_embeddings.word_to_vec
    Importante note: if the sequence length if smaller than max_length_sequence, 
    we pad the input with zeros vectors at the beginning of the embedded matrix
    
    If preprocessed is specified as False, we apply the model predict on X as it is
    """
    def predict(self, X, verbose=1, preprocessed=True):
        if not preprocessed:
            X_embedding = np.zeros(
                (len(X), self.max_length_sequence, self.pretrained_embeddings.dim), 
                dtype=np.float32
            )
            # Leave above code as it is, change X_embedding
            # TODO:
            X_embedding, _ = self._token_embedding(X, None)
        else:
            X_embedding = X
        return self.model.predict(X_embedding, verbose=verbose)
    
    """
    Sample the next word according to the predictions.
    
    Use a lower temperature to force the model to output more
    confident predictions: more peaky distribution.
    """
    def _sample_next_word(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds + EPSILON) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds + EPSILON)
        probs = np.random.multinomial(1, preds, size=1)
        return np.argmax(probs)
    
    def generate_text(self, seed_string, length=50, temperature=1.0):
        if self.model is None:
            print('The language model has not been trained yet!')
            return None
        seed_tokens = [token.orth_ for token in self.parser(seed_string) if token.is_alpha]
        prefix = seed_tokens
        generated = seed_tokens
        for i in range(length):
            predictions = np.ravel(self.predict([prefix], verbose=0, preprocessed=False))
            next_index = self._sample_next_word(predictions)
            next_word = self.word_index_inversed[next_index]
            generated += [next_word]
            prefix = prefix[1:] + [next_word]
        return " ".join(generated)
language_model = WordLanguageModel()
language_model.load_data('rousseau.txt')
language_model.extract_vocabulary()
language_model.build_dataset()
epochs = 5
language_model.train(epochs=epochs)
language_model.generate_text("un état ne saurait réussir à", temperature=0.5)
language_model.generate_text("la république ne doit pas", temperature=1)
from sklearn.datasets import fetch_20newsgroups
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
print("sklearn object type : {}".format(type(newsgroups_train)))
print("sklearn object keys :")
for k in newsgroups_train:
    print(k)
print("Classes to predict : {}".format(os.linesep.join(newsgroups_train['target_names'])))
print()
print("Integer mapped-classes to predict :")
print(newsgroups_train['target'])
class_int_str = dict(
    zip(range(len(newsgroups_train['target_names'])), newsgroups_train['target_names'])
)
class_int_str
print("Example of document in dataset:", os.linesep)
sample_idx = np.random.randint(len(newsgroups_train["data"]))
print(newsgroups_train["data"][sample_idx])
sample_idx_class = class_int_str[newsgroups_train["target"][sample_idx]]
print("Example class to predict : {}".format(sample_idx_class))
MAX_NB_WORDS = 20000  # number of different integers mapping our vocabulary

# get the raw text data
texts_train = newsgroups_train["data"]
texts_test = newsgroups_test["data"]

# finally, vectorize the text samples into a 2D integer tensor of shape (nb_sequences, sequence_length)
# TODO:
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_train)
sequences_train = tokenizer.texts_to_sequences(texts_train)
sequences_test = tokenizer.texts_to_sequences(texts_test)

if tokenizer is not None:
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
print("First raw text example: ", os.linesep, texts_train[0])
if sequences_train is not None:
    print("First text conversion to token_ids: ", os.linesep, sequences_train[0])
    print("First text number of token_ids: {}".format(len(sequences_train[0])))
if tokenizer is not None:
    word_to_index = tokenizer.word_index.items()
    index_to_word = dict((i, w) for w, i in word_to_index)
if sequences_train is not None:
    print("Original sentence retrieved :", os.linesep)
    print(" ".join([index_to_word[i] for i in sequences_train[0]]))
MAX_SEQUENCE_LENGTH = 200

# pad 1-D sequences with 0s
# use the pad_sequences method on your sequences
# TODO:
x_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
if x_train is not None and x_test is not None:
    print('Shape of data tensor:', x_train.shape)
    print('Shape of data test tensor:', x_test.shape)
if x_train is not None:
    print("Example of tensor after padding/truncating : ", os.linesep, x_train[0])
y_train = newsgroups_train["target"]
y_test = newsgroups_test["target"]

# One-hot encode integer-mapped classes
y_train_onehot = to_categorical(np.asarray(y_train))
print('Shape of train target tensor:', y_train_onehot.shape)
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from keras.optimizers import Adam

EMBEDDING_DIM = 50
N_CLASSES = y_train_onehot.shape[1]

# TODO:
model = Sequential()
model.add(Embedding(input_dim=MAX_NB_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(GlobalAveragePooling1D())
model.add(Dense(units = N_CLASSES,activation="softmax"))
model.compile(optimizer=Adam(),loss="categorical_crossentropy",metrics=["accuracy"])
#model.compile(optimizer=Adam(),loss="categorical_crossentropy",metrics=['Accuracy', 'Loss'])
model.summary() if model is not None else None
if model is not None and x_train is not None:
    model.fit(x_train, y_train_onehot, validation_split=0.1,
              epochs=150, batch_size=128)
if model is not None and x_test is not None:
    print("test accuracy:", np.mean(model.predict(x_test).argmax(axis=-1) == y_test))
"""
Get an input tensor and replace the word->integer mapping with pretrained embeddings
Be sure that the word is existing (not a 0 padding) and is in the embeddings' vocabulary
"""

def preprocess_with_pretrained_embeddings(X, language, embeddings):
    pretrained_embeddings = PretrainedEmbeddings(language=language, embeddings=embeddings)
    pretrained_embeddings.download()
    pretrained_embeddings.load()
    X_embedding = np.zeros((X.shape[0], X.shape[1], pretrained_embeddings.dim))
    # TODO:
    for i in range(len(X)):
        for j in range(len(X[i])):
            word = index_to_word[X[i][j]] if X[i][j] != 0 else None
            if word is not None:
                if pretrained_embeddings.word_to_vec(word) is not None:
                    X_embedding[i][j] = pretrained_embeddings.word_to_vec(word)

    return X_embedding
x_train
if x_train is not None and x_test is not None:
    x_train_embedding = preprocess_with_pretrained_embeddings(x_train, language='en', embeddings='glove')
    x_test_embedding = preprocess_with_pretrained_embeddings(x_test, language='en', embeddings='glove')
    print('Embedded training matrix shape:', x_train_embedding.shape)
    print('Embedded test matrix shape:', x_test_embedding.shape)
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam

N_CLASSES = y_train_onehot.shape[1]
DIM_1 = x_train_embedding.shape[1]
DIM_2 = x_train_embedding.shape[2]

# TODO:
model = Sequential()
model.add(GlobalAveragePooling1D(input_shape=(DIM_1, DIM_2)))
model.add(Dense(units = N_CLASSES, activation="softmax"))
model.compile(optimizer=Adam(),loss="categorical_crossentropy",metrics=["accuracy"])
model.summary() if model is not None else None
if model is not None and x_train_embedding is not None:
    model.fit(x_train_embedding, y_train_onehot, validation_split=0.1, 
              epochs=200, batch_size=128)
if model is not None and x_test_embedding is not None:
    print("test accuracy:", np.mean(model.predict(x_test_embedding).argmax(axis=-1) == y_test))
from keras.models import Sequential
from keras.layers import Embedding, MaxPooling1D, LSTM, Dense
from keras.optimizers import Adam

EMBEDDING_DIM = 50
N_CLASSES = y_train_onehot.shape[1]
pooling_size = 5
hidden_size = 64

# TODO:
model = Sequential()
model.add(Embedding(input_dim=MAX_NB_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(MaxPooling1D(pooling_size))
model.add(LSTM(units = hidden_size))
model.add(Dense(units = N_CLASSES, activation="softmax"))
model.compile(optimizer=Adam(),loss="categorical_crossentropy",metrics=["accuracy"])
model.summary() if model is not None else None
if model is not None and x_train is not None:
    model.fit(x_train, y_train_onehot, validation_split=0.1, 
              epochs=25, batch_size=128)
if model is not None and x_test is not None:
    print("test accuracy:", np.mean(model.predict(x_test).argmax(axis=-1) == y_test))
