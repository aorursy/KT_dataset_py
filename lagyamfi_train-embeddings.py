import numpy as np
import unicodedata
import re
import os
from gensim.models import Word2Vec, FastText


NUMBER_OF_DATASET = 10000
ENG_PATH = "../jw300.en-tw.en"
TWI_PATH = "../jw300.en-tw.tw"

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def normalize_line(s, language="eng"):
    """
    Perform some cleanup on supplied str based on language.
    
    Parameters
    ----------
    s : str
    language: str
        default is "eng" for english. option is "twi"
    
    Returns
    -------
    str of cleaned sentence
    """
    s = unicode_to_ascii(s)
    s = re.sub(r'([!.?])', r' \1', s)
    s = s.lower()
    if language == "twi":
        s = re.sub(r'[^a-zA-Z.ƆɔɛƐ!?’]+', r' ', s)
    elif language == "eng":
        s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    return s

def read_dataset(file_path, number=None, normalize=False, language="eng"):
    """
    Read NUMBER_OF_DATASET lines of data in supplied file_path
    Perform normalization (if normalize=True) based on input language(default:"eng", option:"twi")

    Returns
    -------
    List[list] of processed word tokens for sentences in file_path
    """

    with open(file_path) as file:
        data = file.read()
    data = data.split("\n")
    if number:
        assert number < len(data), "Number of dataset less than required subset"
        data = data[:number]
    if normalize:
        data = [normalize_line(line, language=language).split() for line in data]
    return data

def get_embedding(data, typeFunc=Word2Vec, size=100, window=5, min_count=5, sg=0, save=False):
    """
    Generate embeddings for input data. Currently works with either word2vec or Fasttext from gensim

    Parameters
    ----------
    data : list[list]
        preprocessed word tokens
    typeFunc : gensim.model name
        Either Word2Vec or FastText (without "") (default is Word2Vec)
    size : int
        Dimension of embeddings to be generated (default=100)
    window : int
        size of window to be considered for word embeddings (default=5)
    min_count : int
        
    sg : int (0,1)
    
    save : bool
        if True, save generated embeddings in current working directory of script

    Returns
    -------
    Embeddings of type gensim.model
    """
    embeddings  = typeFunc(data,size=size, window=window, min_count=min_count, workers=4, sg=sg)
    if save:
        embeddings.save(f"./{typeFunc.__name}_embedding.mod")
    return embeddings

def get_similar(input_word:str, model, number=1):
    """
    Return number of similar words from learned word embeddings

    Parameters
    ----------
    input_word : str
        word to find similar words of
    model : gensim model type of learned word embeddings
    number : number of similar words to input_word to return

    Returns
    -------
    A list of tuple(s) of number of words similar to input_word and a score of the their closeness to input_word
    """
    assert isinstance(model, (Word2Vec, FastText)), "Model provided cannot be identified as either FastText or Word2Vec"
    sim_words = model.wv.most_similar(input_word)
    return sim_words[:number] # incremented by 1 as first returned value is the word itself


def prepare_for_visualization(model, model_path=None, save_dir="."):
    """
    Generates tsv formats of metadata and tensors/vectors for embeddings. 
    Useful for tensorflow embeddings projector.

    Parameters
    ----------
    model : gensim model type
        embeddings created using either word2vec or fasttext
    model_path : path, optional
        Path to a saved embeddings file (default is None)
    save_dir : path
        Path to directory to save created tsv files. (default is current working directory of script)

    Returns
    -------
    A tuple of tensors and metadata of embeddings
    """
    if model_path:   # to do -> check correctness of path
        model = gensim.models.KeyedVectors.load_word2vec_format(f"{model_path}", binary=False, encoding="utf-16")
    with open(f"{save_dir}/embedding_tensors.tsv", 'w+') as tensors:
        with open(f"{save_dir}/embedding_metadata.dat", 'w+') as metadata:
            for word in model.wv.index2word:
                #encoded=word.encode('utf-8')
                encoded = word
                metadata.write(encoded + '\n')
                vector_row = '\t'.join(map(str, model[word]))
                tensors.write(vector_row + '\n')
    return tensors, metadata
# Read twi data from supplied path to TWI file and preprocess
twi_data = read_dataset(TWI_PATH, NUMBER_OF_DATASET, normalize=True, language="twi")

# create embeddings from preprocessed twi data
embeddings = get_embedding(twi_data, FastText, size = 100, sg=1)

# test to see some similar words to some word using the learned embeddings
input_word = ""
get_similar(input_word, embeddings)

# generate tsv files for the tensors and the meta to be used for visualization
prepare_for_visualization(embeddings)
!ls ..
