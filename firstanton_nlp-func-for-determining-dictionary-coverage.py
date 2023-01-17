import pandas as pd

import numpy as np



from tqdm import tqdm

from keras.preprocessing.text import Tokenizer



tqdm.pandas()



train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
def build_vocab(texts):

    sentences = texts.progress_apply(lambda x: x.split()).values

    vocab = {}

    for sentence in sentences:

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab



def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')



def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)

    

embedding_index = load_embeddings("../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec")
tokenizer = Tokenizer(lower=False)

tokenizer.fit_on_texts(train.comment_text.tolist() + test.comment_text.tolist())
%%time

def check_coverage_new(word_counts, wanted_keys):

    a = {key: val for key, val in word_counts.items() if key not in wanted_keys}

    print(f'Found embeddings for {1-len(a)/len(word_counts):.2%} of vocablen')

    print(f'Found embeddings for {1-sum(a.values())/sum(word_counts.values()):.2%} of all text')

    return sorted(a.items(), key= lambda x : x[1], reverse=True)



wanted_keys = embedding_index.keys()

sort_x = check_coverage_new(tokenizer.word_counts, wanted_keys)
%%time



import operator 



def check_coverage(vocab, embeddings_index):

    a = {}

    oov = {}

    k = 0

    i = 0

    for word in tqdm(vocab):

        try:

            a[word] = embeddings_index[word]

            k += vocab[word]

        except:



            oov[word] = vocab[word]

            i += vocab[word]

            pass



    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))

    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))

    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]



    return sorted_x



sort_y = check_coverage(tokenizer.word_counts, embedding_index)
%%time

def check_coverage_new(word_counts, wanted_keys):

    a = {key: val for key, val in word_counts.items() if key not in wanted_keys}

    print(f'Found embeddings for {1-len(a)/len(word_counts):.2%} of vocablen')

    print(f'Found embeddings for {1-sum(a.values())/sum(word_counts.values()):.2%} of all text')

    return sorted(a.items(), key= lambda x : x[1], reverse=True)



sort_x = check_coverage_new(tokenizer.word_counts, wanted_keys)
sort_x[:10], sort_y[:10]