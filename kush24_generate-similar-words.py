

from tqdm import tqdm

import codecs

from scipy import spatial
#load Fasttext embeddings

print('loading word embeddings...')

embeddings_index = {}

f = codecs.open('../input/fasttext/wiki.simple.vec', encoding='utf-8')

for line in tqdm(f):

    values = line.rstrip().rsplit(' ')

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()

print('found %s word vectors' % len(embeddings_index))
#load Glove embeddings

print('loading word embeddings...')

embeddings_dicts = {}

f = codecs.open('../input/glove840b300dtxt/glove.840B.300d.txt', encoding='utf-8')

for line in tqdm(f):

    values = line.rstrip().rsplit(' ')

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_dicts[word] = coefs

f.close()

print('found %s word vectors' % len(embeddings_dicts))



!pip install fasttext



!git clone https://github.com/facebookresearch/fastText.git
!cd FastText
!pip install fastText
import fasttext.util
import fasttext

import fasttext.util

fasttext.util.download_model('en', if_exists='ignore')  # English

import fastText
ft = fastText.load_model('../input/fasttext-common-crawl-bin-model/cc.en.300.bin')
ft.get_word_vector("additional")
ft.get_nearest_neighbors('investment')

def find_closest_embeddings(embedding):

    return sorted(embeddings_index.keys(), key=lambda word: spatial.distance.euclidean(embeddings_index[word], embedding))
find_closest_embeddings(embeddings_index["culture"])[:30]
def find_closest_embeddings(embedding):

    return sorted(embeddings_dicts.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dicts[word], embedding))
find_closest_embeddings(embeddings_dicts["investment"])[:20]