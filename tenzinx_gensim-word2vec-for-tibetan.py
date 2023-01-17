from time import time  # To time our operations

from collections import defaultdict  # For word frequency

from pathlib import Path



import logging  # Setting up the loggings to monitor gensim

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
data_fn = Path('../input/tokenized_lemmatized_paragraphs.txt')
tokenized_paras = [para.split(' ') for para in data_fn.read_text().split('\n')]
tokenized_paras[0]
word_freq = defaultdict(int)

for para in tokenized_paras:

    for i in para:

        word_freq[i] += 1

len(word_freq)
sorted(word_freq, key=word_freq.get, reverse=True)[:10]
import multiprocessing



from gensim.models import Word2Vec
cores = multiprocessing.cpu_count() # Count the number of cores in a computer
w2v_model = Word2Vec(min_count=20,

                     window=5,

                     size=150,

                     sample=6e-5, 

                     alpha=0.03, 

                     min_alpha=0.0007, 

                     negative=20,

                     workers=cores-1)
t = time()



w2v_model.build_vocab(tokenized_paras, progress_per=10000)



print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
t = time()



w2v_model.train(tokenized_paras, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)



print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
w2v_model.init_sims(replace=True)
w2v_model.wv.most_similar(positive=["སྟོབས་"])
w2v_model.wv.most_similar(positive=["མཛད་པ་"])
w2v_model.wv.most_similar(positive=["བླ་མ་"])
w2v_model.wv.most_similar(positive=["རྩ་བ་"])
w2v_model.wv.most_similar(positive=["ཉིད་"])
w2v_model.wv.save_word2vec_format("./bo_word2vec_lammatized",

                              "./vocab",

                               binary=False)
!ls
from gensim.models import KeyedVectors
wv_from_text = KeyedVectors.load_word2vec_format('bo_word2vec_lammatized', binary=False)
wv_from_text.wv.most_similar(positive=["ཉིད་"])