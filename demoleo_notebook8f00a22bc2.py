# import sys  

# reload(sys)  

# sys.setdefaultencoding('utf8')  
import numpy as np

import pandas as pd

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
sentences = [tokenizer.tokenize(x) for x in df_all['all_texts'].values]

sentences = [y for x in sentences for y in x]
from nltk.tokenize import word_tokenize

w2v_corpus = [word_tokenize(x) for x in sentences]
from gensim.models.word2vec import Word2Vec

model = Word2Vec(w2v_corpus, size=128, window=5, min_count=5, workers=4)
# 先拿到全部的vocabulary

vocab = model.vocab



# 得到任意text的vector

def get_vector(text):

    # 建立一个全是0的array

    res =np.zeros([128])

    count = 0

    for word in word_tokenize(text):

        if word in vocab:

            res += model[word]

            count += 1

    return res/count  
from scipy import spatial

# 这里，我们再玩儿个新的方法，用scipy的spatial

def w2v_cos_sim(text1, text2):

    try:

        w2v1 = get_vector(text1)

        w2v2 = get_vector(text2)

        sim = 1 - spatial.distance.cosine(w2v1, w2v2)

        return float(sim)

    except:

        return float(0)

# 这里加个try exception，以防我们得到的vector是个[0,0,0,...]
# word2vec similarity

df_all['w2v_cos_sim_in_title'] = df_all.apply(lambda x: w2v_cos_sim(x['search_term'], x['product_title']), axis=1)

df_all['w2v_cos_sim_in_desc'] = df_all.apply(lambda x: w2v_cos_sim(x['search_term'], x['product_description']), axis=1)