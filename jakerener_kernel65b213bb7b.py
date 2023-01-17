

import gensim
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def avg_sentence_vector(words, model, num_features, index2word_set):
    #function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features, 1), dtype="float32")
    nwords = 0

    for word in words:
        if word in index2word_set:
            nwords = nwords+1
            featureVec = np.add(featureVec, model[word])

    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('/kaggle/input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin', binary=True)  


#get average vector for sentence 1
sentence_1 = "I hate apples"
sentence_1_avg_vector = avg_sentence_vector(sentence_1.split(), model=word2vec_model, num_features=100, index2word_set=set(word2vec_model.index2word))

#get average vector for sentence 2
sentence_2 = "I do not like the taste of apples"
sentence_2_avg_vector = avg_sentence_vector(sentence_2.split(), model=word2vec_model, num_features=100, index2word_set=set(word2vec_model.index2word))

sen1_sen2_similarity =  cosine_similarity(sentence_1_avg_vector,sentence_2_avg_vector)

print(sen1_sen2_similarity[0][0])

