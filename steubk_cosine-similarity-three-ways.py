import numpy as np 

import joblib
def batch_generator (embeddings, batch_size=1000):

    start = 0

    stop = 0

    while stop < embeddings.shape[0] :

        stop = stop + batch_size

        yield embeddings[start:stop]

        start = stop

        

        

train_embeddings = joblib.load("/kaggle/input/data-for-cosine-similarity/train_embeddings.joblib")

test_embeddings = joblib.load("/kaggle/input/data-for-cosine-similarity/test_embeddings.joblib")



print(f'train_embeddings:{train_embeddings.shape}, test_embeddings:{test_embeddings.shape}')

%%time

from scipy import spatial



scipy_similarity = np.zeros ((test_embeddings.shape[0], train_embeddings.shape[0]))



for test_index in range(test_embeddings.shape[0]):

    scipy_similarity[test_index] = 1 - spatial.distance.cdist(

        test_embeddings[np.newaxis, test_index, :], train_embeddings,

        'cosine')[0]

    

%%time



from scipy import spatial





batch_scipy_similarity = np.zeros ((test_embeddings.shape[0], train_embeddings.shape[0]))





test_batch_size=500

train_batch_size=1000

for i, test_emb in enumerate(batch_generator (test_embeddings, batch_size=test_batch_size)):

    for j, train_emb in enumerate(batch_generator (train_embeddings, batch_size=train_batch_size)):

        batch_scipy_similarity[i*test_batch_size:(i+1)*min(test_batch_size,test_emb.shape[0]),

                      j*train_batch_size:(j+1)*min(train_batch_size,train_emb.shape[0])] =  1 - spatial.distance.cdist(

                                        test_emb, train_emb,

                                        'cosine')

## check difference 

np.sum( np.abs (scipy_similarity) - np.abs(batch_scipy_similarity))
%%time



import tensorflow as tf

import numpy as np





tf_similarity = np.zeros ((test_embeddings.shape[0], train_embeddings.shape[0]))





test_batch_size=500

train_batch_size=1000

for i, test_emb in enumerate(batch_generator (test_embeddings, batch_size=test_batch_size)):

    for j, train_emb in enumerate(batch_generator (train_embeddings, batch_size=train_batch_size)):

        



        b = tf.constant (train_emb)

        a = tf.constant (test_emb)



        similarity = tf.reduce_sum(a[:, tf.newaxis] * b, axis=-1)

        similarity /= tf.norm(a[:, tf.newaxis], axis=-1) * tf.norm(b, axis=-1)

        

        tf_similarity[i*test_batch_size:(i+1)*min(test_batch_size,test_emb.shape[0]),

                      j*train_batch_size:(j+1)*min(train_batch_size,train_emb.shape[0])] = similarity.numpy()





## check difference 

np.sum(np.abs(scipy_similarity) - np.abs(tf_similarity))