# GLOVE_EMBEDDING_PATH = '../input/glove840b300dtxt/glove.840B.300d.txt'



# def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')



# def load_embeddings(embed_dir):

#     embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in tqdm(open(embed_dir)))

#     return embedding_index



# glove = load_embeddings(GLOVE_EMBEDDING_PATH)
import pickle

from time import time



t = time()

with open('../input/glove.840B.300d.pkl', 'rb') as fp:

    glove = pickle.load(fp)

print(time()-t)
len(glove)
list(glove.keys())[0]
glove[',']