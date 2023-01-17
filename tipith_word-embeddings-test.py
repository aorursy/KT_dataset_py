import numpy as np
import tqdm
!find .. -type f -exec du -ah {} +
!head -n3 ../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec
def load_emb(fname):
    embeddings = {}
    with open(fname) as f:
        for line in tqdm.tqdm(f, unit=' words'):
            word, *coeffs = line.split(' ')
            if len(coeffs) >= 100:
                embeddings[word] = np.asarray(coeffs, dtype=np.float32)
    return embeddings

def em_to_matrix(_em):
    _W = np.stack(_em.values())
    _vocab = {w: ind for ind, w in enumerate(_em.keys())}
    return Embeddding(normalize(_W, axis=1), _vocab)
    
def normalize(_W, axis=None):
    W_norm = np.zeros(_W.shape)
    d = np.sum(_W ** 2, axis=axis) ** (0.5)
    W_norm = (_W.T / d).T
    return W_norm  # each row is a unit vector
    
def cosine_similarity(_W, _vec):
    return np.dot(_W, _vec.T)  # expects everything already being unit


class EmbeddingItem:
    
    def __init__(self, vec, word_history=[]):
        self.vec = vec
        self.history = word_history
        
    def __add__(self, item):
        return EmbeddingItem(normalize(self.vec + item.vec), self.history + item.history)
    
    def __sub__(self, item):
        return EmbeddingItem(normalize(self.vec - item.vec), self.history + item.history)
    
    def __mul__(self, item):
        return EmbeddingItem(normalize(self.vec.T * item.vec), self.history + item.history)
    

class Embeddding:
    
    def __init__(self, _W, _vocab):
        self.W = _W
        self.vocab = _vocab
        self.ivocab = {v: k for k, v in _vocab.items()}
        
    def __getitem__(self, word) -> EmbeddingItem:
        return EmbeddingItem(np.copy(self.W[self.vocab[word], :]), word_history=[word])
    
    def n_closest(self, n, vec: EmbeddingItem):
        distances = cosine_similarity(self.W, vec.vec)
        for word in vec.history:
            distances[self.vocab[word]] = -np.Inf
        return [(self.ivocab[ind], distances[ind]) for ind in np.argsort(-distances)[:n]]
glove = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
fastext = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
em = load_emb(fastext)
emb = em_to_matrix(em)
def show(_emb, _vec):
    print('-'*30)
    for word, dist in _emb.n_closest(3, _vec):
        print('{:<20} {:.2f}'.format(word, dist))

show(emb, emb['berlin'] - emb['germany'] + emb['finland'])
show(emb, emb['mother'] - emb['woman'])
show(emb, emb['king'] - emb['man'])
show(emb, emb['man'] + emb['dress'])
