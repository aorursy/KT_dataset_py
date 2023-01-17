import numpy as np 



import os

print(os.listdir("../input"))



from sklearn.metrics.pairwise import pairwise_distances
print('Loading word vectors...')

word2vec = {}

embedding = []

idx2word = []

with open('../input/glove.6B.100d.txt') as f:

    for row in f:

        value = row.split(" ")

        word = value[0]

        vec = np.asarray(value[1:], dtype = 'float32')

        word2vec[word] = vec

        embedding.append(vec)

        idx2word.append(word)



print('Found %s word vectors.' % len(word2vec))

embedding = np.array(embedding)

V,D = embedding.shape

print(embedding.shape)
def analogy(x1, x2, y1):

    for w in (x1, x2, y1):

        if w not in word2vec:

            print("%s not in dictionary" % w)

            return

    x1vec = word2vec[x1]

    x2vec = word2vec[x2]

    y1vec = word2vec[y1]

    Vo = x2vec - x1vec + y1vec

    

    distance = pairwise_distances(Vo.reshape(1,D), embedding, metric = 'l2').reshape(V)

    ids = distance.argsort()[:4]

    words = [idx2word[idm] for idm in ids]

     

    best = [word for word in words if word not in (x1, x2, y1)]

    print('best match word ', best)

    print('so,\n',x1, "-", x2, "=", y1, "-", best[0])
analogy( 'man','king', 'women')
analogy( 'paris','france', 'london')
analogy( 'paris','france', 'rome')
analogy('woman', 'man',  'mother')
analogy('woman', 'man', 'sister')
from matplotlib import pyplot as plt

%matplotlib inline

from sklearn.decomposition import PCA





from gensim.test.utils import get_tmpfile, datapath

from gensim.models import KeyedVectors

from gensim.scripts.glove2word2vec import glove2word2vec
datafile = '../input/glove.6B.100d.txt'

globe_file = get_tmpfile('glove.6B.100d.word2vec.txt')

glove2word2vec(datafile, globe_file)
model = KeyedVectors.load_word2vec_format(globe_file)
model.most_similar('mechanical')
model.most_similar('sex')
def analogy(x1, x2, y1):

    return model.most_similar(positive = [y1, x2], negative = [x1])[0][0]
analogy('man', 'king', 'woman')
analogy( 'paris','france', 'london')
analogy( 'paris','france', 'rome')
analogy('woman', 'man',  'mother')
analogy('woman', 'man', 'sister')
analogy('good', 'fantastic', 'bad')
analogy('tall', 'tallest', 'long')
def display_pca_scatterplot(model, words):

        

    word_vectors = np.array([model[w] for w in words])



    twodim = PCA().fit_transform(word_vectors)[:,:2]

    

    plt.figure(figsize=(6,6))

    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')

    for word, (x,y) in zip(words, twodim):

        plt.text(x+0.05, y+0.05, word)
display_pca_scatterplot(model, 

                        ['coffee', 'tea', 'beer', 'wine', 'brandy', 'rum', 'champagne', 'water',

                         'spaghetti', 'borscht', 'hamburger', 'pizza', 'falafel', 'sushi', 'meatballs',

                         'dog', 'horse', 'cat', 'monkey', 'parrot', 'koala', 'lizard',

                         'frog', 'toad', 'monkey', 'ape', 'kangaroo', 'wombat', 'wolf',

                         'france', 'germany', 'hungary', 'luxembourg', 'australia', 'fiji', 'china',

                         'homework', 'assignment', 'problem', 'exam', 'test', 'class',

                         'school', 'college', 'university', 'institute'])