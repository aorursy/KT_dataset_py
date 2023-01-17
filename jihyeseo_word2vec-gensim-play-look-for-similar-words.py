import gensim
from nltk.corpus import brown
model = gensim.models.Word2Vec(brown.sents())
model.save('brown.embedding')
new_model = gensim.models.Word2Vec.load('brown.embedding')
len(new_model['university'])
new_model.similarity('university','school') > 0.3
from nltk.data import find
word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
len(model.vocab)
len(model['university'])
model.most_similar(positive=['university'], topn = 3)
model.most_similar(positive=['chest'], topn = 10)
model.most_similar(positive=['waist'], topn = 10)
model.most_similar(positive=['hip'], topn = 10)
model.doesnt_match('breakfast cereal dinner lunch'.split())
model.doesnt_match('table chair dish napkin'.split())
model.doesnt_match('rice cereal milk spoon fork'.split())
model.doesnt_match('grape banana orange pineapple pine apple'.split())
model.most_similar(positive=['woman','king'], negative=['man'], topn = 10)
model.most_similar(positive=['Paris','Germany'], negative=['Berlin'], topn = 1)
model.most_similar(positive=['Seoul','Germany', 'China'], negative=['Berlin', 'Japan'], topn = 10)
model.most_similar(positive=['Seoul','Germany'], negative=['Berlin'], topn = 10)
# model.most_similar(positive=['Seoul','North Korea'], negative=['South Korea'], topn = 1)
# KeyError: "word 'North Korea' not in vocabulary"
import numpy as np
labels = []
count = 0
max_count = 50
X = np.zeros(shape=(max_count,len(model['university'])))

for term in model.vocab:
    X[count] = model[term]
    labels.append(term)
    count+= 1
    if count >= max_count: break

# It is recommended to use PCA first to reduce to ~50 dimensions
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
X_50 = pca.fit_transform(X)

# Using TSNE to further reduce to 2 dimensions
from sklearn.manifold import TSNE
model_tsne = TSNE(n_components=2, random_state=0)
Y = model_tsne.fit_transform(X_50)

# Show the scatter plot
import matplotlib.pyplot as plt
plt.scatter(Y[:,0], Y[:,1], 20)

# Add labels
for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy = (x,y), xytext = (0, 0), textcoords = 'offset points', size = 10)

plt.show()
import gensim
from gensim.models.word2vec import Word2Vec
# Load the binary model
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary = True);

# Only output word that appear in the Brown corpus
from nltk.corpus import brown
words = set(brown.words())
print (len(words))

# Output presented word to a temporary file
out_file = 'pruned.word2vec.txt'
f = open(out_file,'wb')

word_presented = words.intersection(model.vocab.keys())
f.write('{} {}\n'.format(len(word_presented),len(model['word'])))

for word in word_presented:
    f.write('{} {}\n'.format(word, ' '.join(str(value) for value in model[word])))

f.close()
