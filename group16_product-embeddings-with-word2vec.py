!pip install adjustText
import pandas as pd

import numpy as np

from gensim.models.word2vec import Word2Vec

from sklearn.manifold import TSNE

from adjustText import adjust_text



%matplotlib inline

import matplotlib.pyplot as plt
np.random.seed(42)
data = pd.read_csv('../input/groceries/groceries - groceries.csv', header=0)

data.head(5)
sentences = []

for i, row in data.iterrows():

    vals = row.values[1:].astype(str)

    

    # Remove the nans

    vals = vals[vals != 'nan']

    

    # Order does not really matter in shopping baskets (unlike English sentences)

    # so this is a form of augmentation

    for _ in range(min(3, len(vals))):

        np.random.shuffle(vals)

        sentences.append(list(vals))

        

print('\n'.join([', '.join(x) for x in sentences[:10]]))
print('Embedding {} sentences...'.format(len(sentences)))

    

model = Word2Vec(

    sentences,

    size=10,

    window=3,

    workers=1,

    sg=0,

    iter=25,

    negative=25,

    min_count=1,

    seed=42,

    compute_loss=True

)



print(model.get_latest_training_loss())
products = list(model.wv.vocab.keys())

embeddings = []

for product in products:

    embeddings.append(model.wv[product])

embeddings = np.array(embeddings)

print(len(products), embeddings.shape)



tsne = TSNE(random_state=42)

X_tsne = tsne.fit_transform(embeddings)



plt.figure(figsize=(10, 10))

plt.scatter(X_tsne[:, 0], X_tsne[:, 1])



texts = []

for x, y, lab in zip(X_tsne[:, 0], X_tsne[:, 1], products):

    text = plt.text(x, y, lab)

    texts.append(text)

    

adjust_text(texts, lim=5, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

plt.show()