import pandas as pd

import numpy as np

import gensim

from matplotlib import pyplot as plt

from sklearn.decomposition import PCA



%matplotlib inline
train_orders = pd.read_csv("../input/order_products__train.csv")

prior_orders = pd.read_csv("../input/order_products__prior.csv")

products = pd.read_csv("../input/products.csv").set_index('product_id')
train_orders["product_id"] = train_orders["product_id"].astype(str)

prior_orders["product_id"] = prior_orders["product_id"].astype(str)
train_products = train_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())

prior_products = prior_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())
sentences = prior_products.append(train_products)

longest = np.max(sentences.apply(len))

sentences = sentences.values
model = gensim.models.Word2Vec(sentences, size=100, window=longest, min_count=2, workers=4)
vocab = list(model.wv.vocab.keys())
pca = PCA(n_components=2)

pca.fit(model.wv.syn0)
def get_batch(vocab, model, n_batches=3):

    output = list()

    for i in range(0, n_batches):

        rand_int = np.random.randint(len(vocab), size=1)[0]

        suggestions = model.most_similar(positive=[vocab[rand_int]], topn=5)

        suggest = list()

        for i in suggestions:

            suggest.append(i[0])

        output += suggest

        output.append(vocab[rand_int])

    return output



def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):

    """From Tensorflow's tutorial."""

    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"

    plt.figure(figsize=(18, 18))  #in inches

    for i, label in enumerate(labels):

        x, y = low_dim_embs[i,:]

        plt.scatter(x, y)

        plt.annotate(label,

                     xy=(x, y),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

#     plt.savefig(filename)

    plt.show()
embeds = []

labels = []

for item in get_batch(vocab, model, n_batches=3):

    embeds.append(model[item])

    labels.append(products.loc[int(item)]['product_name'])

embeds = np.array(embeds)

embeds = pca.fit_transform(embeds)

plot_with_labels(embeds, labels)

model.save("product2vec.model")