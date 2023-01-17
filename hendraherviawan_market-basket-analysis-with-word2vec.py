# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import gensim
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from IPython.display import display, HTML
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.

#__________________
# read the datafile
df_initial = pd.read_csv('../input/data.csv',encoding="ISO-8859-1",
                         dtype={'CustomerID': str,'InvoiceNo': str, 'StockCode': str})
df_initial['InvoiceDate'] = pd.to_datetime(df_initial['InvoiceDate'])
#df_initial['StockCode'] = df_initial['StockCode'].astype(str)
print('Dataframe dimensions:', df_initial.shape)
 
# show first lines
display(df_initial[:5])
## Drop transaction that have negative price or price

#df_initial.dropna(axis = 0, subset = ['CustomerID'], inplace = True)
df_initial.drop((df_initial[(df_initial['Quantity'] <= 0) | (df_initial['UnitPrice'] < 0)]).index, inplace=True)
print('Dataframe dimensions:', df_initial.shape)
## Delete duplicate transaction

print('Entris dupliquées: {}'.format(df_initial.duplicated().sum()))
df_initial.drop_duplicates(inplace = True)
print('Dataframe dimensions:', df_initial.shape)
##

products = df_initial[['StockCode', 'Description']].drop_duplicates()
products.head()
##

stockcode = df_initial.groupby("InvoiceNo").apply(lambda order: order['StockCode'].tolist())
stockcode[0:5]
## 
model = gensim.models.Word2Vec(stockcode.values, size=5, window=6, min_count=2, workers=4)
vocab = list(model.wv.vocab.keys())

pca = PCA(n_components=2)
pca.fit(model.wv.vectors)
## 

def get_batch(vocab, model, n_batches=3):
    output = list()
    for i in range(0, n_batches):
        rand_int = np.random.randint(len(vocab), size=1)[0]
        suggestions = model.wv.most_similar(positive=[vocab[rand_int]], topn=5)
        suggest = list()
        for i in suggestions:
            suggest.append(i[0])
        output += suggest
        output.append(vocab[rand_int])
    return output

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    """From Tensorflow's tutorial."""
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(20, 16))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    #plt.savefig(filename)
    plt.show()
embeds = []
labels = []

for item in get_batch(vocab, model, n_batches=3):
    embeds.append(model[item])
    labels.append(products.loc[products.StockCode == item]['Description'].values)
embeds = np.array(embeds)
embeds = pca.fit_transform(embeds)
plot_with_labels(embeds, labels)
