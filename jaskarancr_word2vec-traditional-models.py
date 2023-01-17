# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test_df=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
df.head(5)
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(df.target)
plt.xlabel('class')
plt.title('Count of labels')
import nltk
tk=nltk.tokenize.TreebankWordTokenizer()
tweet_tokens = [tk.tokenize(sent) for sent in df['text']]
test_tweet_tokens=[tk.tokenize(sent) for sent in test_df['text']]
from nltk.corpus import stopwords
for i in range(len(tweet_tokens)):
    tweet_tokens[i] = [w for w in tweet_tokens[i] if w not in stopwords.words('english')]
for i in range(len(test_tweet_tokens)):
    test_tweet_tokens[i] = [w for w in test_tweet_tokens[i] if w not in stopwords.words('english')]
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('/kaggle/input/nlpword2vecembeddingspretrained/GoogleNews-vectors-negative300.bin', binary=True)
model.wv['hello'].shape
model.similar_by_word('fuck', topn=5)
vocabulary = model.wv.vocab
len(vocabulary)
# importing bokeh library for interactive dataviz
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook

# defining the chart
output_notebook()
plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A map of 10000 word vectors",
    tools="pan,wheel_zoom,box_zoom,reset,hover",
    x_axis_type=None, y_axis_type=None, min_border=1)

# getting a list of word vectors. limit to 10000. each is of 300 dimensions
word_vectors = [model[w] for w in list(model.wv.vocab.keys())[:5000]]

# dimensionality reduction. converting the vectors to 2d vectors
from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
tsne_w2v = tsne_model.fit_transform(word_vectors)

# putting everything in a dataframe
tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
tsne_df['words'] = list(model.wv.vocab.keys())[:5000]

# plotting. the corresponding word appears when you hover on the data point.
plot_tfidf.scatter(x='x', y='y', source=tsne_df)
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips={"word": "@words"}
show(plot_tfidf)
def document_vector(doc):
    """Create document vectors by averaging word vectors. Remove out-of-vocabulary words."""
    doc = [word for word in doc if word in model.wv.vocab]
    return np.mean(model[doc], axis=0)

docs=[]
for x in tweet_tokens:
    doc= [word for word in x if word in model.wv.vocab]
    docs.append(doc)
len(docs)
docs_test=[]
for x in test_tweet_tokens:
    doc= [word for word in x if word in model.wv.vocab]
    docs_test.append(doc)
len(docs_test)
list_w2v=[]
for i in range(0,len(docs)):
    if docs[i]==[]:
        list_w2v.append(np.zeros(300,))
    else:
        list_w2v.append(np.mean(model[docs[i]], axis=0))
len(list_w2v)
df.shape
list_w2v_test=[]
for i in range(0,len(docs_test)):
    if docs_test[i]==[]:
        list_w2v_test.append(np.zeros(300,))
    else:
        list_w2v_test.append(np.mean(model[docs_test[i]], axis=0))
len(docs_test)
len(list_w2v_test)
label_ls=list(df['target'])
len(label_ls)
test_df.shape
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
clf1 = LogisticRegression(C=1, max_iter=1000, solver='lbfgs')
scores = cross_val_score(clf1, list_w2v,label_ls, cv=5,scoring='f1')
scores
clf1.fit(list_w2v,label_ls)

sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = clf1.predict(list_w2v_test)
sample_submission.to_csv("submission.csv", index=False)
