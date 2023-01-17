

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import pandas as pd

import numpy as np

import tensorflow_hub as hub

from sklearn import metrics

import lightgbm as lgb

from lightgbm import LGBMClassifier



target = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')['target']

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

ssub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

import pandas_profiling as pp
pp.ProfileReport(train)
import re



#extracting hashtags using simple regex

train['hashtags']=train['text'].apply(lambda x:re.findall('#\w*',x))
train.head(5)
import matplotlib.pyplot as plt

from wordcloud import WordCloud

labels=['Negative','Positive']

no_clusters=2

for c in range(2):

    print('Target:-',labels[c])

    hts=list(train[train['target']==c]['hashtags'])



    hashes=[]

    for ht in  hts:

        for h in ht:

            hashes.append(h.strip())



    string_hash=' '.join(hashes)



    hash_values=pd.Series(hashes).value_counts()



    hval=hash_values.reset_index()



    #wordcloud plot

    d = {}

    for a, x in hval.values:

        d[a] = x



    wordcloud = WordCloud(max_font_size=40)

    wordcloud.generate_from_frequencies(frequencies=d)

    plt.figure(figsize=(70,70))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.show()


embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/3")

X_train_embeddings = embed(train.text.values)

X_test_embeddings = embed(test.text.values)
from sklearn.manifold import TSNE



#2-D dimensional representation

X_embedded = TSNE(n_components=2,perplexity=50).fit_transform(X_train_embeddings['outputs'])

xy_df=pd.DataFrame(X_embedded)

xy_df['tweets']=train.text.values

xy_df['Target']=target

xy_df.columns=['x', 'y', 'tweets','Target']
from bokeh.models import ColumnDataSource

from bokeh.models import HoverTool

from bokeh.io import output_file

from bokeh.io import output_notebook, show

from bokeh.plotting import figure



output_notebook()
colormap = {1: 'red',0: 'blue'}

colors = [colormap[x] for x in xy_df['Target']]

xy_df['colors'] = colors



src = ColumnDataSource(xy_df)





p = figure(plot_height=650, title="TSNE Tweet Embedding Viz ")

p.circle(x='x', y='y',source=src,legend='Target',color='colors')

p.xgrid.grid_line_color = None

p.y_range.start = 0

p.xaxis.axis_label = "x"

p.yaxis.axis_label = "y"

p.xaxis.major_label_orientation = 1



from bokeh.models import CustomJS



callback = CustomJS(code="""

    var tooltips = document.getElementsByClassName("bk-tooltip");

    for (var i = 0, len = tooltips.length; i < len; i ++) {

        tooltips[i].style.top = ""; // unset what bokeh.js sets

        tooltips[i].style.left = "";

        tooltips[i].style.bottom = "0px";

        tooltips[i].style.left = "0px";

    }

    """)

hover = HoverTool(callback=callback,tooltips = [('Tweet', '@tweets'),('Target','@Target')])



p.add_tools(hover)

show(p)
from sklearn.feature_extraction.text import TfidfVectorizer



#using top 50 words as the features

vectorizer = TfidfVectorizer(max_features=30,stop_words='english')

all_tweets=list(train['text'])+list(test['text'])

X = vectorizer.fit_transform(all_tweets)



tweet_array=X.toarray()

tf_train=tweet_array[0:len(train)]

tf_test=tweet_array[len(train):len(tweet_array)]



## Merging TF-IDF and Universal Sentence Encoder

train_df=np.concatenate([X_train_embeddings['outputs'],tf_train],axis=1)

test_df=np.concatenate([X_test_embeddings['outputs'],tf_test],axis=1)
import lightgbm as lgb

text_clf = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05)



text_clf.fit(train_df, target)
pred=text_clf.predict(test_df)



ssub["target"] = pred

ssub.to_csv("submission.csv",index=False)