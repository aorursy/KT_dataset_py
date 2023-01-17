# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
don = pd.read_csv("../input/Donations.csv")
# Any results you write to the current directory are saved as output.
#Find Donors who gave donations to multiple projects 
import base64
import numpy as np
import pandas as pd
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from collections import Counter
from scipy.misc import imread
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from matplotlib import pyplot as plt
%matplotlib inline

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

projects = pd.read_csv("../input/Projects.csv")

ds = projects.loc[projects['Project Current Status'].isin(['Fully Funded','Expired','Live'])]
df3 = ds.merge(don, on = "Project ID" )
df3['Donor ID'] = df3['Donor ID'].str.strip()

#consider only donations made to different projects, as we are interested in what makes the donor open his wallet

df4 = df3.drop_duplicates(subset=['Project ID', 'Donor ID'])
df4['Donor ID'].value_counts()
# vectorizer takes care of stopwords, but we seek lemmatisation as well... hence this class to override the function inside the vectoriser

from nltk.stem import WordNetLemmatizer
lemm = WordNetLemmatizer()
class LemmaCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: (lemm.lemmatize(w) for w in analyzer(doc))


# create a sparse matrix for lda 
df4_sample = df4.sample(n = 100000, random_state = 69)
text = list(df4_sample["Project Essay"])
# Calling our overwritten Count vectorizer
tf_vectorizer = LemmaCountVectorizer(max_df=0.65, 
                                     min_df=2,
                                     stop_words='english',
                                     decode_error='ignore')
tf = tf_vectorizer.fit_transform(text)
feature_names = tf_vectorizer.get_feature_names()
count_vec = np.asarray(tf.sum(axis=0)).ravel()
zipped = list(zip(feature_names, count_vec))
x, y = (list(x) for x in zip(*sorted(zipped, key=lambda x: x[1], reverse=True)))
# Now I want to extract out on the top 15 and bottom 15 words
Y = np.concatenate([y[0:15], y[-16:-1]])
X = np.concatenate([x[0:15], x[-16:-1]])

# Plotting the Plot.ly plot for the Top 50 word frequencies
data = [go.Bar(
            x = x[0:50],
            y = y[0:50],
            marker= dict(colorscale='Jet',
                         color = y[0:50]
                        ),
            text='Word counts'
    )]

layout = go.Layout(
    title='Top 50 Word frequencies after Preprocessing'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='basic-bar')

# Plotting the Plot.ly plot for the Top 50 word frequencies
data = [go.Bar(
            x = x[-100:],
            y = y[-100:],
            marker= dict(colorscale='Portland',
                         color = y[-100:]
                        ),
            text='Word counts'
    )]

layout = go.Layout(
    title='Bottom 100 Word frequencies after Preprocessing'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='basic-bar')
lda = LatentDirichletAllocation(n_components=63, max_iter=5,
                                learning_method = 'online',
                                learning_offset = 50.,
                                random_state = 0)
lda.fit(tf)
text[890]
tf1 = tf_vectorizer.transform([text[890]])
doc_topic = lda.transform(tf1)
doc_topic
import numpy   
topic_high = numpy.where(doc_topic > 0.05)
numpy.where(doc_topic > 0.05)

topic_high = list(topic_high)[1]
top_tup = tuple(map(lambda x:(x,doc_topic[0,x]),topic_high))
sor_ry = sorted(top_tup, key=lambda tup: tup[1], reverse = True)

from wordcloud import WordCloud, STOPWORDS

tf_feature_names = tf_vectorizer.get_feature_names()

first_topic = lda.components_[sor_ry[0][0]]
first_topic_words = [tf_feature_names[i] for i in first_topic.argsort()[:-50 - 1 :-1]]

firstcloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          width=2500,
                          height=1800
                         ).generate(" ".join(first_topic_words))
plt.imshow(firstcloud)
plt.axis('off')
plt.show()
sixtieth_topic = lda.components_[sor_ry[1][0]]
sixtieth_topic_words = [tf_feature_names[i] for i in sixtieth_topic.argsort()[:-50 - 1 :-1]]

scloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          width=2500,
                          height=1800
                         ).generate(" ".join(sixtieth_topic_words))
plt.imshow(scloud)
plt.axis('off')
plt.show()
supp_essays = df4[df4['Donor ID'] == df4_sample.iloc[890]['Donor ID']]
supp_essays
text_test = list(supp_essays['Project Essay'])

tf1 = tf_vectorizer.transform([text_test[1]])

doc_topic = lda.transform(tf1)
print(doc_topic)
import numpy   
numpy.where(doc_topic > 0.05)
text_test = list(supp_essays['Project Essay'])

tf1 = tf_vectorizer.transform([text_test[2]])

doc_topic = lda.transform(tf1)
print(doc_topic)
import numpy   
numpy.where(doc_topic > 0.05)
type(doc_topic)
supp_essays['Project Essay'].loc[supp_essays['Donation Amount'].idxmax()]
# To test our hypothesis, we are going to break down the essays into text topics, and then see if there is a relation between the amount donated and the topics relevance

tf1 = tf_vectorizer.transform(text_test)
doc_topic = lda.transform(tf1)
supp_cp = supp_essays
d_copy = doc_topic
df = pd.DataFrame(d_copy)
supp_cp = supp_cp.reset_index(drop=True)
dataset = pd.concat([supp_cp['Donation Amount'],df],axis = 1)


dataset
topics = dataset.drop(['Donation Amount'], axis = 1)
topics.mean().plot(kind='bar', figsize=(20,10))