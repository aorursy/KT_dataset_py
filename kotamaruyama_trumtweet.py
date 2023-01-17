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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import  Counter
plt.style.use('ggplot')
stop=set(stopwords.words('english'))
import re
from nltk.tokenize import word_tokenize
import gensim
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
df= pd.read_csv('../input/trumpalltweetcsv/trumpalltweet.csv')
df.head()
df['tweet']
example="New competition launched :https://www.kaggle.com/c/nlp-getting-started"
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

remove_URL(example)
df=df.apply(lambda x : remove_URL(x))
df.head()
example = """<div>
<h1>Real or Fake</h1>
<p>Kaggle </p>
<a href="https://www.kaggle.com/c/nlp-getting-started">getting started</a>
</div>"""
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
print(remove_html(example))
df=df.apply(lambda x : remove_html(x))
def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

example="I am a #king"
print(remove_punct(example))
df=df.apply(lambda x : remove_punct(x))
df.head()
df
df.head()
fig,(ax1)=plt.subplots(1,figsize=(10,15))
tweet_len=df.str.split().map(lambda x: len(x))
ax1.hist(tweet_len,color='red')
ax1.set_title('tweets')
plt.show()
fig,(ax1)=plt.subplots(1,figsize=(10,5))
word=df.str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='red')
ax1.set_title('average word in twitter')
stop_words = set(stopwords.words("english"))
print(stop_words)
stop_words = stopwords.words('english')
stop_words.append('realDonaldTrump')

print(stop_words)
stop_words = stopwords.words('english')
stop_words.append('realdonaldtrump')
stop_words.append('trump')
stop_words1=stop_words
def create_corpus(df):
    corpus=[]
    
    for x in df.str.split():
        for i in x:
            corpus.append(i)
    return corpus
corpus=create_corpus(df)
counter=Counter(corpus)
most=counter.most_common()
X=[]
Y=[]
for word,count in most[:100]:
    if (word not in stop_words1) :
        X.append(word)
        Y.append(count)
sns.barplot(x=Y,y=X,linewidth=9.5)