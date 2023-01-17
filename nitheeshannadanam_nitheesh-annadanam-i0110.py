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
import pandas as pd;
import numpy as np;
import scipy as sp;
import sklearn;
import sys;
from nltk.corpus import stopwords;
import nltk;


from gensim.models import ldamodel
from gensim import matutils, models
import gensim.corpora;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
import pickle;
from gensim.models import CoherenceModel

data_text=pd.read_csv("/kaggle/input/unstructured-l0-nlp-hackathon/data.csv")
data_text.columns=['Id','fullReview']
import re
data_text['fullReview']=[re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([0-9]+)","",str(x)) for x in data_text['fullReview']]
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

st  = stopwords.words('english')
dfc=data_text
dfc['clean_txt'] = dfc['fullReview'].apply(lambda x: ' '.join([word for word in x.split() if word not in (st) and len(word)>0 ]))




#nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize , RegexpTokenizer
dfc['pos']=[nltk.pos_tag(word_tokenize(text)) for text in dfc['clean_txt']]
def get_all(posl):
    return([x[0] for x in posl if  x[1]=='NN' 
           or x[1]=='NNS' or x[1]=='NNP' or x[1]=='NNPS' or x[1].startswith('JJ')])
        
    
dfc['clean_all']=dfc['pos'].apply(get_all)
def lem_app(adjl):
    if len(adjl)>0:
        return([lemmatizer.lemmatize(x) for x in adjl])
dfc['clean_all_lem']=dfc['clean_all'].apply(lem_app)
data_text=dfc
train_headlines = [value[5] for value in data_text.iloc[0:].values]
num_topics = 5;
id2word = gensim.corpora.Dictionary(train_headlines)
corpus = [id2word.doc2bow(text) for text in train_headlines]
import time 
start = time.time()
lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics,random_state=1,passes=50,eta = 1)
print ('time taken to run the model ' , time.time()-start)
def get_lda_topics(model, num_topics):
    word_dict = {}
    topics = model.show_topics(num_topics,10)
    word_dict = {'Topic '+str(i):[x.split('*') for x in words.split('+')] \
                 for i,words in lda.show_topics(num_topics,10)}
    return pd.DataFrame.from_dict(word_dict)
get_lda_topics(lda, num_topics)
cg=lda[corpus]
cgl=list(cg)
cgl=[max(y,key=lambda x:x[1]) for y in cgl]
final_list= list(zip([x[0] for x in cgl],dfc['Id'] ))
dfc['idn']=[x[1] for x in final_list]
dfc['topic_num']=[x[0] for x in final_list]
dfc.loc[dfc.topic_num.isin([4]),'Topic']='Automobiles'
dfc.loc[dfc.topic_num.isin( [3]),'Topic']="glassdoor_reviews"
dfc.loc[dfc.topic_num.isin( [2]),'Topic']="tech_news"
dfc.loc[dfc.topic_num.isin( [1]),'Topic']="room_rentals"
dfc.loc[dfc.topic_num.isin( [0]),'Topic']="sports_news"
dfcfinal=dfc[['Id','Topic']]
dfcfinal.to_csv('submission.csv')

dfcfinal.to_csv('/kaggle/working/submit.csv')