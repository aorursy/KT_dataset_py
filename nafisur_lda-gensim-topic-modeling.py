import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import pickle
import gensim
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
lemma=WordNetLemmatizer()
# Load the list of documents
with open('../input/newsgroups', 'rb') as f:
    newsgroup_data = pickle.load(f)


len(newsgroup_data)
# with open('filename.pickle', 'wb') as handle:
#     pickle.dump(newsgroup_data, handle, protocol=2)
# with open(newsgroup_data, 'wb') as handle:
#     pickle.dump(your_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
# pickle.dump(newsgroup_data, protocol=2)
newsgroup_data[1]
import re
newsgroup_data1=[]
for i in range(len(newsgroup_data)):
    t=newsgroup_data[i]
    t=re.sub('[^a-zA-Z]',' ',t)
    t=re.sub('(\\b\\w\\b)',' ',t)
    t=re.sub('(\s+)',' ',t)
    t=str(t).lower()
    t=word_tokenize(t)
    t=[lemma.lemmatize(w) for w in t ]
    t=' '.join(t)
    newsgroup_data1.append(t)

newsgroup_data1[1]
vect = CountVectorizer(min_df=30, max_df=0.2, stop_words='english')
X = vect.fit_transform(newsgroup_data1)
X.shape
vect.get_feature_names()[0:10]
dtm = vect.transform(newsgroup_data1)
repr(dtm)
import pandas as pd
pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names()).head()
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
corpus
id_map = dict((v, k) for k, v in vect.vocabulary_.items())
ldamodel = gensim.models.ldamodel.LdaModel(corpus,num_topics=10,id2word=id_map,random_state=34,passes=25)
print(ldamodel)
ldamodel.print_topics(num_topics=10, num_words=20)
def topic_names():
    
    # Your Code Here
    
    #return ldamodel.print_topics()
    return ["Education", "Religion", "Science","Sports","Banks","Automobiles", "Computers & IT",  "Automobiles", "Computers & IT"]

topic_names()
new_doc=["It's my understanding that the freezing will start to occur because of the growing distance of Pluto and Charon from the Sun, due to it's elliptical orbit. It is not due to shadowing effects. Pluto can shadow Charon, and vice-versa."]
def topic_distribution():
    
    X_2 = vect.transform(new_doc)
    corpus_2 = gensim.matutils.Sparse2Corpus(X_2, documents_columns=False)
    #ldamodel_2 = gensim.models.ldamodel.LdaModel(corpus_2, num_topics=10, random_state=34, passes=25, id2word=id_map)
    
    return list(ldamodel.get_document_topics(corpus_2))[0]

topic_distribution()
