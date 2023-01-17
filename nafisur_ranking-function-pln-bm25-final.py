import os
print(os.listdir("../input/docs/docs"))

import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize


def read_doc(doc_dir_name):
    doc_dir=os.getcwd()+'/'+doc_dir_name+'/'
    doc_names=os.listdir(doc_dir)
    doc_length=len(doc_names)
    all_docs=[]
    for i in range(doc_length):
        path=doc_dir+doc_names[i]
        with open(str(path),'r') as f:
            content=f.read()
        all_docs.append(content)
    
    df=pd.DataFrame({'Doc_name':doc_names,'raw_doc':all_docs})
    return df

def preprocessing(df):
    clean_corpus=[]
    for i in range(0,len(df)):
        text=df[i]
        #text=BeautifulSoup(review,'lxml').text
        text=re.sub('[^a-zA-Z]',' ',text)
        text=str(text).lower()
        text=word_tokenize(text)
        #text=[stemmer.stem(w) for w in text if w not in Cstopwords]
        #review=[lemma.lemmatize(w) for w in text ]
        text=' '.join(text)
        clean_corpus.append(text)
    return clean_corpus



doc=read_doc("../input/docs/docs")
doc['processed_text']=preprocessing(doc['raw_doc'])
doc.head()
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
cv.fit(doc['processed_text'])
bow=cv.transform(doc['processed_text'])
bow_df=pd.DataFrame(bow.toarray(), columns=cv.get_feature_names())
bow_df.head()
c=bow.toarray()
c
query = 'News about 234 Presidential campaign'
q=preprocessing([query])
a=cv.transform(q).toarray()
a
def ranking_pln(a,c,b=0.25):
    m=len(c)
    avdl=sum(sum(c))/float(m)
    dfw=sum(c.astype(bool))
    idf=np.log((m+1.0)/dfw)
    num=np.log(1.0+np.log(1.0+c))
    dem=1-b+(b*(sum(c.T)/avdl))
    dem=np.matrix(dem)
    pln_term=np.array(num/dem.T)
    pln=pln_term*idf
    score=np.dot(a,pln.T)
    return score
ranking_pln(a,c,b=0.25)
score=list(ranking_pln(a,c,b=0.25)[0])
df=doc
df['PLN_score']=score
df.head()
ranking=df.sort_values('PLN_score',ascending=False)
ranking
def ranking_okapi_bm25(a, c, b=0.25, k=3):
    m=len(c)
    avdl=sum(sum(c))/float(m)
    dfw=sum(c.astype(bool))
    idf=np.log((m+1.0)/dfw)
    num=(k+1.0)*c
    dem=k*(1.0-b+(b*(sum(c.T)/avdl)))
    dem=np.matrix(dem)
    dem=c+dem.T
    okapi=np.array(num/dem)*idf
    score=np.dot(a,okapi.T)
    return score
ranking_okapi_bm25(a, c, b=0.25, k=3)
okapi_score=list(ranking_okapi_bm25(a, c, b=0.25, k=3)[0])
df['Okapi_score']=okapi_score
df.head()
ranking=df.sort_values('Okapi_score',ascending=False)
ranking
