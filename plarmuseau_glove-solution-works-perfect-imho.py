import pandas as pd

# 1  Import glove 10 rows only , i lack memory and speed guys...

Vt_=pd.read_table("../input/glove.6B.50d.txt",usecols=[0,1,2,3,4,5,6,7,8,9,10,11],header=None, delim_whitespace=1,quoting=3)

Vt_.columns=['word',1,2,3,4,5,6,7,8,9,10,11]



#import test database

test = pd.read_csv("../input/testw.csv")

test['category'] = 'physics'



# 2  Tfidf vectorize with the glove.

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf3_vectorizer = TfidfVectorizer(vocabulary=Vt_['word'])

D_ = tfidf3_vectorizer.fit_transform(test['title']+'. '+test['content'])

# 82000*400k woorden 574000 words classified compressed in sparse matrix 300mb

from scipy.sparse import csr_matrix

D_=csr_matrix(D_)

Vword=Vt_['word']

Vt_=Vt_.drop('word', axis=1)



# 3 Find the most relevant words...

for xi in range (0,10):

    idnr=test.ix[xi].id

    Dtemp=pd.DataFrame(D_[xi,:].todense())

    Dtemp=Dtemp.append(Vword.T)

    Dtemp=Dtemp.append(Vt_.T)

    Dtemp=Dtemp.T

    Dtemp.columns=['rf','word','v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','v11']

    Dtemp.sort_values(by='rf')

    Dbis=Dtemp[Dtemp['rf']>0]

    maxrf=Dbis.max(axis=0).rf/2.2

    Dbis=Dtemp[Dtemp['rf']>maxrf]

    Dbis=Dbis.drop(['rf','word'],axis=1)

    Dcorr=Dbis.dot(Dbis.T)/(abs(Dbis).dot(abs(Dbis.T)))

    lengte=len(Dtemp[Dtemp['rf']>0])

    corrm=Dcorr[0:lengte]

    corrm=corrm.fillna(value=0)

    superwoord=corrm[corrm>0.85].sum()

    prnttmp=''

    for wn in superwoord[superwoord>1].index:

        prnttmp+=' '+Dtemp.ix[wn].word

    print(idnr,prnttmp)