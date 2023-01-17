import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

direct='3c-shared-task-influence'



train=pd.read_csv('../input/'+direct+'/train.csv')

train
test=pd.read_csv('../input/'+direct+'/test.csv')

test
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfTransformer



#tfidf

#transformer = TfidfTransformer(smooth_idf=False)

cv = CountVectorizer(ngram_range=(1, 1))

citing_tf=cv.fit_transform(train['citing_title'].append(test['citing_title']))



cited_tf=cv.transform(train['cited_title'].append(test['cited_title']))

conte_tf=cv.transform(train['citation_context'].append(test['citation_context']))



cit_words=cv.get_feature_names()



np.dot(citing_tf.T,cited_tf) 

print(citing_tf.shape[1])

citcont_dot= np.linalg.inv( np.dot(cited_tf.T,cited_tf).todense()+np.random.rand(1439,1439)/1000 )

from scipy.sparse.linalg import svds, eigs

U, s, V = svds(cited_tf.astype('float'), k=400 )

Xi=V.T.dot(np.diag(s).dot(U.T))

# doing SVD regression

Xi.shape,Xi[:,:3000].dot(train['citation_influence_label']+1)



pd.DataFrame( np.sqrt(eigs(cited_tf.astype('float').dot(cited_tf.astype('float').T), k=500)[0]).real ).plot()
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(max_iter=2000)

logreg.fit(Xi.T[:3000], train['citation_influence_label'].values)

predictions = logreg.predict(Xi.T[3000:])

pred=pd.DataFrame(predictions,columns=['citation_influence_label'])

pred['unique_id']=test['unique_id']

pred.groupby('citation_influence_label').count()

pred.to_csv('submission.csv', index=False)