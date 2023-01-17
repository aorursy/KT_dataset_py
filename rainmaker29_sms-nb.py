# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import nltk
sms = pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv",encoding='latin')
sms.head()
sms.columns[2]
# Most of the 2,3,4 columns have null values
print(sms.iloc[:,2].isna().sum(),
      sms.iloc[:,3].isna().sum(),
      sms.iloc[:,4].isna().sum())
sms = sms.drop([sms.columns[2],sms.columns[3],sms.columns[4]],axis=1)
sms.head()
sms.v1.value_counts()
sns.countplot(sms["v1"])
sms.describe()
sms.groupby('v1').describe().T
sms['length'] = sms['v2'].apply(len)
sms.head()
sns.set()
sms['length'].plot(bins=50,kind="hist")
sms.length.describe()
sms.hist(column='length',by='v1',bins=50,figsize=(12,3))
# The spam messages are longer
import string

test = "We can turn, the world to go !"

puncless = [c for c in test if c not in string.punctuation]

puncless = "".join(puncless)
puncless
from nltk.corpus import stopwords
stopwords.words('english')[:20]
list(puncless.split())
stopless = [w for w in list(puncless.split()) if w not in stopwords.words('english')]
stopless
from nltk.corpus import stopwords
def text_process(msg):
    puncless = [c for c in msg if c not in string.punctuation]
    
    puncless = "".join(puncless)
    
    return [w for w in list(puncless.split()) if w.lower() not in stopwords.words('english')]
sms['v2'].head(5).apply(text_process)

#Vectorization
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer = text_process).fit(sms['v2'])
print(len(bow_transformer.vocabulary_))
sms_bow = bow_transformer.transform(sms['v2'])
print("Sparse shape",sms_bow.shape)
print("Non zero",sms_bow.nnz)
sparsity = (100.0 * sms_bow.nnz / (sms_bow.shape[0] * sms_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity,4)))
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(sms_bow)
sms_tfidf = tfidf_transformer.transform(sms_bow)
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(sms_tfidf,sms['v1'])
#Testing
print("Predicted:",model.predict(sms_tfidf)[0])
print("expected:",sms.v1[3])
pred = model.predict(sms_tfidf)
from sklearn.metrics import classification_report
print(classification_report(sms['v1'],pred))
from sklearn.metrics import accuracy_score
print(accuracy_score(pred,sms['v1']))
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
model2 = neigh.fit(sms_tfidf,sms['v1'])
pred2 = model2.predict(sms_tfidf)

print("KNeighbors Classifier accuracy : ",accuracy_score(pred2,sms['v1']))
from sklearn.svm import LinearSVC
model3 = LinearSVC(random_state=0).fit(sms_tfidf,sms['v1'])
pred3 = model3.predict(sms_tfidf)
print("SVC accuracy : ",accuracy_score(pred3,sms['v1']))

from sklearn.model_selection import train_test_split
