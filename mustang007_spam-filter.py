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
dataset= pd.read_csv('/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv')
dataset.head()
dataset.info()
import matplotlib.pyplot as plt
%matplotlib inline
dataset.groupby(['Category']).count().plot.bar()
dataset.describe()
dataset.drop_duplicates(inplace=True)
import string
import nltk
from nltk.corpus import stopwords
stopword = stopwords.words('english')
def text_preprocessing(texts):
    tex=texts.strip()
    texts_word=[word for word in tex.split() if "@" not in word]
    tex=" ".join(texts_word)
    texts_word=[word for word in tex.split() if "#" not in word]
    tex=" ".join(texts_word)
    texts_word=[word for word in tex.split() if "www." not in word]
    tex=" ".join(texts_word)
    texts_word=[word for word in tex.split() if "http" not in word]
    tex=" ".join(texts_word)
    
    texts_word=[word for word in tex if word not in string.punctuation]
    tex="".join(texts_word)
    texts_word=[word for word in tex.split() if word not in stopword]
    tex=" ".join(texts_word)
    texts_word=[word for word in tex.split() if word.isalpha()]
    tex=" ".join(texts_word)
    texts_word=[word.lower() for word in tex.split()]
    tex=" ".join(texts_word)
    tex=tex.strip()
    return tex.split()
    
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
dataset['Message'] = dataset['Message'].astype('str')
cv=CountVectorizer(analyzer=text_preprocessing).fit(dataset.Message)
cv_trans=cv.transform(dataset.Message)
cv_trans
tfidf=TfidfTransformer().fit(cv_trans)
tfidf_trans=tfidf.transform(cv_trans).toarray()
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
X_train,X_test,y_train,y_test=train_test_split(tfidf_trans,dataset.Category,test_size=0.15,random_state=42)
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()

model.fit(X_train,y_train)
print("train accuracy: {0} \ntest accuracy: {1}".format(accuracy_score(y_train,model.predict(X_train)),
                                                       accuracy_score(y_test,model.predict(X_test))))
confusion_matrix(y_test,model.predict(X_test))
# from sklearn.model_selection import GridSearchCV
# C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
# gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# kernel=['rbf','linear']
# hyper={'kernel':kernel,'C':C,'gamma':gamma}
# gd=GridSearchCV(estimator=MultinomialNB(),param_grid=hyper,verbose=True)
# gd.fit(X,Y)
# print(gd.best_score_)
# print(gd.best_estimator_)