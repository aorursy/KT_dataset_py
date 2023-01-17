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
df=pd.read_csv("../input/fake-news/train.csv")
df.head()
x=df.drop('label',axis=1)
x.head()
y=df['label']
y.head()
x.shape
df=df.dropna()
df.shape
df.head(10)
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
message=df.copy()
message.head(10)
message.reset_index(inplace=True)
message.head(10)
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
ps=PorterStemmer()
corpus=[]

for i in range(0,len(message)):
    review=re.sub('[^a-zA-Z]',' ',message['title'][i])
    review=review.lower()
    review=review.split()
    review=[word for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
corpus
cv=CountVectorizer(max_features=5000,ngram_range=(1,3))
x=cv.fit_transform(corpus).toarray()
x
x.shape
y=message['label']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=1)
cv.get_feature_names()[50:75]
cv.get_params()
a_df=pd.DataFrame(xtrain,columns=cv.get_feature_names())
a_df.head()
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(xtrain,ytrain)
from sklearn.metrics import accuracy_score
import itertools
from sklearn.metrics import confusion_matrix

ypred=model.predict(xtest)
score=accuracy_score(ypred,ytest)
score*100
con=confusion_matrix(ytest,ypred)
plot_confusion_matrix(con,classes=['Fake','Real'])
from sklearn.linear_model import PassiveAggressiveClassifier
mod=PassiveAggressiveClassifier(n_iter_no_change=50)
mod.fit(xtrain,ytrain)
from sklearn.metrics import accuracy_score
import itertools
from sklearn.metrics import confusion_matrix

ypred=mod.predict(xtest)
score=accuracy_score(ypred,ytest)
score*100
con=confusion_matrix(ytest,ypred)
plot_confusion_matrix(con,classes=['Fake','Real'])
classifier=MultinomialNB(alpha=0.1)
from sklearn.metrics import accuracy_score
previous_score=0
for alpha in np.arange(0,1,0.1):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(xtrain,ytrain)
    y_pred=sub_classifier.predict(xtest)
    score = accuracy_score(ytest, y_pred)
    if score>previous_score:
        classifier=sub_classifier
    print("Alpha: {}, Score : {}".format(alpha,score))
features=cv.get_feature_names()
features[:20]
classifier.coef_[0]
sorted(zip(classifier.coef_[0],features),reverse=True)[:20]
sorted(zip(classifier.coef_[0],features))[:20]
