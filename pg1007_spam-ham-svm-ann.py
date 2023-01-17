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
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding='iso-8859-1')

df
df=df.iloc[:,0:2].values
df=pd.DataFrame(df)
df.columns=['type','text']
df
y=df['type']
x=df['text']
from sklearn.preprocessing import LabelEncoder
leb = LabelEncoder()
y=leb.fit_transform(y) 

y
x
# library to clean data 
import re  
import nltk  
nltk.download('stopwords') 
from nltk.corpus import stopwords 
# Stemming
from nltk.stem.porter import PorterStemmer 
corpus = []  
  
for i in range(0, 5572):  
      
    review = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', df['text'][i])
    review  = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr',
                     review)
    review  = re.sub(r'£|\$', 'moneysymb', review)
    review = re.sub(
        r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        'phonenumbr', review)
    review  = re.sub(r'\d+(\.\d+)?', 'numbr', review)

    # collapse whitespace (spaces, line breaks, tabs) into a single space.
    # eliminate any leading or trailing whitespace.
    review  = re.sub(r'[^\w\d\s]', ' ', review)
    review = re.sub(r'\s+', ' ', review)
    review = re.sub(r'^\s+|\s+?$', '', review)

    review = review.lower()  
    review = review.split()  
    ps = PorterStemmer()   
    review = [ps.stem(word) for word in review 
                if not word in set(stopwords.words('english'))]  
    review = ' '.join(review)   
    corpus.append(review)  
# to extract useful ngrams and create bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
# to create bag of words model
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_ngrams = vectorizer.fit_transform(corpus)
X_ngrams.shape
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm

X_train, X_test, y_train, y_test = train_test_split( X_ngrams,y,test_size=0.3)

clf = svm.LinearSVC(loss='hinge',random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
metrics.f1_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
precision,recall,thresholds=precision_recall_curve(y_test,y_pred)
auc_recall_pre=auc(recall,precision)
auc_recall_pre
from sklearn.metrics import roc_curve
false_positive,true_positive,_=roc_curve(y_test,y_pred)
plt.plot(false_positive,true_positive,label='Linear SVM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC_curve')
plt.legend()
plt.show()
# Using Artificial Neural Network
import tensorflow as tf
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=10000, kernel_initializer='normal',activation='relu',input_dim=36228))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=5000,kernel_initializer='normal', activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=1000,kernel_initializer='normal', activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, kernel_initializer='normal',activation='sigmoid'))
# compiling model
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.summary()

ann.fit(X_train.todense(), y_train, validation_split=0.15,batch_size = 300, epochs = 15)
y_pred=ann.predict(X_test.todense())
y_pred
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
