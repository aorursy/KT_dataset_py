# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train="../input/train-balanced-sarcasm.csv"
from nltk.tokenize import sent_tokenize,word_tokenize
import numpy as nm
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
pst=PorterStemmer()
from sklearn.feature_extraction.text import CountVectorizer

frame1=pd.read_csv(train)
frame=frame1[0:10000]
list_all_words=[]

for comment1 in frame.comment:
    temp=word_tokenize(comment1)
    for word in temp:
        pst.stem(word)
        list_all_words.append(word)  
        '''list_stop=stopwords.words('english')
for word in list_all_words:
    if word in list_stop:
        list_all_words.remove(word)'''
target_list=[]
for lab in frame.label:
    target_list.append(lab)
y=pd.Series(target_list)
count_vect=CountVectorizer(input=list_all_words,lowercase=True,stop_words='english',min_df=2)
X_count_vect=count_vect.fit_transform(frame.comment)
X_names=count_vect.get_feature_names()
X_count_vect=pd.DataFrame(X_count_vect.toarray(),columns=X_names)
print(X_count_vect)

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.metrics import classification_report

#NAIVES BAYES CLASSIFIER
X_train_csv,X_test_csv,y_train_csv,y_test_csv=train_test_split(X_count_vect,y,test_size=0.25,random_state=5)
gnb=GaussianNB()
y_pred_gnb=gnb.fit(X_train_csv,y_train_csv).predict(X_test_csv)
fit_cb=MultinomialNB()
y_mnb=fit_cb.fit(X_train_csv,y_train_csv)
y_pred_mnb=y_mnb.predict(X_test_csv)
print(metrics.accuracy_score(y_test_csv,y_pred_mnb))

print('execute')

#SVM
svc = LinearSVC()
svc.fit(X_train_csv, y_train_csv)
#y_pred = svc.predict(x_train)
predicted_class = svc.predict(X_test_csv)
print(metrics.accuracy_score(y_test_csv,predicted_class))
print('execute')