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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

spam_df=pd.read_csv("/kaggle/input/spam-mails-dataset/spam_ham_dataset.csv")
spam_df.head()
X_text=spam_df["text"]
type(X_text)
X_text.head()
y_label=spam_df["label"]
y_label.head()
y_label[y_label.isnull()].count()
X_text_train, X_text_test, y_label_train, y_label_test = train_test_split(X_text,
                                                      y_label, 
                                                    test_size=0.33,
                                                    random_state=53)
(X_text_train.shape),(X_text_test.shape),(y_label_train.shape),(y_label_test.shape)
tfIdfVecorizer=TfidfVectorizer(stop_words='english')
tfIdfVecorizer
count_train=tfIdfVecorizer.fit_transform(X_text_train)
count_train
tfIdfVecorizer.get_feature_names()[0:10]
len(tfIdfVecorizer.get_stop_words())
tfIdfVecorizer.get_stop_words()
count_test=tfIdfVecorizer.transform(X_text_test)
count_test
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(count_train,y_label_train)
label_pred = model.predict(count_test)
score=accuracy_score(y_label_test,label_pred)
score
con=confusion_matrix(y_label_test,label_pred)
con
navie_classifier=MultinomialNB()
##Fit the classifier to the training data
navie_classifier.fit(count_train,y_label_train)
## predict the data
label_pred=navie_classifier.predict(count_test)
label_pred
score=accuracy_score(y_label_test,label_pred)
score
con=confusion_matrix(y_label_test,label_pred)
con
print(classification_report(y_label_test,label_pred))

