import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

fakedataset = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")

fake = fakedataset[:5000]
realdataset = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")

real = realdataset[:5000]
real["class"] = 1

fake["class"] = 0
real["text"] = real["title"] + " " + real["text"]

fake["text"] = fake["title"] + " " + fake["text"]



real.drop(["subject", "date", "title"], axis = 1)

fake.drop(["subject", "date", "title"], axis = 1)
dataset = real.append(fake, ignore_index = True)
del real, fake
import nltk



nltk.download("stopwords")

nltk.download("punkt")
import re

import string

stopwords = nltk.corpus.stopwords.words('english')

stemmer = nltk.PorterStemmer()



def count_punct(text):

    count = sum([1 for char in text if char in string.punctuation])

    return round(count/(len(text) - text.count(" ")), 3)*100



dataset['body_len'] = dataset['text'].apply(lambda x: len(x) - x.count(" "))

dataset['punct%'] = dataset['text'].apply(lambda x: count_punct(x))



def clean_text(text):

    text = "".join([word.lower() for word in text if word not in string.punctuation])

    tokens = re.split('\W+', text)

    text = [stemmer.stem(word) for word in tokens if word not in stopwords]

    return text

    
from sklearn.model_selection import train_test_split



X=dataset[['text', 'body_len', 'punct%']]

y=dataset['class']



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer





tfidf_vect = TfidfVectorizer(analyzer=clean_text)

tfidf_vect_fit = tfidf_vect.fit(X_train['text'])



tfidf_train = tfidf_vect_fit.transform(X_train['text'])

tfidf_test = tfidf_vect_fit.transform(X_test['text'])



X_train_vect = pd.concat([X_train[['body_len', 'punct%']].reset_index(drop=True), 

           pd.DataFrame(tfidf_train.toarray())], axis=1)

X_test_vect = pd.concat([X_test[['body_len', 'punct%']].reset_index(drop=True), 

           pd.DataFrame(tfidf_test.toarray())], axis=1)



X_train_vect.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.metrics import accuracy_score as acs

import matplotlib.pyplot as plt

import seaborn as sns
rf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1)



rf_model = rf.fit(X_train_vect, y_train)



y_pred = rf_model.predict(X_test_vect)



precision, recall, fscore, train_support = score(y_test, y_pred, pos_label=1, average='binary')

print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(

    round(precision, 3), round(recall, 3), round(fscore,3), round(acs(y_test,y_pred), 3)))



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

class_label = [0, 1]

df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)

sns.heatmap(df_cm, annot=True, fmt='d')

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
