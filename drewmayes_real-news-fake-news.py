import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.utils import shuffle

from sklearn.svm import SVC

from sklearn.feature_extraction.text import TfidfVectorizer
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

true_data = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv");

fake_data = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv");
true_data.head()
#Inserting the identifier that the news report was true

true_data.insert(4, "Label", "true", True)

true_data.head()
#Inserting the identifer that the news report was false

fake_data.insert(4, "Label", "fake", True)

fake_data.head()
#Combining the two datasets

everything_data = true_data.append(fake_data)

everything_data.head()
#Shuffling after appending

everything_data = shuffle(everything_data)

everything_data.head()
#Reordering indicies

everything_data.reset_index(inplace = True, drop = True)

everything_data.head()
#This iteration looks exclusivly at a corpus of the 'text' part of each news report

from sklearn.feature_extraction import text

corpus = everything_data['text']

vectorizer = text.CountVectorizer(binary=True).fit(corpus)

vectorized_text = vectorizer.transform(corpus)
TfidF = text.TfidfTransformer(norm='l1')

tfidf = TfidF.fit_transform(vectorized_text)
#Linear Support Vector Machine

from sklearn.svm import LinearSVC

labels = everything_data.Label

features = tfidf

model = LinearSVC()

X_train, X_test, y_train, y_test= train_test_split(features, labels, test_size=0.33, random_state=0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))