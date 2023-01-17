import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import spacy

from sklearn.svm import LinearSVC
data = pd.read_csv('/kaggle/input/amazon-music-reviews/Musical_instruments_reviews.csv')

data.head()
(data.overall.value_counts() / data.overall.count()) * 100
data.loc[data.overall < 4, 'overall'] = 0

data.loc[data.overall >= 4, 'overall'] = 1

(data.overall.value_counts() / data.overall.count()) * 100
(data.isnull().sum() / data.count()) * 100
data.dropna(subset=['reviewText'], inplace=True)
(data.isnull().sum() / data.count()) * 100
nlp = spacy.load('en_core_web_lg')
with nlp.disable_pipes():

    doc_vectors = np.array([nlp(text).vector for text in data.reviewText])

    

doc_vectors.shape
X_train, X_test, y_train, y_test = train_test_split(doc_vectors, data.overall, test_size=0.1)
svc = LinearSVC(dual=False, max_iter=10000)

svc.fit(X_train, y_train)

print(f"Accuracy {svc.score(X_test, y_test) * 100}")
def predict(text):

    with nlp.disable_pipes():

        test_vector = nlp(text).vector

    return svc.predict([test_vector])[0]
print(predict("I really enjoyed this product."))

print(predict("The sound was terrible. Product is overpriced."))

print(predict("The reed was very loose, but overall you get what you pay for."))

print(predict("You get so much for the price: the thousands of sounds/software that come with it are really useful, especially for a starting producer . The product is well designed and the built is solid. Would totally recommend !"))

print(predict("Software that came in the bundle took nearly 3 hours to upload, then found out it was corrupt!"))