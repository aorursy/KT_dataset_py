import numpy as np 

import pandas as pd 

import nltk

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import confusion_matrix
df = pd.read_csv("../input/Reviews.csv", nrows = 50000, usecols = ["Score", "Text"])

df["Score"] = np.where(df["Score"] >= 4, "positive", "negative")
df.info()
df.head()
sns.countplot(data = df, x= df["Score"]).set_title("Score distribution", fontweight = "bold")

plt.show()
texts = df["Text"]



import nltk



texts_transformed = []

for review in texts:

    sentences = nltk.sent_tokenize(review)

    adjectives = []

    

    

    for sentence in sentences:

        words = nltk.word_tokenize(sentence) 

        words_tagged = nltk.pos_tag(words)

        

        adj_add = [adjectives.append(word_tagged[0]) for word_tagged in words_tagged if word_tagged[1] == "JJ"]

                

    texts_transformed.append(" ".join(adjectives)) 
X = texts_transformed

y = df["Score"]



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)



cv = CountVectorizer(max_features = 50)

cv.fit(X_train)



X_train = cv.transform(X_train)

X_test = cv.transform(X_test)
arr = X_train.toarray()



print(arr.shape)
model = MultinomialNB()

model.fit(X_train, y_train)



print(model.score(X_test, y_test))
y_test_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_test_pred))
def classifier(adjective):

    return model.predict(cv.transform([adjective]))

print(classifier('great'))

print(classifier('bad'))
adj = list(zip(model.coef_[0], cv.get_feature_names()))

adj = sorted(adj, reverse = True)

for a in adj:

    print(a)