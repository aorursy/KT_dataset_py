import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#importing library for regular expressions

import re



# get the hashtags set with the hashtags that appear in the text more then "level" times

def find_hashtags(train, level=0): 

    train["text"] = train["text"].apply(lambda x: x.lower()) #we map all the text to lowercase

    train["tag"] = train["text"].apply(lambda x: re.findall(r"#(\w+)", x)) # we find all of the hashtags

    tags=set([]) 

    for loc_tags in train["tag"]: 

        for tag in loc_tags:

            if not (tag is None):

                train[tag]=train["text"].apply(lambda x: int("#"+tag in x))

                if train[tag].sum()>level:

                    tags.add(tag)

                del train[tag]

    del train["tag"]

    return tags



#we get extra features for the dataset, that illustrated does the specific hashtag belong to the text

def to_hashtags(train, tags):

    for tag in tags:

        train[tag]=train["text"].apply(lambda x: int("#"+tag in x))

    train["text"]=train["text"].apply(lambda x: x.replace("#", ""))

    del train["text"]

    del train["location"]

    del train["keyword"]

    del train["id"]

    return train



#now just load data get the hashtags and try to figure can the hashtags alone give the information about markup

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

tags = find_hashtags(train)

train_df = to_hashtags(train, tags) 

#right here we get the embedding of 0 and 1 where the text either contains the hashtag or not

test_df = to_hashtags(test, tags)

#N.B. note, that we use the same "tags" set for the test sample, since we got to get the embedding

#into the same space. We do not find the tags for the test data.





#by the standard approach we can validate the models by spliting marked data into train and test.

from sklearn.model_selection import train_test_split



train_target = train_df["target"]

train_features = train_df.copy() # here we use copy method not to change the "train_df" itself during processing

del train_features["target"]



X_train, X_test, y_train, y_test = train_test_split(train_features, train_target, test_size=0.2)



from sklearn.metrics import precision_recall_fscore_support 

#it is also usefull not only to score the model, but to find the precision and recall of the model.



from sklearn.svm import SVC #Support Vector Classifier

model = SVC()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))



from sklearn.linear_model import SGDClassifier #Support Vector Classifier trained with Stochastic Gradient Descent

model = SGDClassifier()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))



from sklearn.naive_bayes import MultinomialNB #Naive Bayes

model = MultinomialNB()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))
import re



def find_keywords(train, level=0):

    train["keyword"] = train["keyword"].apply(lambda x: str(x).lower().split(","))

    keywords=set([])

    for loc_keywords in train["keyword"]:

        for keyword in loc_keywords:

            if not (keyword == "nan"):

                train[keyword]=train["keyword"].apply(lambda x: int(keyword in x))

                if train[keyword].sum()>level:

                    keywords.add(keyword)

                del train[keyword]

    return keywords



def to_keywords(train, keywords):

    train["keyword"] = train["keyword"].apply(lambda x: str(x).lower().split(","))

    for keyword in keywords:

        train[keyword]=train["keyword"].apply(lambda x: int(keyword in x))

    del train["text"]

    del train["location"]

    del train["id"]

    del train["keyword"]

    return train



train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

keywords = find_keywords(train)

train_df = to_keywords(train, keywords)

test_df = to_keywords(test, keywords)



from sklearn.model_selection import train_test_split



train_target = train_df["target"]

train_features = train_df.copy()

del train_features["target"]



X_train, X_test, y_train, y_test = train_test_split(train_features, train_target, test_size=0.2)



from sklearn.metrics import precision_recall_fscore_support



from sklearn.svm import SVC

model = SVC()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))



from sklearn.linear_model import SGDClassifier

model = SGDClassifier()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))



from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))
from nltk.corpus import stopwords #we must reduce the stopwords from the text



train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

train["text"] = train["text"].apply(lambda x: x.lower())



stopWords = set(stopwords.words('english')) 

#so we just take the stopwords of English language, since the text is in English.

#I am not that sophisticated in languages, but I guess that is would be far more

#fun to clean the stopwords out of agglutinative languages such as German, or

#if you are processing hieroglyphic text.



for w in stopWords:

    train["text"] = train["text"].apply(lambda x: x.replace(" "+w+" ", " "))



    

#Here we use the TF-IDF model for text embedding

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()



train["text"] =train["text"].apply(lambda x: x.replace("#","")) #for sure just make text "plain".

#print(train["text"])



#here also, it is usefull to replace urls with "site" word,

#especially if you already extracted the urls.



tfidf_train = vectorizer.fit_transform(train["text"])

#print(vectorizer.get_feature_names())



#The problem is that the dimention of the text embedding in TF-IDF model or BOW model

#is too large, so we got to reduce the dimentionality, thus, we try to make some

#feature as the Principal Component Analysis.



from sklearn.decomposition import PCA

pca = PCA(n_components=500)

lower_dim_tfidf_train = pca.fit_transform(tfidf_train.todense())

train_df = lower_dim_tfidf_train



#by standard make the validation

from sklearn.model_selection import train_test_split



train_target = train["target"]

train_features = train_df.copy()



X_train, X_test, y_train, y_test = train_test_split(train_features, train_target, test_size=0.2)



#and find precision and recall, not only the accuracy.

from sklearn.metrics import precision_recall_fscore_support



from sklearn.svm import SVC

model = SVC()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))



from sklearn.linear_model import SGDClassifier

model = SGDClassifier()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))



from sklearn.tree import DecisionTreeClassifier #here we add Decision Tree Classifier 

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))



from sklearn.ensemble import RandomForestClassifier  #here we add Random Forest Classifier

model = RandomForestClassifier()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))
some_texts = train["text"]



from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(some_texts)]



model = Doc2Vec(documents, vector_size=1000, workers=4, epochs=1)

textdf = model.docvecs



from sklearn.model_selection import train_test_split



train_target = train["target"]

train_features = textdf



X_train, X_test, y_train, y_test = train_test_split(train_features, train_target, test_size=0.2)



from sklearn.metrics import precision_recall_fscore_support



from sklearn.svm import SVC

model = SVC()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))



from sklearn.linear_model import SGDClassifier

model = SGDClassifier()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))



from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))



from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))
# определим датасет'

some_texts = train["text"]



from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(some_texts)]



model = Doc2Vec(documents, vector_size=1000, workers=4, epochs=10)

textdf = model.docvecs



from sklearn.model_selection import train_test_split



train_target = train["target"]

train_features = textdf



X_train, X_test, y_train, y_test = train_test_split(train_features, train_target, test_size=0.2)



from sklearn.metrics import precision_recall_fscore_support



from sklearn.svm import SVC

model = SVC()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))



from sklearn.linear_model import SGDClassifier

model = SGDClassifier()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))



from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))



from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))
# определим датасет'

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

some_texts = train["text"].apply(lambda x: x.lower()).apply(lambda x: x.replace("#",""))



from nltk.corpus import stopwords



stopWords = set(stopwords.words('english'))



for w in stopWords:

    some_texts = some_texts.apply(lambda x: x.replace(" "+w+" ", " "))



from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(some_texts)]



model = Doc2Vec(documents, vector_size=1000, workers=4, epochs=50)

textdf = model.docvecs



from sklearn.model_selection import train_test_split



train_target = train["target"]

train_features = textdf



X_train, X_test, y_train, y_test = train_test_split(train_features, train_target, test_size=0.2)



from sklearn.metrics import precision_recall_fscore_support



from sklearn.svm import SVC

model = SVC()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))



from sklearn.linear_model import SGDClassifier

model = SGDClassifier()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))



from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))



from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

tfidf_train = vectorizer.fit_transform(train["text"].apply(lambda x: x.lower()).apply(lambda x: x.replace("#","")))

#print(vectorizer.get_feature_names())



train_df = tfidf_train.todense()



from sklearn.model_selection import train_test_split



train_target = train["target"]

train_features = train_df.copy()



X_train, X_test, y_train, y_test = train_test_split(train_features, train_target, test_size=0.2)



from sklearn.metrics import precision_recall_fscore_support



from sklearn.svm import SVC

model = SVC()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))



from sklearn.linear_model import SGDClassifier

model = SGDClassifier()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))



from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(precision_recall_fscore_support(model.predict(X_test), y_test))
#!apt install default-libmysqlclient-dev -y

#!pip install pattern
"""import pattern.en as ptrn



from nltk import pos_tag



for i in range(5):

    print(train["text"].iloc[i].split())

    words = word_tokenize(train["text"].iloc[i])

    wordsFiltered = []



    for w in words:

        if w not in stopWords:

            wordsFiltered.append(w)

    print(wordsFiltered)

    for w in wordsFiltered:

        print(ptrn.suggest(w))

    print()

"""