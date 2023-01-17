import pandas as pd

import nltk

%matplotlib inline

traindf = pd.read_csv('../input/train.csv')

traindf.head()

authorsdupl=traindf['author'].to_list()

authors=set(authorsdupl)

# print(authors)
dictionaryauthors={}

for x in authors:

    dictionaryauthors[x]=""



text=traindf['text'].to_list()



for i in range(len(text)):

    

    dictionaryauthors[authorsdupl[i]]+=text[i]



# print(dictionaryauthors['HPL'])



    

    
stories_by_author_tokens={}

stories_by_author_length_distributions = {}

for author in authors:

    tokens = nltk.word_tokenize(dictionaryauthors[author])



    # Filter out punctuation

    stories_by_author_tokens[author] = ([token for token in tokens

                                            if any(c.isalpha() for c in token)])



    # Get a distribution of token lengths

    token_lengths = [len(token) for token in stories_by_author_tokens[author]]

    stories_by_author_length_distributions[author] = nltk.FreqDist(token_lengths)

    stories_by_author_length_distributions[author].plot(15,title=author)
testdf=pd.read_csv("../input/test.csv")



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split



vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform(traindf['text'])

print(vectors.shape)

X_train, X_test, y_train, y_test = train_test_split(vectors, authorsdupl, test_size=0.2, random_state=1337)

%%time

svm = LinearSVC()

svm.fit(X_train, y_train)
predictions = svm.predict(X_test)

print(list(predictions[0:10]))

print(y_test[:10])
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))





from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(traindf['text'])

print(X_train_counts.shape)



from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

print(X_train_tfidf.shape)

# predictionstest=svm.predict(vectorstest)

# print(list(predictionstest[0:10]))
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(X_train_tfidf,traindf['author'])
X_new_counts = count_vect.transform(testdf['text'])

X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

print(predicted[:10])
from sklearn.linear_model import SGDClassifier

clf1=SGDClassifier(loss='hinge', penalty='l2',

                         alpha=1e-3, random_state=42,

                      max_iter=5, tol=None).fit(X_train_tfidf,traindf['author'])

predictedsvm=clf1.predict(X_new_tfidf)

print(predictedsvm[:10])