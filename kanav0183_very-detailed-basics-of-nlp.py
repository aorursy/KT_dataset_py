from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect
corpus = ['Hi my name is kanav.','I love reading.','Kanav loves reading scripts.']
X= vect.fit_transform(corpus)
X # note the dimensions of X(3X9) means 3 rows and 9 columns. 
vect.get_feature_names()
X.toarray()
vect.transform(['hi,whats your name?.']).toarray()
import nltk
porter = nltk.PorterStemmer()
[porter.stem(t) for t in vect.get_feature_names()]
list(set([porter.stem(t) for t in vect.get_feature_names()]))
WNlemma = nltk.WordNetLemmatizer()
[WNlemma.lemmatize(t) for t in list(set([porter.stem(t) for t in vect.get_feature_names()]))]
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run",'v'))
from nltk import word_tokenize, pos_tag
sentence = "Kaggle is very good learning platform,Do you agree?"
sen_token = word_tokenize(sentence)
pos_tag(sen_token)

