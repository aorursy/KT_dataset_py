from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer()



corpus = [

 'the brown fox jumped over the brown dog',

 'the quick brown fox',

 'the brown brown dog',

 'the fox ate the dog'

]



X = vectorizer.fit_transform(corpus)



print(vectorizer.get_feature_names())

print(X.toarray())