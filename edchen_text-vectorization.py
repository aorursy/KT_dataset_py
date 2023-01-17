from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer()



corpus = [

  'the brown fox jumped over the brown dog',

  'the quick brown fox',

  'the brown brown dog',

  'the fox ate the dog'

]



X = vectorizer.fit_transform(corpus)



vectorizer.get_feature_names()

# ['ate', 'brown', 'dog', 'fox', 'jumped', 'over', 'quick', 'the']



X.toarray()           

# array([[0, 2, 1, 1, 1, 1, 0, 2],

#       [0, 1, 0, 1, 0, 0, 1, 1],

#       [0, 2, 1, 0, 0, 0, 0, 1],

#       [1, 0, 1, 1, 0, 0, 0, 2]], dtype=int64)
