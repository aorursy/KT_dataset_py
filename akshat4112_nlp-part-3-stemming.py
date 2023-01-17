from nltk.stem import PorterStemmer 

from nltk.tokenize import word_tokenize
obj = PorterStemmer()
words = ["play","player","playing","played"]
for w in words: 

    print(w, " : ",obj.stem(w))