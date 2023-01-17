import nltk

from nltk.stem import WordNetLemmatizer 

nltk.download('wordnet')
Lemmatizer = WordNetLemmatizer()

print("words :", Lemmatizer.lemmatize("words")) 

print("corpora :", Lemmatizer.lemmatize("corpra")) 

  

# a denotes adjective in "pos" 

print("better :", Lemmatizer.lemmatize("better", pos ="a")) 