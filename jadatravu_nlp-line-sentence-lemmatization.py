from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
sentence = "Mary leaves the room"
word_tokens = sentence.split(" ")
word_tokens
[wordnet_lemmatizer.lemmatize(test) for test in word_tokens]
!pip install pywsd
from pywsd.utils import lemmatize_sentence
lemmatize_sentence("Mary leaves the room")