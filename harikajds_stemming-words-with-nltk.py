import nltk

from nltk.stem import PorterStemmer

from nltk.tokenize import sent_tokenize, word_tokenize
#Import stemmer

PS = PorterStemmer()
#Examples for same words

Example_words = ["python","pythoner","pythonly","pyth","pyth","pythoningly", "pythoned","pythoning"]
#Filtering PS using Example_words 

for words in Example_words:

    Stem = PS.stem(words)

    print(Stem)