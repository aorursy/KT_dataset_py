import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize



text = "Hello, this is tokenizing. Which is helpfull? I do belive this is good way."



sent_token = sent_tokenize(text)



print("This is sent_tokenizer")

sent_token

word_token = word_tokenize(text)



print("This is word_tokenizer")

word_token