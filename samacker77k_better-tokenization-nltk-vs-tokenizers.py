from nltk.tokenize import word_tokenize
from datetime import datetime
sentence = 'The town was fairly large with a dozen or\
            so business buildings on each side of the street but, as I said, most were closed.'
def nltkTokenizer(sentence):
    start = datetime.now()
    tokens = word_tokenize(sentence)
    end = datetime.now()
    time_taken = (end-start).microseconds
    print("Tokens\n")
    print(tokens)
    print("-"*50)
    print("\nTime taken\n")
    print("-"*10)
    print(str(time_taken)+" microseconds")
nltkTokenizer(sentence)
!python3 -m pip install tokenizers
# Download pre-trained vocabulary file

!wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
from tokenizers import (BertWordPieceTokenizer)
tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
def hfTokenizer(text):
    start = (datetime.now())
    print(tokenizer.encode(text).tokens)
    end = (datetime.now())
    print("Time taken - {} microseconds".format((end-start).microseconds))

hfTokenizer(sentence)
