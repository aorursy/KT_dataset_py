text = """There are multiple ways we can perform tokenization on given text data. We can choose any method based on langauge, library and purpose of modeling."""

# Split text by whitespace

tokens = text.split()

print(tokens)
# Lets split the given text by full stop (.)

text = """Characters like periods, exclamation point and newline char are used to separate the sentences. But one drawback with split() method, that we can only use one separator at a time! So sentence tonenization wont be foolproof with split() method."""

text.split(". ") # Note the space after the full stop makes sure that we dont get empty element at the end of list.
import re



text = """There are multiple ways we can perform tokenization on given text data. We can choose any method based on langauge, library and purpose of modeling."""

tokens = re.findall("[\w]+", text)

print(tokens)
text = """Characters like periods, exclamation point and newline char are used to separate the sentences. But one drawback with split() method, that we can only use one separator at a time! So sentence tonenization wont be foolproof with split() method."""

tokens_sent = re.compile('[.!?] ').split(text) # Using compile method to combine RegEx patterns

tokens_sent
!pip install --user -U nltk
from nltk.tokenize import word_tokenize



text = """There are multiple ways we can perform tokenization on given text data. We can choose any method based on langauge, library and purpose of modeling."""

tokens = word_tokenize(text)

print(tokens)
from nltk.tokenize import sent_tokenize



text = """Characters like periods, exclamation point and newline char are used to separate the sentences. But one drawback with split() method, that we can only use one separator at a time! So sentence tonenization wont be foolproof with split() method."""

sent_tokenize(text)
!pip install spacy

!python -m spacy download en
# Load English model from spacy

from spacy.lang.en import English



# Load English tokenizer. 

# nlp object will be used to create 'doc' object which uses preprecoessing pipeline's components such as tagger, parser, NER and word vectors

nlp = English()



text = """There are multiple ways we can perform tokenization on given text data. We can choose any method based on langauge, library and purpose of modeling."""



# Now we will process above text using 'nlp' object. Which is use to create documents with linguistic annotations and various nlp properties

my_doc = nlp(text)



# Above step has already tokenized our text but its in doc format, so lets write fo loop to create list of it

token_list = []

for token in my_doc:

    token_list.append(token.text)



print(token_list)
# Load English tokenizer, tager, parser, NER and word vectors

nlp = English()



# Create the pipeline 'sentencizer' component

sbd = nlp.create_pipe('sentencizer')



# Add component to the pipeline

nlp.add_pipe(sbd)



text = """Characters like periods, exclamation point and newline char are used to separate the sentences. But one drawback with split() method, that we can only use one separator at a time! So sentence tonenization wont be foolproof with split() method."""



# nlp object is used to create documents with linguistic annotations

doc = nlp(text)



# Create list of sentence tokens



sentence_list =[]

for sentence in doc.sents:

    sentence_list.append(sentence.text)

print(sentence_list)
!pip install Keras
from keras.preprocessing.text import text_to_word_sequence



text = """There are multiple ways we can perform tokenization on given text data. We can choose any method based on langauge, library and purpose of modeling."""



tokens = text_to_word_sequence(text)

print(tokens)
from keras.preprocessing.text import text_to_word_sequence



text = """Characters like periods, exclamation point and newline char are used to separate the sentences. But one drawback with split() method, that we can only use one separator at a time! So sentence tonenization wont be foolproof with split() method."""



text_to_word_sequence(text, split= ".", filters="!.\n")
!pip install gensim
from gensim.utils import tokenize



text = """There are multiple ways we can perform tokenization on given text data. We can choose any method based on langauge, library and purpose of modeling."""



tokens = list(tokenize(text))

print(tokens)
from keras.preprocessing.text import text_to_word_sequence



text = """Characters like periods, exclamation point and newline char are used to separate the sentences. But one drawback with split() method, that we can only use one separator at a time! So sentence tonenization wont be fullproof with split() method."""



tokens = text_to_word_sequence(text, split= ".")

print(tokens)
from gensim.summarization.textcleaner import split_sentences



text = """Characters like periods, exclamation point and newline char are used to separate the sentences. But one drawback with split() method, that we can only use one separator at a time! So sentence tonenization wont be foolproof with split() method."""



list(split_sentences(text))