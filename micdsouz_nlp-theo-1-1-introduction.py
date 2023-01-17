from nltk.tokenize import sent_tokenize
text = "Awesome! I am learning NLP."
for sent in sent_tokenize(text):
    print(sent)
from spacy.lang.en import English
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))
doc = nlp('Hello, world. Here are two sentences.')
for sent in doc.sents:
    print(sent)
import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')

doc = nlp('My name is Michael, I live in Bangalore.')
displacy.render(doc,style='dep',jupyter=True,options={'distance':140})
from nltk.tokenize import word_tokenize
text = "God is Great! I won a lottery."
for word in word_tokenize(text):
    print(word)
from spacy.lang.en import English
nlp = English()
# Created by processing a string of text with the nlp object
doc = nlp("God is Great! I won a lottery.")

# Iterate over tokens in a Doc
for token in doc:
    print(token.text)
## Download Wordnet through NLTK in python console:
import nltk
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer 

# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Lemmatize Single Word
print(lemmatizer.lemmatize("bats"))
print(lemmatizer.lemmatize("are"))
print(lemmatizer.lemmatize("feet"))

# Define the sentence to be lemmatized
sentence = "The striped bats are hanging on their feet for best support"

# Tokenize: Split the sentence into words
word_list = nltk.word_tokenize(sentence)
print(word_list)

# Lemmatize list of words and join
lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
print(lemmatized_output)
print(lemmatizer.lemmatize("stripes", 'v'))
print(lemmatizer.lemmatize("stripes", 'n'))  
import spacy

# Initialize spacy 'en' model, keeping only tagger component needed for lemmatization
nlp = spacy.load('en', disable=['parser', 'ner'])

sentence = "The striped bats are hanging on their feet for best support"

# Parse the sentence using the loaded 'en' model object `nlp`
doc = nlp(sentence)

# Extract the lemma for each token and join
" ".join([token.lemma_ for token in doc])
from textblob import TextBlob, Word

# Lemmatize a word
word = 'stripes'
w = Word(word)
w.lemmatize()
# Lemmatize a sentence
sentence = "The striped bats are hanging on their feet for best"
sent = TextBlob(sentence)
" ". join([w.lemmatize() for w in sent.words])
import pattern
from pattern.en import lemma, lexeme

sentence = "The striped bats were hanging on their feet and ate best fishes"
" ".join([lemma(wd) for wd in sentence.split()])
# Lexeme's for each word 
[lexeme(wd) for wd in sentence.split()]
from pattern.en import parse
print(parse('The striped bats were hanging on their feet and ate best fishes',lemmata=True, tags=False, chunks=False))
from gensim.utils import lemmatize
sentence = "The striped bats were hanging on their feet and ate best fishes"
[wd.decode('utf-8').split('/')[0] for wd in lemmatize(sentence)]