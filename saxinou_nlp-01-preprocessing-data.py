import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



"""

Importation des librairies

"""

import re

import string

import numpy as np

import nltk

from collections import Counter

from nltk import tokenize
text1 = "ThIs's   ã sent tokenize test  .  this's sent two. is this sent three? sent 4 is cool! Now it's your turn."

print(text1)
""" 

Naive Split

"""

print("With a naive split \n", text1.split(" "))



"""

Tokenizing text into words

"""

import nltk

tokens = nltk.word_tokenize(text1)

print("\nTokenizing text into words With NLTK \n", tokens)



"""

Equivalent method with TreebankWordTokenizer

"""

from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

print("\nEquivalent method with TreebankWordTokenizer \n", tokenizer.tokenize(text1))





"""

Equivalent method with WordPunctTokenizer  

"""

from nltk.tokenize import WordPunctTokenizer 

tokenizer = WordPunctTokenizer()

print("\nEquivalent method with WordPunctTokenizer \n", tokenizer.tokenize(text1))
def tokenize_word_text(text): 

    tokens = nltk.word_tokenize(text) 

    tokens = [token.strip() for token in tokens] 

    return tokens 

# Launching example:

# tokenize_word_text(text1)
"""

Sentence tokenize in NLTK with sent_tokenize 

The sent_tokenize function uses an instance of NLTK known as PunktSentenceTokenizer

This instance of NLTK has already been trained to perform tokenization on different European languages on the basis of letters or punctuation that mark the beginning and end of sentences

"""

from nltk.tokenize import sent_tokenize

print("Sentence tokenize in NLTK With sent_tokenize \n", sent_tokenize(text1))



"""

Autres manières 

"""

## using PunktSentenceTokenizer for sentence tokenization 

punkt_st = nltk.tokenize.PunktSentenceTokenizer() 

sample_sentences = punkt_st.tokenize(text1) 

print("\nSentence tokenize with PunktSentenceTokenizer \n ", print(sample_sentences) )

"""

Two ways to proceed:

"""



""" 

Method 1

"""

german_text = u"Die Orgellandschaft Südniedersachsen umfasst das Gebiet der Landkreise Goslar, Göttingen, Hameln-Pyrmont, Hildesheim, Holzminden, Northeim und Osterode am Harz sowie die Stadt Salzgitter. Über 70 historische Orgeln vom 17. bis 19. Jahrhundert sind in der südniedersächsischen Orgellandschaft vollständig oder in Teilen erhalten. "

print("\n",sent_tokenize(german_text, language='german'))

print("\n",sent_tokenize(german_text, language='polish'))



""" 

Method 2

"""

import nltk.data



# English

tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

print("English token ", tokenizer.tokenize(text1))



# French

french_tokenizer = nltk.data.load("tokenizers/punkt/french.pickle")

print("\nFrench token ", french_tokenizer.tokenize("Il fait beau aujourd'hui. Vas-tu sortir ? N'y a-t-il pas du pain ?"))
"""

Starting point : tokens

"""

tokens = tokenize_word_text(text1)

print(tokens)
"""

Converting all letters to lower or upper case (common : lower case)

"""

def convert_letters(tokens, style = "lower"):

    if (style == "lower"):

        tokens = [token.lower() for token in tokens]

    else :

        tokens = [token.upper() for token in tokens]

    return(tokens)

tokens = convert_letters(tokens)

print(tokens)
"""

Remove blancs on words

"""

def remove_blanc(tokens):

    tokens = [token.strip() for token in tokens]

    return(tokens)

tokens = remove_blanc(tokens)

print(tokens)
"""

On sentence  

"""

def remove_before_token(sentence, keep_apostrophe = False):

    sentence = sentence.strip()

    if keep_apostrophe:

        PATTERN = r'[?|$|&|*|%|@|(|)|~]'

        filtered_sentence = re.sub(PATTERN, r' ', sentence)

    else :

        PATTERN = r'[^a-zA-Z0-9]'

        filtered_sentence = re.sub(PATTERN, r' ', sentence)

    return(filtered_sentence)

remove_before_token(text1)



"""

After tokenization  

"""

def remove_after_token(tokens): 

    pattern = re.compile('[{}]'.format(re.escape(string.punctuation))) 

    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens]) 

    filtered_text = ' '.join(filtered_tokens) 

    return filtered_text 

remove_after_token(tokens)
"""

Expanding contraction 

"""

CONTRACTION_MAP = {"ain't": "is not", "aren't": "are not","can't": "cannot", 

                   "can't've": "cannot have", "'cause": "because", "could've": "could have", 

                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 

                   "doesn't": "does not", "don't": "do not", "hadn't": "had not", 

                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 

                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 

                   "he'll've": "he he will have", "he's": "he is", "how'd": "how did", 

                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 

                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 

                   "I'll've": "I will have","I'm": "I am", "I've": "I have", 

                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 

                   "i'll've": "i will have","i'm": "i am", "i've": "i have", 

                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 

                   "it'll": "it will", "it'll've": "it will have","it's": "it is", 

                   "let's": "let us", "ma'am": "madam", "mayn't": "may not", 

                   "might've": "might have","mightn't": "might not","mightn't've": "might not have", 

                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 

                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 

                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",

                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 

                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 

                   "she's": "she is", "should've": "should have", "shouldn't": "should not", 

                   "shouldn't've": "should not have", "so've": "so have","so's": "so as", 

                   "this's": "this is",

                   "that'd": "that would", "that'd've": "that would have","that's": "that is", 

                   "there'd": "there would", "there'd've": "there would have","there's": "there is", 

                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 

                   "they'll've": "they will have", "they're": "they are", "they've": "they have", 

                   "to've": "to have", "wasn't": "was not", "we'd": "we would", 

                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 

                   "we're": "we are", "we've": "we have", "weren't": "were not", 

                   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 

                   "what's": "what is", "what've": "what have", "when's": "when is", 

                   "when've": "when have", "where'd": "where did", "where's": "where is", 

                   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 

                   "who's": "who is", "who've": "who have", "why's": "why is", 

                   "why've": "why have", "will've": "will have", "won't": "will not", 

                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 

                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",

                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 

                   "you'll've": "you will have", "you're": "you are", "you've": "you have" } 

def expand_contractions(sentence, contraction_mapping): 

     

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),  

                                      flags=re.IGNORECASE|re.DOTALL) 

    def expand_match(contraction): 

        match = contraction.group(0) 

        first_char = match[0] 

        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())                        

        expanded_contraction = first_char+expanded_contraction[1:] 

        return expanded_contraction 

         

    expanded_sentence = contractions_pattern.sub(expand_match, sentence) 

    return expanded_sentence 

     

expanded_corpus = [expand_contractions(txt, CONTRACTION_MAP)  

                     for txt in sent_tokenize(text1)]     



print ("Text before expanding contraction : \n ", text1)

print ("\n Text after expanding contraction : \n ",expanded_corpus) 
"""

Removing accent marks and other diacritics - before tokens words

"""

import unidecode

def remove_accent_before_tokens(sentences):

    res = unidecode.unidecode(sentences)

    return(res)

tmp = remove_accent_before_tokens(text1)

print("After removing accent markes before tokenize words : \n", tmp)



"""

Removing accent marks and other diacritics - After tokens words

"""



def remove_accent_after_tokens(tokens):

    tokens = [unidecode.unidecode(token) for token in tokens]

    return(tokens)

tokens = remove_accent_after_tokens(tokens)

print("After removing accent markes ", tokens)
"""

Use a stopwords list

"""

stopword_list = nltk.corpus.stopwords.words('english')

print("StopWords List in English : \n", stopword_list)



""" 

Create your own stopwords list

"""

stopwords = ['a', 'about', 'above', 'across', 'after', 'afterwards']

stopwords += ['again', 'against', 'all', 'almost', 'alone', 'along']

stopwords += ['this', 'is', 'your']



"""

Fonction StopWords

"""

def removeStopwords_after_tokens(wordlist, stopwords):

    return [w for w in wordlist if w not in stopwords]



# Example of calling the function : 

# tokens = nltk.word_tokenize(text1)

# removeStopwords_before_tokens(tokens, stopwords)
tokens = nltk.word_tokenize(text1)

type(tokens)
# Given a list of words, remove any that are in a list of stop words.

def removeStopwords_before_tokens(text, stopwords):

    tokens = nltk.word_tokenize(text)

    return [w for w in tokens if w not in stopwords]



# Example of calling the function : 

# removeStopwords_before_tokens(tokens, stopwords)
"""

Method 1 : Using the brown corpus in NLTK and "in" operator

"""

from nltk.corpus import brown

word_list = brown.words()

len(word_list)



word_set = set(word_list)

"looked" in word_set
"""

Method 2 : Peter Norvig sur un seul mot

"""



import re

import nltk

from collections import Counter



def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('../input/big.txt').read()))



def P(word, N=sum(WORDS.values())): 

    "Probability of `word`."

    return WORDS[word] / N



def correction(word): 

    "Most probable spelling correction for word."

    return max(candidates(word), key=P)



def candidates(word): 

    "Generate possible spelling corrections for word."

    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])



def known(words): 

    "The subset of `words` that appear in the dictionary of WORDS."

    return set(w for w in words if w in WORDS)



def edits1(word):

    "All edits that are one edit away from `word`."

    letters    = 'abcdefghijklmnopqrstuvwxyz'

    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]

    deletes    = [L + R[1:]               for L, R in splits if R]

    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]

    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]

    inserts    = [L + c + R               for L, R in splits for c in letters]

    return set(deletes + transposes + replaces + inserts)



def edits2(word): 

    "All edits that are two edits away from `word`."

    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

    

"""

Exemple avec des mots au hasard 

"""

print(correction('speling'))

print(correction('fial'))

print(correction("misstkaes"))
"""

Exemple avec notre text initial 

"""

text1 = "ThIs's   ã sent tokenize test  wiith miststakes in speling.  this's sent two. is this sent three? sent 4 is cool! Now it's your turn."



def correct_word_in_sentence(text):

    tokens = nltk.word_tokenize(text)

    r = [correction(token) for token in tokens]

    return (r)

tmp = ' '.join(correct_word_in_sentence(text1))

type(tmp)

print(tmp)
"""

Converting numbers into words









                                                TO DO







"""





"""

Driver

"""



text1 = "Heloo ! ThIs's  ã sent tokenize test  wiith misstkaes in speling.  this's sent two. is this sent three? sent 4 is cool! Now it's your turn."



def preprocessing(content):

    sentences = tokenize.sent_tokenize(content)

    cleaned_sentences = []

    for s in sentences :

        # 1- Lower case 

        s = s.lower()

        

        # 2- Supprimer les blancs :

        s = s.strip()  

        """

        A revoir

        """

        # 3- Unicode - supprimer les accents

        s = remove_accent_before_tokens(s)

        

        # 4- Expanding contraction

        # Remarque : on fait l'expansion en même temps que l'on transforme la list en str

        s = ''.join([expand_contractions(txt, CONTRACTION_MAP)

                     for txt in sent_tokenize(s)]  )

        

        # 5- Remove punctuation ?

        # A faire après avoir fait l'expanding contraction car sinon supprimer les ' qui symbolise la contraction

        s = remove_before_token(s)



        # 6- Correcting words

        # s = ' '.join(correct_word_in_sentence(s))

        s = ' '.join(correct_word_in_sentence(s))

        

        # 7- Remove StopWords

        s = ' '.join(removeStopwords_before_tokens(s, stopwords))

        

        # 8- Remove Numbers or transform it in char

        """

        TODO

        """

        

        # Enregistrement des resultats

        cleaned_sentences.append(s)

    return(cleaned_sentences) 

        



result = ' '.join(preprocessing(text1))

print(result)

from nltk.stem import WordNetLemmatizer

from nltk.stem import PorterStemmer

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.stem import SnowballStemmer

from nltk.stem.lancaster import LancasterStemmer

    

# With Porter Stemmer

ps = PorterStemmer()

example_words = ["python","pythoner","pythoning","pythoned","pythonly"]

print (">>> Exemple with Porter Stemmer")

for w in example_words:

    print(ps.stem(w))



# With Lancaster Stemmer

ls = LancasterStemmer()

print ("\n>>> Exemple with Lancaster Stemmer")

for w in example_words:

    print(ls.stem(w))

   

# With Snowball Stemmer

ss = SnowballStemmer("english")

print ("\n>>> Exemple with Snowball Stemmer")

for w in example_words:

    print(ss.stem(w))

    

"""

Example with a sentence

"""

ls = LancasterStemmer()

print ("\n>>> Exemple with a sentence ")

new_text = "It is important to by very pythonly while you are pythoning with python."

words = word_tokenize(new_text) # First, we tokenize

for w in words:

    print(ls.stem(w))

    

"""

Final example with our own text

"""

ls = LancasterStemmer()

words = word_tokenize(result) # First, we tokenize

for w in words:

    print("Our word before stemming : {} >>> after : {} ".format(w,ls.stem(w)))
# With WordNet Lemmatizer

example_words = ["python","pythoner","pythoning","pythoned","pythonly"]

from nltk.stem.wordnet import WordNetLemmatizer 

lem = WordNetLemmatizer()

print ("\n >>> Exemple with WordNet Lemmatizer")

for w in example_words:

    print("Lemmatisation of {} : {} ".format(w,lem.lemmatize(w)))

    

"""

Example with a sentence

"""

print ("\n>>> Exemple with a sentence ")

new_text = "It is important to by very pythonly while you are pythoning with python."

words = word_tokenize(new_text) # First, we tokenize

for w in words:

    print("Lemmatisation of {} : {} ".format(w,lem.lemmatize(w)))

    
ls = LancasterStemmer()

lem = WordNetLemmatizer()

def lexicon_normalization(text):

    words = word_tokenize(text) 

    print(words)

    

    # 1- Stemming

    words_stem = [ls.stem(w) for w in words]

    print(type(words_stem))

    print(words_stem)

    

    # 2- Lemmatization

    words_lem = [lem.lemmatize(w) for w in words_stem]

    print(type(words_lem))

    print(words_lem)



lexicon_normalization(result)