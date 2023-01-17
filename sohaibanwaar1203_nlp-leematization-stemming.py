from nltk.tokenize import sent_tokenize, word_tokenize

import nltk 
example_text = "Hello there, how are you doing today? The weather is great and python is awesome. The sky is pinkish-blue. You should not eat cardboard."

example_text1="My Name is Sohaib I am good boy alia is a girl Iam"
sentences = sent_tokenize(example_text)

print(sentences)

print(len(sentences))

#print(type(abc))
sentences = sent_tokenize(example_text1)

print(sentences)

print(len(sentences))
words=word_tokenize(example_text)

print(words)

print(len(words))

words=word_tokenize(example_text1)

print(words)

print(len(words))
# Stop Words

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
# Downloads the data.

import nltk

nltk.download('stopwords')





# Using the stopwords.

from nltk.corpus import stopwords



# Initialize the stopwords

stoplist = stopwords.words('english')

stop_words = set(stopwords.words("english"))



print(type(stop_words))
stop_word_Example=[]

for i in words:

    for j in stop_words:

        if i ==j:

            stop_word_Example.append(i)

stop_word_Example            
filtered_sentence = []

for w in words:

    if w not in stop_words:

        filtered_sentence.append(w)

print(filtered_sentence)
# Stemming

from nltk.stem import PorterStemmer

from nltk.tokenize import word_tokenize
ps = PorterStemmer()
example_words = ["python","pythoner","pythoning","pythoned","pythonly"]
for w in example_words:

    print(ps.stem(w))
new_text = "it is very important to be pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
words = word_tokenize(new_text)

for w in words:

    print(ps.stem(w))
# POS Tagging

# Unsupervised Machine Learning Tokenizer

from nltk.tokenize import PunktSentenceTokenizer

from nltk.corpus import state_union

from nltk.tokenize import word_tokenize
train_text = state_union.raw("2005-GWBush.txt")

sample_text = state_union.raw("2006-GWBush.txt")
# ---- That is how we make our own tokenizer

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)
def process_content():

    try:

        for i in tokenized:

            words = word_tokenize(i)

            # one line code for POS tagging

            tagged = nltk.pos_tag(words)

            print(tagged)

    except Exception as e:

        print(str(e))
process_content()
# chunking

def process_content():

    try:

        for i in tokenized:

            words = word_tokenize(i)

            tagged = nltk.pos_tag(words)

            

            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""

            

            chunkParser = nltk.RegexpParser(chunkGram)

            chunked = chunkParser.parse(tagged)

            

            print(chunked)

            #chunked.draw()

            

    except Exception as e:

        print(str(e))
process_content()
# Chinking

# we chink something from a chunk

def process_content():

    try:

        for i in tokenized:

            words = word_tokenize(i)

            tagged = nltk.pos_tag(words)

            

            chunkGram = r"""Chunk: {<.*>+}

                            }<VB.?|IN|DT|>+{"""

            # <.*>+ = one or more of anything

            # <VB.?|IN|DT|>+ = one or more of Verb, Preposition or Determiner will be CHINKED OUT !!!

            

            chunkParser = nltk.RegexpParser(chunkGram)

            chunked = chunkParser.parse(tagged)

            

            print(chunked)

            #chunked.draw()

            

    except Exception as e:

        print(str(e))
# Named Entity Recognition



from nltk.tokenize import PunktSentenceTokenizer

from nltk.corpus import state_union

from nltk.tokenize import word_tokenize



train_text = state_union.raw("2005-GWBush.txt")

sample_text = state_union.raw("2006-GWBush.txt")



custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():

    try:

        for i in tokenized:

            words = word_tokenize(i)

            tagged = nltk.pos_tag(words)

            

            # Named Entity

            namedEnt = nltk.ne_chunk(tagged)

            namedEnt.draw()

    except Exception as e:

        print(str(e))
process_content()
# Lemaatizing

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("cats"))

print(lemmatizer.lemmatize("cacti"))

print(lemmatizer.lemmatize("geese"))

print(lemmatizer.lemmatize("rocks"))

print(lemmatizer.lemmatize("python"))

print(lemmatizer.lemmatize("went"))
# if we give it something and provide its POS tag, thing gets lemmatized

print(lemmatizer.lemmatize("better", pos="a"))

# a = adjective
print(lemmatizer.lemmatize("caresses"))

print(lemmatizer.lemmatize("goes"))

print(lemmatizer.lemmatize("recogination"))