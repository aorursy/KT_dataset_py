import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
example_text = "Hi , How are you doing today? You got a nice job at IBM. Wow thats an awesome car. Weather is great."
print(sent_tokenize(example_text))
print(word_tokenize(example_text))
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
print(stop_words)
example_text = "Hi Mr.Pavan , How are you doing today?. Cool you got a nice job at IBM. Wow thats an awesome car. Weather is great."
words = word_tokenize(example_text)
filtered_sentence = []
for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)
print(filtered_sentence)    
from nltk.stem import PorterStemmer
txt = "John is an intelligent individual.He intelligently does smart work. He is a top performer in the company."
sentences = sent_tokenize(txt)
stemmer = PorterStemmer()
new_sentence = []
for i in range(len(sentences)):
    words = word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words]
    new_sentence.append(' '.join(words))
print(new_sentence)
from nltk.stem import WordNetLemmatizer
txt = "John is an intelligent individual.He intelligently does smart work. He is a top performer in the company."
sentences = sent_tokenize(txt)
lemmtizer = WordNetLemmatizer()
new__lemmatize_sentence = []
for i in range(len(sentences)):
    words = word_tokenize(sentences[i])
    words = [lemmtizer.lemmatize(word) for word in words]
    new__lemmatize_sentence.append(' '.join(words))
print(new__lemmatize_sentence)
from nltk.tokenize import PunktSentenceTokenizer
# Now, let's create our training and testing data:
train_txt="Crocodiles (subfamily Crocodylinae) or true crocodiles are large aquatic reptiles that live throughout the tropics in Africa, Asia, the Americas and Australia. Crocodylinae, all of whose members are considered true crocodiles, is classified as a biological subfamily. A broader sense of the term crocodile, Crocodylidae that includes Tomistoma, is not used in this article. The term crocodile here applies to only the species within the subfamily of Crocodylinae. The term is sometimes used even more loosely to include all extant members of the order Crocodilia, which includes the alligators and caimans (family Alligatoridae), the gharial and false gharial (family Gavialidae), and all other living and fossil Crocodylomorpha."
sample_text ="Crocodiles are large aquatic reptiles which are carnivorous.Allegators belong to this same reptile species"
# Next, we can train the Punkt tokenizer like:
cust_tokenizer = PunktSentenceTokenizer(train_txt)
# Then we can actually tokenize, using:
tokenized = cust_tokenizer.tokenize(sample_text)
print("Speech Tagging Output")
def process_text():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)

    except Exception as e:
        print(str(e))

process_text()
from nltk.tokenize import PunktSentenceTokenizer

# Now, let's create our training and testing data:
train_txt="Crocodiles (subfamily Crocodylinae) or true crocodiles are large aquatic reptiles that live throughout the tropics in Africa, Asia, the Americas and Australia. Crocodylinae, all of whose members are considered true crocodiles, is classified as a biological subfamily. A broader sense of the term crocodile, Crocodylidae that includes Tomistoma, is not used in this article. The term crocodile here applies to only the species within the subfamily of Crocodylinae. The term is sometimes used even more loosely to include all extant members of the order Crocodilia, which includes the alligators and caimans (family Alligatoridae), the gharial and false gharial (family Gavialidae), and all other living and fossil Crocodylomorpha."
sample_text ="Crocodiles are large aquatic reptiles which are carnivorous.Allegators belong to this same reptile species"

# Next, we can train the Punkt tokenizer like:
cust_tokenizer = PunktSentenceTokenizer(train_txt)

# Then we can actually tokenize, using:

tokenized = cust_tokenizer.tokenize(sample_text)
print("Chunked Output")
def process_text():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk:{<NNS.?>*<JJ>+}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            #chunked.draw()
            print(chunked)

    except Exception as e:
        print(str(e))

process_text()
from nltk.tokenize import PunktSentenceTokenizer

# Now, let's create our training and testing data:
train_txt="Crocodiles (subfamily Crocodylinae) or true crocodiles are large aquatic reptiles that live throughout the tropics in Africa, Asia, the Americas and Australia. Crocodylinae, all of whose members are considered true crocodiles, is classified as a biological subfamily. A broader sense of the term crocodile, Crocodylidae that includes Tomistoma, is not used in this article. The term crocodile here applies to only the species within the subfamily of Crocodylinae. The term is sometimes used even more loosely to include all extant members of the order Crocodilia, which includes the alligators and caimans (family Alligatoridae), the gharial and false gharial (family Gavialidae), and all other living and fossil Crocodylomorpha."
sample_text ="Crocodiles are large aquatic reptiles which are carnivorous.Allegators belong to this same reptile species"

# Next, we can train the Punkt tokenizer like:
cust_tokenizer = PunktSentenceTokenizer(train_txt)

# Then we can actually tokenize, using:

tokenized = cust_tokenizer.tokenize(sample_text)

print("Chinked Output")
def process_text():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO>+{"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            #chunked.draw()
            print(chunked)

    except Exception as e:
        print(str(e))

process_text()
from nltk.tokenize import PunktSentenceTokenizer

# Now, let's create our training and testing data:
train_txt="Crocodiles (subfamily Crocodylinae) or true crocodiles are large aquatic reptiles that live throughout the tropics in Africa, Asia, the Americas and Australia. Crocodylinae, all of whose members are considered true crocodiles, is classified as a biological subfamily. A broader sense of the term crocodile, Crocodylidae that includes Tomistoma, is not used in this article. The term crocodile here applies to only the species within the subfamily of Crocodylinae. The term is sometimes used even more loosely to include all extant members of the order Crocodilia, which includes the alligators and caimans (family Alligatoridae), the gharial and false gharial (family Gavialidae), and all other living and fossil Crocodylomorpha."
sample_text ="Crocodiles are large aquatic reptiles which are carnivorous.Allegators belong to this same reptile species"

# Next, we can train the Punkt tokenizer like:
cust_tokenizer = PunktSentenceTokenizer(train_txt)

# Then we can actually tokenize, using:

tokenized = cust_tokenizer.tokenize(sample_text)

print("Named Entity Output")
def process_text():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged,binary = True)
            namedEnt.draw()
            print(namedEnt)

    except Exception as e:
        print(str(e))

process_text()
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import gutenberg
sample = gutenberg.raw("bible-kjv.txt")
tok = sent_tokenize(sample)
print(tok[5:15])