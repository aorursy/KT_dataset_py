import nltk
from nltk.tokenize import word_tokenize,sent_tokenize



example = "Hi there!! How are you doing. I am doing just fine. Hope you like this kernel."

# tokenizing each word in example text

print(word_tokenize(example))
# tokenizing each sentence in example text

print(sent_tokenize(example))
from nltk.corpus import stopwords



STOPWORDS = stopwords.words('english')

print(STOPWORDS)
example = "Hi there I am Sarthak Rana!! How are you doing. I am doing just fine. Hope you learn a lot from this kernel."



#convert to lowercase

example = example.lower()

print("Length of text before removing stopwords : {0}".format(len(example.split())))



# remove stopwords

[word for word in example.split() if word not in STOPWORDS]
print("Length of text after removing stopwords : {0}".format(len([word for word in example.split() if word not in STOPWORDS])))
example = ["Hi there I am Sarthak Rana!! How are you doing. I am doing just fine. Hope you like this kernel.", 

           "Today was a very great day. I got promoted. Might surprise my girl with a present :P"]



lowercase_example = []

for text in example:

    lowercase_example.append(text.lower())



[[word for word in sentence.split() if word not in STOPWORDS] for sentence in lowercase_example]
from nltk import pos_tag
# with single example

print("WITH SINGLE TEXT\n")

example = "John is an intelligent individual. He intelligently does smart work. He is a top performer at Google."

example = word_tokenize(example)

print(pos_tag(example))
# With multiple examples

print("\nWITH MULTIPLE TEXTS\n")

example_list = ["Hi there I am Sarthak Rana!! How are you doing. I am doing just fine. Hope you like this kernel.", 

               "Today was a very great day. I got promoted. Might surprise my girl with a present!!"]

for example in example_list:

    example = word_tokenize(example)

    print(pos_tag(example))
from nltk.stem import PorterStemmer



stemmer = PorterStemmer()

text = "John is an intelligent individual. He intelligently does smart work. He is a top performer in the company."

[stemmer.stem(word) for word in text.split()]
from nltk.stem import WordNetLemmatizer



lemmatizer = WordNetLemmatizer()

example = "John is an intelligent individual. He intelligently does smart work. He is a top performer at Google."

[lemmatizer.lemmatize(word) for word in word_tokenize(example)]
lemmatizer = WordNetLemmatizer()

example = "John is an intelligent individual. He intelligently does smart work. He is a top performer at Google."

example = word_tokenize(example)

lemmatized_tokens = []

for token, tag in pos_tag(example):

    if tag.startswith('NN'):

        pos = 'n'

    elif tag.startswith('VB'):

        pos = 'v'

    else:

        pos = 'a' 

    lemmatized_tokens.append(lemmatizer.lemmatize(token, pos))
lemmatized_tokens
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

            print(chunked)



    except Exception as e:

        print(str(e))



process_text()
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

            print(chunked)



    except Exception as e:

        print(str(e))



process_text()
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

            print(namedEnt)



    except Exception as e:

        print(str(e))



process_text()