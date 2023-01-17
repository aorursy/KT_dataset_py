import nltk

import re

import pandas as pd
doc = '../input/pg2680.txt'

with open(doc, encoding='utf-8-sig') as file:

    burgess = file.read()

print (burgess[:400])
# convert all letters to lowercase in order to standardize the text.

burgess = burgess.lower()



# here, we just split the whole text by spaces 

tokens = [word for word in burgess.split()]

print(tokens[:100])
tokens = nltk.word_tokenize(burgess)

print(tokens[:100])
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

tokens = tokenizer.tokenize(burgess)

print(tokens[:100])
token_freq = nltk.FreqDist(tokens)

token_freq["marcus"]
%matplotlib inline 



token_freq.plot(20, cumulative=False)
stop_words = set(nltk.corpus.stopwords.words('english'))

print(stop_words)
# filter each sentence by removing the stop words

def remove_stopwords(tokens):

    return [word for word in tokens if word not in stop_words]

        

print(remove_stopwords(tokens)[:100])
token_freq = nltk.FreqDist(remove_stopwords(tokens))

token_freq.plot(20, cumulative=False)
sents = nltk.sent_tokenize(burgess)

print(sents[:10])
import string



def remove_punctuation(text):

    text = "".join([word for word in text if word not in string.punctuation])

    tokens = nltk.word_tokenize(text)

    return tokens



print(remove_punctuation(burgess)[:100])
#-----PorterStemmer------

stemmer = nltk.stem.PorterStemmer()

print('##### Porter Stemmer #####')

print (stemmer.stem('running'))

print (stemmer.stem('runner'))

print (stemmer.stem('decreases'))

print (stemmer.stem('multiplying\n'))



#-----LancasterStemmer------

stemmer = nltk.stem.LancasterStemmer()

print('##### Lancaster Stemmer #####')

print (stemmer.stem('running'))

print (stemmer.stem('runner'))

print (stemmer.stem('decreases'))

print (stemmer.stem('multiplying\n'))



#-----SnowballStemmer------

# we need to specify language to initiate this stemmer

stemmer = nltk.stem.SnowballStemmer("english") 

print('##### Snowball Stemmer #####')

print (stemmer.stem("running"))

print (stemmer.stem('runner'))

print (stemmer.stem('decreases'))

print (stemmer.stem('multiplying\n'))



#-----WordNetLemmatizer------

lemmatizer = nltk.stem.WordNetLemmatizer()

print('##### WordNet Lemmatizer #####')

print(lemmatizer.lemmatize('running'))

print (lemmatizer.lemmatize('runner'))

print(lemmatizer.lemmatize('decreases'))

print(lemmatizer.lemmatize('multiplying\n'))
print(lemmatizer.lemmatize('playing', pos="v"))

print(lemmatizer.lemmatize('playing', pos="n"))

print(lemmatizer.lemmatize('playing', pos="a"))

print(lemmatizer.lemmatize('playing', pos="r"))
sents = nltk.sent_tokenize(burgess)

print (sents[10])

tokens = nltk.word_tokenize(sents[10])

nltk.pos_tag(tokens)
sentence = "Bill Gates, CEO of Microsoft Inc. is living in California."



tokens = nltk.word_tokenize(sentence)

# chunks = nltk.ne_chunk(nltk.pos_tag(tokens))

# chunks    
tokens = remove_punctuation(burgess)

bigrams = nltk.bigrams(tokens)

print(list(bigrams)[:50])
trigrams = nltk.trigrams(tokens)

print(list(trigrams)[:50])
fourgrams = nltk.ngrams(tokens, 4)

print(list(fourgrams)[:50])
sentence_1 = "I love studying AI"

sentence_2 = "I love studying statistics"

sentence_3 = "I love working as Data Scientist"



# so, lets respresent this in a numerical way

# We'll at first combine all these sentences and then tokenize and get distinct tokens.

# we basically combine the sentences with spaces between them.

text = " ".join([sentence_1, sentence_2, sentence_3])



# now we tokenize and get the distinct tokens as a list by using set() function.

tokens = list(set(nltk.word_tokenize(text)))

print(tokens)
from sklearn.feature_extraction.text import CountVectorizer



vect = CountVectorizer()



pattern = re.compile('(\r\n)+')  # Windows new line 

clean_text = re.sub(pattern, ' ', burgess)

clean_sent = nltk.sent_tokenize(clean_text)



# Now remove the punctuation and stopwords from each sentence

clean_corpus = []

for sent in clean_sent:

    words = nltk.word_tokenize(sent)

    remove_punc = [word for word in words if word not in string.punctuation]

    clean_corpus.append(' '.join([word for word in remove_punc if not word in stop_words]))



clean_corpus[:5]
vect.fit(clean_corpus)

matrix = vect.transform(clean_corpus)

print(vect.get_feature_names()[:50])
cv_df = pd.DataFrame(matrix.toarray(), columns=vect.get_feature_names())

cv_df.head()
cv_df.shape
from sklearn.feature_extraction.text import TfidfVectorizer



vect = TfidfVectorizer()

vect.fit(clean_corpus)

matrix = vect.transform(clean_corpus)

tf_df = pd.DataFrame(matrix.toarray(), columns=vect.get_feature_names())

tf_df.head()
# Now look at the first sentence by sorting the tfidf scores in a decreasing order

tf_df.iloc[10].sort_values(ascending=False).head(10)
import spacy



en = spacy.load('en')

doc = en('galaxies recede from us at speeds proportional to their distances, going faster the farther away they are.')



# now text is parsed by nearly all the basic NLP techniques and we are ready to extract what we need.

# tokenization 

for token in doc:

    print(token.text)
for token in doc:

    print(token.text, token.lemma_, token.pos_, token.tag_,

         token.dep_, token.shape_, token.is_alpha, token.is_stop, sep=" ==> ")
#lets view this parsing in Pandas dataframe



tokens = [token.text for token in doc]

lemmas = [token.lemma_ for token in doc]

pos = [token.pos_ for token in doc]

tag = [token.tag_ for token in doc]

dep = [token.dep_ for token in doc]

shape = [token.shape_ for token in doc]

is_alpha = [token.is_alpha for token in doc]

is_stop = [token.is_stop for token in doc]



pd.DataFrame({"tokens":tokens, "lemmas":lemmas, "pos":pos, "tag":tag, 

             "dep": dep, "shape":shape, "is_alpha":is_alpha, "is_stop":is_stop}, 

             columns=["tokens", "lemmas", "pos", "tag", "dep", "shape", "is_alpha", "is_stop"])
en = spacy.load('en')

doc = en("Steve Jobs, the CEO of Apple Inc. is living in San Francisco.")



for ent in doc.ents:

    print(ent.text, ent.label_)
# lets see how it works with the following sentences.



doc1 = "I have big exam tomorrow and I need to study hard to get a good grade."

doc2 = "My wife likes to go out with me but I prefer staying at home and studying."

doc3 = "Kids are playing football in the field and they seem to have fun"

doc4 = "Sometimes I feel depressed while driving and it's hard to focus on the road."

doc5 = "I usually prefer reading at home but my wife prefers watching a TV."



# array of documents aka corpus

corpus = [doc1, doc2, doc3, doc4, doc5]



tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

token_list = [tokenizer.tokenize(sentence.lower()) for sentence in corpus]



def remove_stopwords(words):

    return [word for word in words if word not in stop_words]



tokenized_data = [remove_stopwords(token) for token in token_list]

tokenized_data
from gensim import corpora, models



# Build a Dictionary - association word to numeric id

dictionary = corpora.Dictionary(tokenized_data)

 

# Transform the collection of texts to a numerical form

corpus = [dictionary.doc2bow(text) for text in tokenized_data]



# We are asking LDA to find 10 topics in the data

lda_model = models.LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)



for idx in range(10):

    # Print the first 10 most representative topics

    print("Topic #%s:" % idx, lda_model.print_topic(idx, 5))

new_sentence = "My wife plans to go out tonight"

lda_model.get_document_topics(dictionary.doc2bow(new_sentence.split()))